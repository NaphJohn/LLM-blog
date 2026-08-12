---
title: （九）Wan2.2-I2V/T2V-A14B：双专家 MoE 视频模型与 vllm-omni 引擎剖析
description: Wan2.2 双专家扩散 MoE 架构、I2V 图像条件注入、flow matching 去噪、双专家按时间步切换，以及 vllm-omni 0.21.0 的 TP4 张量并行与代码路径；附 I2V vs T2V 对照。
pubDate: 2026-08-11
series: vLLM 与 SGLang 框架解码手记
lang: zh
altLang: en
altHref: /en/
layout: ../../layouts/BlogPost.astro
---

> **文档信息** ｜ 更新时间：2026-08-11 ｜ 更新人：王航凯 ｜ 更新内容：模型架构 / 生成原理 / vllm-omni 代码路径 / T2V 架构对照 ｜ 版本：v0.1
> 配套文档：`Wan2.2-I2V-A14B-P800-vllm-omni-部署测试报告.md`（部署 / 性能 / 精度）。本文所有组件参数均来自 `69.75:/home/whk/Wan2.2-I2V-A14B-Diffusers` 权重的实际 config；所有代码路径均来自部署镜像 `vllm_omni_kunlun:py310_torch2.9_0.21.0` 容器内 `/workspace/vllm-omni` 源码实读。

## 一、总览

Wan2.2-A14B 是阿里 Wan 团队的视频生成模型（官方仓库），核心设计是**"双专家"扩散 MoE**：两个同构的 14B DiT（Diffusion Transformer），一个负责去噪前期（高噪声段，管全局构图与运动），一个负责后期（低噪声段，管细节精修），按去噪时间步在 `boundary_ratio` 处切换——任意时刻只有一个 14B 专家在跑（这就是名字里 "A14B" = Active 14B 的含义，总参数约 28B）。

I2V（图生视频）与 T2V（文生视频）共用这套骨架，区别只在输入条件：I2V 把参考图经 VAE 编码后作为逐步拼接的条件通道，让首帧被"钉死"为输入图。

## 二、Wan2.2-I2V-A14B 模型架构

> **模型架构与数据流总览**（参考图 → 20 通道 condition、噪声 latent、UMT5-xxl 文本编码、双专家 DiT 在 boundary_timestep 切换与 CFG 调度、40 步 UniPC flow matching、VAE 解码 → 81 帧 h264 mp4）：
>
> <img src="/wan22_i2v_arch.png" alt="Wan2.2-I2V-A14B 架构与数据流：双专家扩散 MoE（高/低噪声 DiT 各 14B，总参~28B，激活 14B），参考图经 VAE Encoder 编为 20 通道 condition（16 参考图 latent + 4 mask），初始噪声 latent 16 通道，两者每步拼接成 36 通道输入 DiT；prompt 经 UMT5-xxl text encoder（24 层 / d_model 4096）→ 4096 维 text embedding 通过 cross-attention 注入；DiT 每专家 3D RoPE self-attn + cross-attn + FFN，hidden 5120（40 头×128）；高噪声专家负责 t ≥ boundary_timestep（boundary_ratio=0.9，~1/4 步）CFG=3.5，低噪声专家负责 t < boundary_timestep（~3/4 步）CFG=2→3.5；40 步 UniPC flow matching 去噪后 VAE Decoder（slicing/tiling 省显存）→ 81 帧 832×480 RGB → mp4。T2V-A14B 差异：无 I2V 图像条件分支（DiT 输入 16ch 而非 36ch），boundary_ratio=0.875，flow_shift=12.0。" style="max-width:100%;border:1px solid #eaeaea;border-radius:8px;" />

Diffusers 布局（`model_index.json` 的 `_class_name = WanImageToVideoPipeline`，`boundary_ratio: 0.9`），五个组件：

| 组件 | 类 | 关键参数（权重实测 config） | 磁盘大小 |
|---|---|---|---|
| transformer（高噪专家） | WanTransformer3DModel | 40 层，40 头 × head_dim 128（hidden 5120），FFN 13824，in_channels=36，out_channels=16，patch_size=[1,2,2]，QK RMSNorm，cross_attn_norm | 54GB（fp32 存储，≈14B 参数） |
| transformer_2（低噪专家） | WanTransformer3DModel | 与 transformer 完全同构（独立权重） | 54GB |
| text_encoder | UMT5EncoderModel（UMT5-xxl encoder） | 24 层，d_model 4096，64 头，d_ff 10240，vocab 256384（多语言，中英 prompt 都行） | 11GB（≈5.7B） |
| vae | AutoencoderKLWan（Wan2.1 VAE） | z_dim=16，时间压缩 4×（temperal_downsample=[F,T,T]）、空间压缩 8×（dim_mult=[1,2,4,4]），带 16 维 latents_mean/std 归一化统计量 | 485MB |
| scheduler | UniPCMultistepScheduler | prediction_type=flow_prediction、use_flow_sigmas=true（flow matching），num_train_timesteps=1000，solver_order 2（bh2） | — |

要点：

1. **DiT 输入形状换算**。视频先过 VAE：81 帧 → (81-1)/4+1 = 21 个 latent 帧；832×480 → 104×60 latent。再过 patch_size=[1,2,2] 的 3D patch embedding → token 序列长度 = 21 × 52 × 30 = 32760 tokens（这就是视频扩散比 LLM 单 step 贵的原因：每个去噪步都是一次 3 万 token 的全量 forward × 40 步 × CFG 双分支）。
2. **in_channels=36 是 I2V 的身份证**：16（噪声 latent）+ 20（图像条件：4 通道 mask + 16 通道参考图 latent），见第三节。T2V 的 transformer 是 in_channels=16。
3. 每层 = self-attention（3D RoPE 位置编码，rope_max_seq_len=1024）+ cross-attention（对 UMT5 文本 embedding，text_dim=4096）+ FFN；时间步经 freq_dim=256 的正弦编码进 AdaLN 调制。
4. 权重以 fp32 存储（所以落盘 118GB），部署时 `--dtype bfloat16` 加载减半。

## 三、生成原理

### 3.1 Flow matching 去噪

Wan2.2 用 rectified flow / flow matching 训练（不是经典 DDPM ε-prediction）：模型学习从噪声分布到数据分布的"直线"速度场，调度器按 flow_prediction 解释模型输出。采样用 UniPC 二阶多步求解器，40 步走完。

`flow_shift`（sample_shift）控制时间步 sigma 的重分布：shift 越大，采样步越向高噪声段倾斜（大构图阶段分配更多步数）。I2V 官方值 5.0，T2V 480p 官方值 12.0（`wan_i2v_A14B.py` / `wan_t2v_A14B.py`）。注意权重里 `scheduler_config.json` 写的是 3.0，vllm-omni 不用它——由服务端 `--flow-shift` 参数经 `resolve_wan_flow_shift()` 重建调度器（见 4.4）。

### 3.2 双专家按时间步切换（MoE）

`boundary_timestep = boundary_ratio × num_train_timesteps(1000)`，I2V 官方 `boundary_ratio=0.9` → 边界 t=900：

- `t ≥ 900`（去噪前期约前 1/4 步数）：走 transformer（高噪专家），CFG 用 guidance_scale；
- `t < 900`（其余步）：切到 transformer_2（低噪专家），CFG 用 guidance_scale_2。

I2V 官方 guidance 为 (3.5, 3.5)。两个专家各自独立做 classifier-free guidance（正负 prompt 两次 forward 取差），所以每步实际是 2 次 14B forward。

### 3.3 I2V 图像条件注入（与 T2V 的本质差异）

参考图不走独立 image encoder（`model_index.json` 里 `image_encoder: null`），而是直接用 VAE 把首帧编码进 latent 空间做通道级拼接：

1. 参考图 resize 到目标分辨率，拼成"首帧 + 80 帧全零"的伪视频，过 VAE encode → 16 通道 latent_condition，用 latents_mean/std 归一化；
2. 构造 4 通道 mask（首帧位置=1，其余=0，按时间压缩率 4 折叠），与 latent_condition 拼成 20 通道 condition；
3. 每个去噪步把 `[latents(16), condition(20)]` 沿通道 cat 成 36 通道输入送给 DiT——参考图信息在全部 40 步持续注入，而非只做初始化。这保证了首帧与输入图一致、后续帧围绕它演化。

（对应部署报告中的实测：I2V 比 T2V 慢 14~17%，主要就是这条条件分支 + VAE encode 的开销。）

### 3.4 VAE 解码

去噪完的 16×21×104×60 latent 反归一化后过 AutoencoderKLWan decode → 81×480×832 RGB 帧，再由服务端编码为 h264 mp4。部署时开 `--vae-use-slicing --vae-use-tiling`（分片/分块解码），显著降低解码阶段峰值显存。

## 四、vllm-omni 0.21.0 中走的代码

以下路径均相对容器内 `/workspace/vllm-omni`，行号为该镜像源码实测。

### 4.1 请求入口（API 层）

- `POST /v1/videos`（异步任务）与 `POST /v1/videos/sync`（同步）共用表单解析 `vllm_omni/entrypoints/openai/api_server.py:2511 _parse_video_form()`：文件字段 `input_reference`（UploadFile）或字符串字段 `image_reference`（URL/base64 JSON），二者互斥（2547 行显式 400 拒绝同时传）。这就是部署报告附录A-1"字段名不是 image"的出处。
- 图像经 `decode_input_reference()` 解码为 PIL，塞进请求的 `multi_modal_data["image"]`，与 prompt、sampling 参数（width/height/num_frames/steps/guidance/boundary_ratio/seed…）一起构造 `OmniDiffusionRequest`，交给 AsyncOmni 引擎排队。

### 4.2 模型识别与 pipeline 路由

- 服务启动时 `vllm_omni/entrypoints/utils.py` 的 `resolve_model_config_path()` 只认 HF transformers 或 Diffusers 布局——Wan 原生格式（high_noise_model/ 布局）没有 `model_index.json`，在这里直接抛 `ValueError: Could not determine model_type ...`（附录A-5 的报错源头）。
- `vllm_omni/diffusion/data.py:808` 读权重的 `model_index.json`，取 `_class_name = "WanImageToVideoPipeline"`；
- `vllm_omni/diffusion/registry.py:114` 把它映射到 vllm-omni 自己的实现类 `Wan22I2VPipeline`（`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py:137`），并挂上 `get_wan22_i2v_pre_process_func`（registry.py:498，请求预处理）和 `get_wan22_i2v_post_process_func`（registry.py:459，帧后处理/编码）。

即：vllm-omni 不用 diffusers 的 `WanImageToVideoPipeline` 原实现，而是同名语义的自研 pipeline（为了 TP 并行、CFG 并行、cache 后端等）。

### 4.3 Pipeline 主流程（pipeline_wan2_2_i2v.py）

`Wan22I2VPipeline.forward()`（376 行起）按序做：

1. 取参：prompt / image（从 multi_modal_data）/ width / height / num_frames / steps / guidance。`guidance_scale` 给高噪专家，`guidance_scale_2` 给低噪专家（444-450 行）。
2. `boundary_ratio` 解析（452-455 行）：

   ```python
   boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else req.sampling_params.boundary_ratio
   if boundary_ratio is None:
       boundary_ratio = 0.875
       logger.warning("boundary_ratio is required for I2V generation. using default value 0.875")
   ```

   优先级 = 服务端 `--boundary-ratio` > 请求参数 > 写死的 0.875 兜底。它从头到尾不读 `model_index.json` 里的 0.9——这就是附录A-2 坑（I2V 必须显式传 0.9）的代码根源。`__init__`（242 行）里还有个优化：`boundary_ratio=1.0/0.0` 时只加载单个专家。
3. `check_inputs`（863 行）：宽高必须被 16 整除（patch_size 2 × vae_scale_factor_spatial 8），否则 `ValueError: height and width have to be divisible by 16`（附录A-6，854×480 压测失败的原因；T2V pipeline 无此校验）。
4. `encode_prompt`（685 行）：UMT5-xxl 编码正/负 prompt。
5. `prepare_latents`（760 行）：生成初始噪声 latent（seed 由 generator 控制）+ 按 3.3 节构造 20 通道 condition（VAE encode 参考图 + mask 折叠 + latents_mean/std 归一化，代码 820-833 行）。
6. `diffuse` 去噪循环（272 行）：

   ```python
   current_model = self.transformer
   current_guidance_scale = guidance_low
   if boundary_timestep is not None and t < boundary_timestep and self.transformer_2 is not None:
       current_model = self.transformer_2
       current_guidance_scale = guidance_high
   ...
   latent_model_input = torch.cat([latents, condition], dim=1)   # 16+20=36 通道
   noise_pred = self.predict_noise_maybe_with_cfg(...)           # CFG 正负双分支
   latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
   ```

   其中 `boundary_timestep = boundary_ratio * scheduler.num_train_timesteps`（541-543 行）——专家切换、逐步条件拼接、CFG 全在这一个循环里。
7. VAE decode + 后处理：latent 反归一化 → `AutoencoderKLWan.decode`（受 `--vae-use-slicing/--vae-use-tiling` 控制）→ `get_wan22_i2v_post_process_func` 把帧转 uint8 → API 层编码 mp4 返回。

### 4.4 调度器与 flow-shift

`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py:44 build_wan_scheduler(sample_solver, flow_shift)` + 71 `resolve_wan_flow_shift()`：按服务端 `--flow-shift`（本部署 5.0）与请求参数重建 `UniPCMultistepScheduler`，覆盖权重目录里 `scheduler_config.json` 的 3.0。

### 4.5 TP4 张量并行（wan2_2_transformer.py）

vllm-omni 把 diffusers 的 `WanTransformer3DModel` 重写成 vLLM 风格的真张量并行（这是"4 卡协同一个实例"而非文生图那种多实例 round-robin 的原因）：

- self-attention 用 `QKVParallelLinear`（366 行）+ `RowParallelLinear` 输出投影（387 行），40 个头按 TP4 切成每卡 10 个头；
- cross-attention 的 q/k/v 用 `ColumnParallelLinear`（476-497 行）；
- FFN `WanFeedForward`（103 行）：`ColumnParallelGELU`（列切）+ `RowParallelLinear`（行切 + all-reduce），13824 中间维每卡 3456；
- QK 归一化用自研 `DistributedRMSNorm`（40 行）：head 维被切开后先 `tensor_model_parallel_all_reduce` 汇总平方和再归一（61 行）。

每卡持有两个专家各 1/4 的权重 + 完整 text_encoder/VAE，对应实测每卡 58.6~64.5GB 显存。`--cache-backend cache_dit` 在此之上做 DiT 特征缓存（跨步复用变化小的中间激活）加速去噪。

### 4.6 一次请求的完整链路（串起来）

```
curl -F input_reference=@img -F prompt=... /v1/videos/sync
  → api_server.py:_parse_video_form (2511)         # 表单/图像解析
  → OmniDiffusionRequest → AsyncOmni 引擎队列
  → registry: model_index._class_name "WanImageToVideoPipeline" → Wan22I2VPipeline
  → forward(): 参数/boundary_ratio 解析 → check_inputs(÷16)
  → UMT5 encode_prompt → VAE encode 参考图 (prepare_latents, 20ch condition)
  → diffuse: 40 步 × [t≥900: transformer | t<900: transformer_2] × CFG，
             每步 cat([latents16, condition20]) → 36ch，TP4 并行 forward
  → UniPC scheduler step (flow_prediction, shift=5.0)
  → VAE decode (slicing/tiling) → post_process → h264 mp4 响应
```

## 五、Wan2.2-T2V-A14B 架构与原理

T2V 与 I2V 是同一骨架的姊妹模型，vllm-omni 里对应 `Wan22Pipeline`（`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py:300`，`model_index.json _class_name=WanPipeline`）：

- 同构双专家：也是 2 × 14B `WanTransformer3DModel`（40 层 / 5120 hidden / FFN 13824），同样按 `boundary_timestep` 在 diffuse 循环里切换（483 行），同样的 UMT5-xxl + Wan2.1 VAE + UniPC flow matching。`boundary_ratio` 解析逻辑一字不差（618-623 行，同样兜底 0.875 + WARNING——T2V 恰好官方值就是 0.875，所以 T2V 部署感知不到这个坑）。
- 无图像条件分支：transformer in_channels=16（纯噪声 latent），`prepare_latents` 只生成噪声，diffuse 每步直接喂 16 通道，没有 mask/condition 拼接，也没有 VAE encode 参考图这一步。
- 官方采样参数不同（`wan_t2v_A14B.py`）：boundary=0.875、sample_shift=12.0（480p；720p 用 5.0）、guidance (4.0, 3.0)——T2V 高噪段用更强 CFG 抓构图，低噪段放松；I2V 因为有参考图锚定构图，两段都用温和的 3.5。
- 无 ÷16 校验：T2V `check_inputs` 不查宽高整除性（所以 T2V SOP 用 854×480 能跑）。
- 历史差异：T2V SOP 时代用的 ModelScope 原生权重需要 text_encoder key remap（Wan 原生 key → HF UMT5 key）和 VAE latents_mean/std 修复；I2V 直接用 HF 官方 Diffusers 权重，这些问题不存在（本文第二节的 VAE 统计量就是当年 T2V 修复时写入的那组官方值）。

### I2V vs T2V 差异速查

| 维度 | I2V-A14B | T2V-A14B |
|---|---|---|
| pipeline 类（vllm-omni） | Wan22I2VPipeline（pipeline_wan2_2_i2v.py） | Wan22Pipeline（pipeline_wan2_2.py） |
| model_index _class_name | WanImageToVideoPipeline | WanPipeline |
| transformer in_channels | 36（16 latent + 4 mask + 16 图像 latent） | 16 |
| 条件注入 | 参考图 VAE latent 逐步通道拼接 | 仅文本 cross-attention |
| boundary_ratio（官方） | 0.9 | 0.875 |
| flow_shift（官方，480p） | 5.0 | 12.0 |
| guidance（官方） | (3.5, 3.5) | (4.0, 3.0) |
| 宽高校验 | 必须 ÷16 | 无 |
| 请求必传 | input_reference（图） | 仅 prompt |
| 实测时延（P800 TP4 同负载） | ×1.14~1.17 | 基准 |

## 六、参考

- Wan2.2 官方仓库：https://github.com/Wan-Video/Wan2.2（I2V/T2V 配置：wan/configs/wan_i2v_A14B.py、wan_t2v_A14B.py）
- HF 权重：Wan-AI/Wan2.2-I2V-A14B-Diffusers、Wan-AI/Wan2.2-T2V-A14B-Diffusers
- vllm-omni 源码：部署镜像内 /workspace/vllm-omni（vllm 0.21.0 / diffusers 0.38.0 昆仑 XPU 版）
- 部署 / 性能 / 精度实测：本目录 `Wan2.2-I2V-A14B-P800-vllm-omni-部署测试报告.md`
