---
title: vLLM & SGLang 社区跟踪 · 2026-08-11：vLLM v0.27.0 一次性交付、SGLang 统一内存池三连修与 PD 分离打通
description: LLM Infra Daily 第 8 期。窗口 08-10~08-11：vLLM 发布 v0.27.0（561 commits，Kimi K3 全栈 / PyTorch 2.13 环境级升级 / MRv2 扩到非生成式负载 / Rubin sm_107）；SGLang 在统一内存池新数据面上一天内连修三处静默正确性缺陷，并同时打通 PD 分离与 DSPARK 投机解码。本期主线——一边「交付」，一边「还债」。
pubDate: 2026-08-12
series: 社区跟踪手记
lang: zh
altLang: en
altHref: /en/blog/tr3-vllm-sglang-20260811
layout: ../../layouts/BlogPost.astro
---

## ★ 今日最值得关注

**SGLang #33974「统一池支持 DSPARK 投机解码 + 修两个 NaN 根因」**——它在一个 PR 里同时展示了两种层级完全不同的静默损坏，且两者都是同一种经典形态：A 模块的隐含假设被 B 模块打破，中间没有任何断言。理解了它，你就有了一套可迁移到任何框架的自查方法（详见下方「原理 / 代码解读」）。

## 一、今日总览

- **头条**：vLLM 发布 **v0.27.0**（08-10 21:18 UTC，561 commits / 242 贡献者 / 64 位新面孔），距上一版 v0.26.0（07-27）14 天。SGLang 仍为 **v0.5.17**（08-08）。Step 系列无新发布。
- **主干活跃度**：vLLM main 51 commits、SGLang main 66 commits（同窗口）。
- **本期主线：一边「交付」，一边「还债」。**
  - vLLM 用一个 561-commit 的大版本把两周积累一次性交付（Kimi K3 全栈、PyTorch 2.13 环境级破坏性升级、MRv2 扩到非生成式负载、Rubin sm_107）；
  - SGLang 则在 main 上一天内连修统一内存池（unified memory pool）这条新数据面上的三处静默正确性缺陷。
  - 合起来看是一个清晰信号：**新数据面正在把 KV cache 早期踩过的静默 bug 重演一遍**——虚拟/物理 id 分离、页信封布局、可搬移页，每一个都在制造「不报错、但答案是错的」。

## 二、vLLM 要点

### 版本发布 v0.27.0（08-10 21:18 UTC）

561 commits / 242 贡献者（64 新面孔）。核心内容：

- **Kimi K3 全栈一版落地**：模型与内核（#50089/#50000）、Python（#50093）与 Rust（#50104）前端、AttnRes 内核（#50090）、DeepGEMM（#50458）、compressed-tensors 量化权重（#50500）、DSpark AR 融合（#50242）、共享专家可切分而非复制（#50656）。
- **依赖大升级（破坏性环境变更）**：PyTorch 2.13.0 + torchvision 0.28.0 + Triton 3.7.1（#48155），XPU（#48677）与 CPU（#50412）同步跟进；Transformers 5.14.1、FlashInfer 0.6.16.post3、AITER 0.1.19、NCCL 2.30.7（镜像内启用 DeepEPv2）、Helion 1.4.0。
- **FlashAttention 4 在 SM100 上加深**：FP8 KV cache（#42569）、headdim-256（#42669），配套新的 JIT warmup 基础设施（#47451）与 runner 自有 Triton 内核预热（#49903），消除首请求编译停顿。
- **Model Runner V2 扩到非生成式负载**：encoder-only attention（#49331）、embedding/分类的序列池化（#48791）、encoder token 分类（#50293）与 embedding（#50574）、BGE-M3 池化（#50661）、CPU 多模态（#50073）、多层 MTP speculator（#48892）。
- **大规模服务韧性**：DP+EP 外部负载均衡部署的（简化版）容错框架（#44428）、弹性 EP 扩缩容的异步准备（#47288）。
- **下一代硬件早期启用**：NVIDIA Rubin 的 sm_107 目标（#49387）+ SM107 上的 NVLink all-reduce（#49647）；ROCm gfx1250（#46516）。
- **移除项**：模型 Plamo2（#49729）、Ouro（#49786）；参数 max_num_partial_prefills / max_long_partial_prefills（#49244）。

> **对部署的影响**：这是一次**环境级而非包级**升级——torch 2.13 意味着不能原地 `pip install -U vllm`，必须重建镜像/虚拟环境，并回归所有自编译的 kernel 插件与 torch 扩展。建议按「重建镜像 → 冒烟 → 精度回归 → 灰度」排期，不要直接滚生产。

### Bug 修复·正确性 #51727 DeepSeek V4/3.2 词表尺寸多算，导致结构化输出崩溃

`DeepseekV4Tokenizer.__len__` 返回 `vocab_size + len(get_added_vocab()) = 128000 + 1283 = 129283`，但那 1283 个 added tokens 里有 3 个 id 本来就小于 128000（已在基础词表内），真实词表恰为 `config.vocab_size = 129280`——多算 3 个直接把 guided decoding 打崩（issue #50924 / #51467）。

历史成因：这段 override 写于「HF fast tokenizer 的 `len()` 尚不含 added token」的年代；transformers 5.x 的 `TokenizersBackend` 已经包含了——于是它既冗余又错误（`deepseek_v32.py` 里同款陈旧模式一并删除，净 −11 行）。

影响：跑 `DSV4-Flash + --structured-outputs-config.backend guidance` 的服务会起不来。修复在 v0.27.0 之后合入 main，本版不含。

### 性能提升 #51430 收窄 DeepSeek V4 的 eager CUDA graph 区域 → 短 prefill TTFT −53.7%

把 Q 上投影、fused Q norm/RoPE/KV 写入、indexer 输入准备、MLA/indexer 压缩这些「生产者」操作从 eager 区搬回 forward 让 CUDA graph 捕获，eager 区只留 sparse indexer + forward_mqa。

实测（DeepSeek-V4-Flash / 4×GB200 / Rust 前端 / FP8 KV / FP4 indexer cache / PIECEWISE breakable graph / DeepGEMM mega-MoE / MTP2；1000 请求、128 输入 token、并发 1）：

| 指标 | main | 本 PR | 变化 |
|------|------|-------|------|
| Mean TTFT | 41.45 ms | 19.20 ms | **−53.7%** |
| P50 TTFT | 41.18 ms | 19.09 ms | **−53.7%** |
| 请求吞吐（并发 64 饱和） | 430.05 req/s | 433.29 req/s | +0.75% |

GSM8K 1319 题 5-shot：TP4/EP4+MTP2 95.30%、DP4/EP4+MTP2 95.07%，无精度退化。

> ⚠️ 关键上下文：PR 明确写了「此前一次同类的区域收窄改动，曾因真实模型输出损坏被回滚」。这一版刻意把 `forward_mqa` 与 sparse indexer 一起留在 eager 区（而不是把整个 attention backend 捕获进去），并用持久化的 eager scratch 保证被 eager 区消费的张量地址稳定。
> 影响：短 prefill / 低并发交互场景收益极大；这是 128 输入 token 的测试，长 prefill 场景收益不会同比例放大（TTFT 里 kernel launch 开销占比会被真实计算稀释）。

### 安全 #49948 音频采样率伪造绕过解码时长守卫 → 单请求可打 11.7 GiB 内存，OOM 打死 API server

`load_audio_soundfile` 里的 `max_duration_s` 守卫用 `f.frames / f.samplerate` 算时长，完全信任容器头。攻击者把 `samplerate` 设为 655350（FLAC 上限）配 8 声道，就能让上亿帧「看起来」像一段短音频从而绕过时长检查，而 `f.read()` 会实际分配最高 11.7 GiB 的 float32 PCM。

修法：新增 `VLLM_MAX_AUDIO_DECODE_BYTES`（默认 256 MiB），独立于采样率地约束估算 PCM 缓冲大小——soundfile 路径预读检查 `frames × channels × 4`，PyAV 路径解码中增量计字节。两道守卫各有分工：时长守卫早期拦合法长文件，字节守卫无视声称时长拦伪造。

影响：任何对公网暴露音频 / 多模态端点的服务都应关注。这是继 SGLang 0.5.5–0.5.12 多模态 path traversal 之后，多模态入口的又一类资源型 DoS——**多模态入口正在成为 LLM 服务的新攻击面**。

### 新特性 / PD #48414 KV 卸载采用「规范化 CPU 布局」，与并行拓扑彻底解耦

叠在 #48408（逐层规范 KV 页映射）之上。把卸载的 KV 按与并行度无关的规范布局存放：每个 worker 把自己的页片段 scatter 到整个 worker 组共享的 CPU 区域中的规范位置。

```
legacy row:     [worker0: T0|T1][worker1: T0|T1]...   每 worker 各占槽位
canonical row:  [T0 规范页][T1 规范页]                 全 head + 全 token，只存一份
```

MLA latent 与被复制的 GQA head 只存一份而非每 rank 一份（非写者跑空 store）。配置 `kv_connector_extra_config: {"canonical_layout": true}`；在无法认证的配置上请求它会在启动时直接失败，而不是静默降级（这个设计取向值得学）；持久化格式身份（v1-nhd/v1-hnd）纳入 FileMapper 命名空间，杜绝规范 / 传统 / 跨家族字节解析到同一批文件。4×H100 端到端：tp1 传统、tp1/tp2 规范、tp4+dcp2 规范，逐 token 精确。

影响：这是「KV 缓存跨拓扑复用」的关键一步——此前 TP4 落盘的 KV 换成 TP2 就作废，规范布局后可跨并行拓扑共享与迁移，对多集群 / 弹性扩缩容 / 冷启动预热的成本模型影响很直接。

### 其余值得一提

- #47352 MRv2 的 MTP 在草稿步之间共享 topk index buffer（DSV3.2-NVFP4 / TP4，输出吞吐 1629.90 → 1640.71 tok/s，约 +0.66%；更大的意义是给 `AutoRegressiveSpeculator` 引入了回调协议 `AutoRegressiveSpeculatorCallbacks`，让模型专有优化不再污染通用投机框架）
- #50484 Kimi-K3 支持 DCP
- #51602 修 DSpark 的 `parallel_drafting_token_id` 初始化 bug
- #51507 top-k/top-p Triton 采样内核改用 8 warps
- #49436 状态拷贝 Triton 内核的 3D 网格分块
- #51265 Ling-3.0-flash-fp8 支持
- #51178 Rust gRPC 显式 DP rank 路由
- #51573 修 YAML 配置里 `false` 型 `BooleanOptionalAction` 未发 `--no-{key}`

## 三、SGLang 要点

本日 SGLang 只有一条主线：**统一内存池（`--enable-unified-memory`）**。这条为混合 Mamba 模型（Kimi-Linear / Kimi-K3）设计的「KV + mamba state 同池」新数据面，在 24 小时内被连修三处静默正确性缺陷、并同时打通 PD 分离与投机解码。无新版本 tag，稳定线仍是 v0.5.17（08-08）。

### 重要 PR · 正确性 #33974 统一池支持 DSPARK 投机解码 + 修两个 NaN 根因　★ 今日最值得关注

13 文件 +661/−21，08-10 17:35Z 合入。此前 `server_args.py` 里一句 `assert self.speculative_algorithm is None` 硬性禁止统一池搭配任何投机解码；这个 PR 把链式 DSPARK 接进来，并在 Kimi-K3（TP8、2×4 GB300）与 Kimi-Linear-48B（TP1×8 B300 级）上端到端验证时，挖出两个层级完全不同的真实根因（详见下方「原理解读」）。

精度对照最刺眼的一行：干净启动 + 只修零化，GSM8K 掉到 0.120，而 accept rate 显示 0.52（假的）；两个修复都上之后恢复 0.985 / 0.990。

EAGLE / tree / DFLASH / NGRAM 仍被禁止，待各自完成 virtual-vs-dense loc 审计。

### Bug 修复·正确性 #33517 统一池 × Triton 后端 × 确定性推理 三者同开 → NaN logits（且本地静默）

根因：`--enable-deterministic-inference` 会把 Triton extend 路由到 `TritonAttnBackend._forward_extend_unified`——一个单阶段内核，prefix 和 extend 两半 KV 都从池里读（默认的两阶段路径只从池读 prefix，extend 半段直接取自 k/v 入参）。而传给它的 `forward_batch.out_cache_loc` 没做翻译：统一池里它是虚拟 id，而 prefix 的 `kv_indices` 已在 `init_forward_metadata` 里翻成物理 id → prefix 按物理读、刚写入的 extend token 按虚拟读，错位数据经 LM head 变成 NaN。

三缺一都不复现（不开确定性→走两阶段内核不从池读 extend；不开统一池→out_cache_loc 本就是物理的；不用 Triton→别的后端不走这个内核），所以现有测试全部漏网。

> ⚠️ 最阴的一点：只有 armed `SGLANG_ENABLE_ASYNC_ASSERT`（CI 会开）才会 abort 报「NaN detected! sampler: next_token_logits」；本地不开这个标志，任务会正常跑完并返回有限的 top-k logprob——**完全静默**。

### 新特性 / PD 分离 #33362 统一内存池支持 PD 分离（kimi-linear MLA hybrid-Mamba）

15 文件 +835/−51，叠在 #33517 之上（否则测试配置会踩上面那个 NaN）。

矛盾在哪：统一池在分配器之上到处存虚拟 id、按 page-major「信封」布局把各层数据交错存放、压缩时会物理搬移页；而 PD 传输引擎（mooncake RDMA）是在任何 CUDA stream 序之外按 `ptr + index * item_len` 寻址裸字节的。三条假设条条冲突。

解法三件套：

1. **传输方案改为「整信封 + 物理 id」**：`get_contiguous_buf_infos()` 只注册一个区域（裸字节缓冲），`item_len` = 一个 page 全部层的信封字节（page-major 构造保证连续）—— 一次 RDMA 块搬走该页的每一层；mamba 侧同理，`index` 为物理 mamba 槽。逐张量的 dim / layer-id 元数据刻意留空（信封不能按 TP 重切也不能按 PP 取子集），`maybe_send_extra` 在 item 长度不匹配时响亮失败，专门捕获「统一池↔非统一池混用」与「attn TP 不等」。
2. **虚拟→物理翻译收敛到交接点**：新增 `translate_kv_indices_for_transfer()`（基类恒等，统一分配器覆写），在 prefill 的 `send_kv_chunk` 与 decode 的 `pop_preallocated` 各调一次。
3. **压缩安全**：PD + unified 强制惰性压缩（断言拦截；即时释放路径的压缩会在 RDMA 在飞时搬页）；新增 `disagg_move_gate`，只要可能有传输在飞就一页不搬。

### 性能 / PD 分离 #34191 PD 的 prefill 服务器跳过投机验证 scratch —— 每 rank 省下约 24 GB 死重

只改 3 文件 +23/−2，收益却极大。PD 分离下 prefill 服务器根本不跑 `TARGET_VERIFY`（server_args 早就以这个理由拒绝 `--enable-linear-replayssm-spec`），但在开了投机解码的混合线性注意力模型（GDN/KDA）上，它仍在为整套 verify 机制买单：

- **池**：分配仅 verify 用的逐草稿 token mamba 状态快照，`intermediate_ssm_state_cache` ≈ `num_draft_tokens × 整个 mamba 池`。256 槽池（每 rank ~6 GB ssm_state）+ `num_draft_tokens=4`（NEXTN 3 步）= 每 rank ~24 GB 白占——单这一项就足以把深 PP 的 prefill 布局在 CUDA graph 捕获时推进 OOM。
- **图**：捕获永不重放的 target-verify CUDA graph。
- **预热**：dummy forward 跑在 `TARGET_VERIFY` 模式。

三处一起按 `disaggregation_mode == "prefill"` 门控。实测 Qwen3.5-397B-A17B-FP8、1P1D（TP4 prefill / TP4 decode，mooncake，两侧 NEXTN `num_steps=3, eagle_topk=1, num_draft_tokens=4`）、GSM8K 200 题 8-shot：准确率 0.99 vs 0.98、MTP accept length 3.474 vs 3.481（均在采样噪声内），prefill 的 `intermediate_ssm_state_cache` 从 4.31 GB → 不分配。

### Bug 修复·正确性 #31700 DeepSeek-V4/V4-Pro 的 DP-attention gather 语义错误 → 隐藏态被放大 attn_tp_size 倍

触发条件：`moe_a2a_backend=none + data_parallel_size>1 + attn_tp_size>1`。

根因：MoE gather 执行时，`self_attn` 已经在 attention-TP 组内归约过了，隐藏态是复制的；但现有 `dp_gather_partial` 把它们当作未归约的部分贡献，其 reduce-scatter 会把复制值再求和一次——每一个 MoE 层都把隐藏态幅值乘以 `attn_tp_size`。NextN 的 input-id gather 同理。

修法：两处改用 `dp_gather_replicate`；`input_ids` 在两次 replicate gather 前 clone（MAX_LEN 实现会在非 leader 的 attn-TP rank 上把本地输入清零，而 `input_ids[:, None]` 会别名调用方张量）。隐藏态刻意不 clone（gather 后即死，每层克隆大激活得不偿失）。

复现规模：2×8 H200（TP16 / DP4 → attn TP4×DP4）+ DeepSeek-V4-Pro-NVFP4；TP8×DP2 与 2×8 H100 上独立复现。

### 其余值得一提

- #34240 DCP 在 MLA target-verify 路径去掉两次 per-layer launch
- #33662 DSV4 EAGLE prefill 去 host 同步
- #34167 修 DSA top-k v2 在 CUDA 13.1+ 上丢弃非主 rank 输出
- #33639 HiCache 支持 Unified Radix Cache 里的 Mamba 分支（Host 侧节点做按分量的增量备份，不再重复整拷完整 KV）
- #30392 多模态全局缓存与 Mooncake 解耦（存储后端可插拔，无 Mooncake 环境也能用）
- #34234 DFLASH 草稿 KV 池按自身注意力几何算预算（此前沿用 EAGLE 的「草稿与目标同构」假设，而 DFLASH 是独立训练模型：8 KV heads 时估算误差 −11.9%，低估意味着池子发出去的 token 数超过两个池的实际容量）
- #33912 DFLASH 草稿 KV 池尺寸计入 DCP
- #31847 投机解码支持 Inkling DSPARK
- #33484/#33085 ROCm 上 hisparse swap-in 拷贝融合与 128-bit 非临时拷贝
- 扩散侧 15+ 提交（Z-Image 单卡 BCG 修复且与 eager 位精确、LTX-2 quality=high 融合 RMSNorm+modulate，H200 ltx23 单阶段去噪 45.85→43.24s）

## 四、原理 / 代码解读：SGLang #33974 的两个 NaN 根因

选它，是因为它在一个 PR 里同时展示了两个层级完全不同的静默损坏，而且两者都是同一种经典形态：**A 模块的隐含假设被 B 模块打破，且中间没有任何断言**。理解了它，你就有了一套可迁移到任何框架的自查方法。

### 背景：什么是「统一内存池」

Kimi-Linear、Kimi-K3 这类混合模型同时需要注意力 KV 和 Mamba state 两种缓存。传统做法是两个静态池各自 `torch.zeros` 预分配，比例一旦定死就浪费。统一池把它们放进同一个字节池，按 page-major「信封」布局把各层数据交错，并引入虚拟 id ↔ 物理 id 两层地址（分配器之上一律用虚拟 id，物理页可以在压缩时搬家）。

灵活性的代价：凡是绕过分配器直接摸物理内存的东西（内核、RDMA、CUDA graph 捕获），都必须显式做地址翻译；漏掉一处不会报错，只会读到别人的字节。

### 根因 1：分配器发页时不清零 → NaN 从历史字节渗进注意力输出

trtllm-gen 的 MLA 内核是整页读的，超出 seq_len 的行靠算术掩码抹掉。问题是算术掩码对 NaN 不安全—— `NaN × 0` 仍是 NaN，一旦参与后续求和就把整行污染掉。而统一池只在启动时把裸缓冲区清零一次；页被回收再发出去时就重新暴露历史字节。一个请求最后那个部分填充页的尾部如果恰好躺着 NaN 位模式，就会渗进注意力输出。

投机解码放大了暴露面：每个 verify 窗口都落在序列尾部的一张新页上——这就是为什么「加上投机解码之后才炸」。

静态池天然免疫（`torch.zeros` 建池，且只写入有限的 KV）。修法就是在分配边界给共享池同样的保证：每发出一页，就在调度流上做一次连续的信封 memset。

成本实测为零（one_batch_server，isl 8192 / osl 1024，K3 TP8，每档 2 次重复）：bs 1/8/16/24/32 的输出 tok/s 变化 = −0.07% / −0.02% / +0.05% / +0.03% / +0.02%，TTFT 在 ±0.34% 内——每一档的 rep 间方差都大于这个修复的 delta。

> 💡 **值得抄的工程手法**：他们顺手加了 `SGLANG_DEBUG_POISON_POOL`——启动时把池填成 bf16-NaN 位模式而不是零。这把「释放的 GPU 堆里碰巧有没有 NaN」这个开机抽奖变成了一个确定性开关，并作为这类 bug 的常驻回归夹具保留下来（默认零开销）。任何自建内存池的项目都该有这么一个开关。

### 根因 2：int32 步长回绕 —— 只在「统一池 + K3 + 高负载」三重条件下现形的 heisenbug

DSPARK 的 KDA verify 内核走 `nv_cutedsl` 快路径，编译时用的是静态 CuTe layout——也就是说，state 池的步长是编译期常量。在统一池上，信封式跨步的 KDA state view 的槽步长高达约 1400 万（fp32 ssm）/ 2800 万（bf16 conv）个元素。每个常量单独看都塞得进 int32，于是编译器把整个索引算术折叠成 32 位——`slot × stride` 在槽 id 超过约 153（conv）/ 306（ssm） 时静默 `mod 2³²` 回绕。

后果分两档：轻则读进别的槽的字节——干净启动下的静默状态损坏，GSM8K 掉到 0.120，还伴随虚假的接受率暴涨；重则飞出分配区变成非法访问（illegal memory access）。

为什么只在这三重条件下现形？

- 静态连续池的槽步长只有约 19.7 万个元素——永远够不到回绕点。
- Kimi-Linear 的 head dim 过不了 128 的契约，压根不走这个内核（回退 triton）。
- 冷启动发出去的都是小槽 id——所以它与负载相关：压力上来、槽 id 变大，才开始错。

所以它是 unified-only、K3-only、load-only —— 教科书级的 heisenbug。

怎么证的：离线重放哨兵——dump 一次真实的系统内调用，用「连续」与「忠实跨步」两种副本重建 state（间隙分别填零 / 填 NaN），再扫描槽 id 量级。结果：修复前，槽 ≤75 位精确、100–150 非法访问、≥155 静默发散；修复后处处位精确。

修法一句话：在乘 state 步长之前，把 slot id 提升到 int64。

## 五、给你的可执行结论

- **若你在 Kimi-K3 上同时开 `--enable-unified-memory` 与 DSPARK 投机解码，08-10 之前的版本会在负载起来之后静默出错**（不报错、准确率崩、accept rate 还很漂亮）。立刻同步 main 或等下个 tag。
- **反直觉的诊断信号**：accept rate 异常地高且稳定（比如 1.00 / 8.00）反而是危险信号——`argmax(NaN) = 0` 会造出一串全 0 token 的「完美接受」。别把它当成调优成功。
- **三缺一不复现的组合最危险。** #33517 是完美例证：统一池 × Triton × 确定性推理，任意两个都正常。请把「池类型 × 注意力后端 × 投机算法 × 确定性开关」当成一个笛卡尔积来做回归，而不是逐维度测。
- **通用教训（跨框架适用）**：任何「编译期常量步长 + 32 位索引算术」的内核，在数据面从静态连续池换成大跨步视图之后，都必须重新核算 `max_slot × stride` 是否越过 `2³¹`。这类回绕在 vLLM 的 KV offload / 规范布局路线上同样存在结构性风险（此为推断，非社区已报缺陷）。

## 六、常驻专题进展

**① PD 分离**：本窗口 SGLang 在统一池上把 PD 分离正式打通（#33362，15 文件 +835/−51）：把「整信封 + 物理 id」传输方案、虚拟↔物理翻译收敛到交接点、PD+unified 强制惰性压缩三件套一次性补齐；配套 #34191 让 PD 的 prefill 服务器不再为投机验证 scratch 买单（每 rank 省 ~24 GB 死重）。vLLM 侧延续 KV 卸载规范布局（#48414，与并行拓扑彻底解耦）+ 既有的 NIXL/Dynamo 加固线。结论：PD 分离已从「能跑通」走到「能搬运新数据面」——任何换掉 KV cache 底层布局的改动，都必须为 RDMA 引擎补一次地址翻译。

**② 最新推理架构演进**：主轴高度收敛到「新数据面 + 投机解码的正交组合正确性」。SGLang 统一池（KV+Mamba state 同池）在 24h 内连修三处静默缺陷并同时接上 DSPARK 与 PD 分离；vLLM 则用 v0.27.0 把 Kimi-K3 全栈（含 DSpark AR 融合 #50242）、MRv2 扩到非生成式负载、Rubin sm_107 一次性交付。值得注意：投机解码正从「提吞吐」退居为「与 KV 复用 / 混合架构 / PD 分离组合时的正确性载体」——本期最刺眼的 bug 几乎都「开投机之后」才暴露。

**③ 纯 PyTorch vs HuggingFace transformers 取舍**：本期两边同步向 PyTorch 2.13「环境级」升级（vLLM #48155 / SGLang 此前 #28836），数据面自研趋势未变。分层结论持续有效：「模型定义」向 transformers 收敛求广度（vLLM 持续泛化 transformers 后端，day-N 零适配成本），「数据面与运行时」向自研收敛求确定性（SGLang 统一池、Rust 视觉管线、纯 Triton 内核）。新信号：vLLM #51727 揭示 `deepseek_v32` 的 `__len__` override 写于「HF fast tokenizer 的 `len()` 尚不含 added token」的年代，transformers 5.x 已包含——陈旧绕过既冗余又错误，这是框架跟进 transformers 升级时必查的一类回归点。

**④ 阶跃星辰 Step 系列适配**：连续第六日无实质进展。硬证据：StepFun-ai org 最新推送仍 Step-Realtime-CLI（07-23，08-10 有推送但 commits 停 07-17）；Step-3.7-Flash 主力仓库停 06-01（303★）；官方 vLLM fork 停 05-28。vLLM 侧 Step PR：#49642（MTP per-layer config 越界，IndexError 起不来）仍 open、#50290 等；SGLang 侧 #32325（NVFP4 MTP shared-head 2048≠4096）仍 open。WebSearch 确认 Step-3.7/3.5-Flash 社区部署指南广泛存在（skillsbot / xiaolandeng / CSDN），但近 24h 无新模型发布。结论不变：Step 性能卖点强依赖 MTP 自带草稿，而 MTP 恰是最不稳的一块；依赖 Step 上生产的团队，先验 `--speculative-config '{"method":"mtp"}'` 能否起服，并准备好回退到非投机路径。

## 七、对比视角

「一边交付、一边还债」是本期最鲜明的两相对照。

| 维度 | vLLM v0.27.0 | SGLang main（无新 tag） |
|------|--------------|------------------------|
| 本期动作 | 交付 561-commit 大版本 | 还债：统一池三连修静默缺陷 |
| 主轴 | 广度（K3 全栈 / MRv2 非生成 / Rubin） | 深度（新数据面正确性 + PD/投机打通） |
| PyTorch 策略 | 升 2.13.0，环境级破坏性 | 此前已升（#28836，下个 tag 带走） |
| 对部署者含义 | 需重建镜像 + 精度回归 | 同步 main 前核对统一池回归 |
| 共同点 | 双方都在 PyTorch 2.13 上对齐（环境依赖收敛）；双方都把「静默正确性缺陷」当一等公民在修（vLLM #50323 的 NaN CI 门禁、SGLang 的 async-assert + poison-pool 夹具） | |
| 分歧点 | vLLM 把风险暴露在「升级路径」（环境破坏性，可计划） | SGLang 把风险暴露在「数据面内翻」（统一池 new code path，靠回归夹具兜底） | 

> 信息来源（均取自 GitHub API 原始响应）：vLLM v0.27.0 Release；vLLM PR #51727 / #51430 / #49948 / #48414 / #47352 / #50323；SGLang PR #33974 / #33517 / #33362 / #34191 / #31700；StepFun-ai org。本日报所有版本号、PR 编号、性能数字均来自 GitHub API 直查，未作推断部分已标注「推断」。
