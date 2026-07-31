---
title: vLLM & SGLang 社区跟踪 · 2026-07-16（周四）
description: 跟踪对象：vLLM、SGLang、阶跃星辰 Step 系列（Step-3 / Step-3.7 Flash / Step-Video / Step-Audi
pubDate: 2026-07-16
series: 大模型工程实践
lang: zh
layout: ../../layouts/BlogPost.astro
---

> 跟踪对象：vLLM、SGLang、阶跃星辰 Step 系列（Step-3 / Step-3.7 Flash / Step-Video / Step-Audio / Step-3-VL）。
> 时间窗：近 72 小时（2026-07-13 ~ 07-16 09:00 GMT+8）。信息来源均可溯源，版本号/数字均来自官方发布或一手报道，未杜撰。

---

## 一、vLLM 要点（3 条）

1. **[版本发布] v0.25.1 补丁（2026-07-14）** —— 在 v0.25.0 之上合入 2 个提交，两个定向修复：
   - 混合 dtype 量化融合守卫（PR #48330）：修复 NVFP4 模型在激活(BF16)与 RMSNorm 权重(FP32) dtype 不一致时，融合核静默损坏隐状态、输出重复 `!!!!!` 乱码的正确性 bug。
   - TorchCodec 起服阻塞修复（PR #47888）：系统缺 FFmpeg 时 `vllm serve Qwen/Qwen3-VL-*` 不再因 import 失败而卡死。
   - 来源：https://github.com/vllm-project/vllm/releases/tag/v0.25.1
   - **影响**：所有跑 NVFP4（Gemma4 / Qwen 系 FP32 RMSNorm 权重）量化服务的用户应**立即升级到 v0.25.1**，否则可能静默输出乱码且无报错。

2. **[版本背景] v0.25.0（2026-07-12，略超 72h 但为本周期基座）** —— 架构换代：移除 PagedAttention、Model Runner V2 成为稠密模型默认路径、动态推测解码兼容完整 CUDA graph、KV offload + Hybrid Memory Allocator、多硬件扩展（AMD ROCm 7.2.2 / Intel XPU / IBM Power / RISC-V）。
   - 来源：https://github.com/vllm-project/vllm/releases/tag/v0.25.0
   - **影响**：升级跨度大，需全量回归（尤其旧模型如 Baichuan/Aquila/Grok 已移除、`device_ids` 取代 `CUDA_VISIBLE_DEVICES`）。

3. **[新特性] 量化与推理模型支持** —— NVFP4 KV cache + ModelOpt W4A16、MXFP4（Humming backend）、TurboQuant 统一量化；thinking-budget 感知的推测解码（DeepSeek-R1 / Qwen3 推理模型不再与投机冲突）。
   - 来源：https://www.besthub.dev/articles/vllm-0-25-release-new-model-support-multi-hardware-and-major-performance-boosts-364d485d5e05
   - **影响**：低比特量化与推理模型可兼得提速，4-bit 在 Hopper/Blackwell 上性价比更优。

---

## 二、SGLang 要点（3 条）

1. **[社区讨论 / 官方博客] Serving GLM-5.2 NVFP4 两周冲到 500 TPS（2026-07-14）** —— lmsys 博客详解 GLM-5.2 NVFP4 在 8×B300 达 500+ TPS（bs=1）的调优路径：
   - Spec V2 默认开启（重叠调度隐藏 CPU 开销）→ 端到端 **+11% TPS**；
   - **IndexShare MTP**：复用 DSA indexer 的 top-k 跨草稿步，长上下文草稿步成本降 ~1.9x；
   - **TopK-V2（Lightning-TopK）**：把 TopK 当"选择"而非"排序"，8 个 CTA 直方图 + FP32 radix 精选，优化 80k ISL 的 DSA indexer。
   - 来源：https://www.lmsys.org/blog/2026-07-13-glm52-optimization
   - **影响**：在 Blackwell 上跑 GLM-5.2 NVFP4 生产服务可直接复用其 Spec V2 + IndexShare MTP + TopK-V2 调优经验。

2. **[版本补丁] v0.5.15.post1（2026-07-14）随 GLM-5.2 调优推出；最新稳定版仍为 v0.5.15（2026-07-10）**。
   - 来源：https://github.com/sgl-project/sglang/releases/tag/v0.5.15 ；镜像见 https://sourceforge.net/projects/sglang.mirror/files/
   - **影响**：本周期**无新功能版本**，主要是 GLM-5.2 NVFP4 生产化打磨 + DeepSeek V4 FlashMLA 稀疏 prefill 默认（`#29775`）。

3. **[模型支持] v0.5.15 覆盖面广** —— 已含 GLM-5.2、Hunyuan 3、DeepSeek V4（FlashMLA 稀疏 prefill 默认）、Laguna XS.2 等调优；AMD MI300X/MI325X/MI355X 上 EAGLE 验证。
   - **影响**：国产与前沿模型覆盖持续领先，新模型 day-0 支持 + 后续硬件调优的节奏已成常态（模型发布到引擎"真正快"通常滞后约一个月）。

---

## 三、阶跃星辰 Step 系列要点（3 条）

1. **[部署成熟度] Step 3.7 Flash 已获两大框架官方化支持** —— 稀疏 MoE，196B 总参 / 11B 激活 / 1.8B ViT，最高 400 TPS（2026-05-29 开源）。
   - vLLM：预建镜像 `vllm/vllm-openai:stepfun37`，支持 FP8 / BF16 / NVFP4，MTP 投机（`num_speculative_tokens:3`），reasoning/tool 解析器 `step3p5`，需 vLLM 0.25 的 `--async-scheduling`。
   - SGLang：dev 镜像 `lmsysorg/sglang:dev-step-3.7-flash`，EAGLE 多层投机（`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-multi-layer-eagle`）。
   - 来源：https://static.stepfun.com/blog/step-3.7-flash/ ；https://huggingface.co/stepfun-ai/Step-3.7-Flash
   - **影响**：两大框架均已可生产部署 Step 3.7 Flash，三档精度 + 投机解码齐备。

2. **[端侧多模态] Step3-VL-10B 部署活跃** —— 10B 级 VLM（PE-lang 1.8B 视觉编码器 + Qwen3-8B 解码器，PaCoRe 并行协调推理），主打端侧/低成本；社区 7/7 实测：MMMU 78.11 / MathVista 83.97 / OCRBench 86.75，RTX 4090 即可跑。
   - 来源：https://stepfun-ai.github.io/Step3-VL-10B/ ；https://arxiv.org/abs/2601.09668
   - **影响**：小参数端侧 Agent 多模态落地标杆，可与 Step 3.7 Flash 形成"云端大模型 + 端侧小模型"组合。

3. **[其他] Step-Video / Step-Audio** —— 近 72 小时无新发布，维持既有开源状态；官方持续把 Step 3.7 Flash 往 Agent 生态（Claude Code / OpenClaw / Hermes Agent）与本地部署（DGX Station / AMD Ryzen AI Max+ 395 / Mac Studio）推进。

---

## 四、原理 / 代码解读（本周期最值得 1 条）

**vLLM #48330 —— 混合 dtype 量化融合守卫（mixed-dtype quant-fusion guard）**

- **背景**：为了提速，vLLM 用 FlashInfer 把"all-reduce + RMSNorm + 静态量化"三步**融合成一个 GPU kernel**，省去中间结果往返 HBM。这是推理服务里最大的性能杠杆之一。
- **坑在哪**：这个融合核原本只看"算子形状是否匹配"就触发融合，**不检查 dtype**。当激活流是 BF16、而 RMSNorm 权重是 FP32（Gemma4 / Qwen 系 NVFP4 模型常见）时，4-bit NVFP4 张量会被当成错误位模式读取 → **隐状态损坏 → 输出一堆 `!!!!!` 之类乱码，但全程不报错**。
- **修复**：在融合前加一个 dtype-match 守卫——同 dtype 的图照常走融合快路径；激活/RMSNorm 权重 dtype 不一致的"混合 dtype"图，路由到安全（未融合）路径。
- **一句话看懂**：融合核省 HBM 往返是真快，但"默认 dtype 一致"的假设会被打破；加一个 dtype 哨兵即可**兼得速度与正确**。

> 对比：SGLang 同周期的 GLM-5.2 优化（Spec V2 / IndexShare MTP / TopK-V2）也是在"融合 + 重叠调度"上做文章，但焦点是吞吐而非正确性——两家都在用同一类 kernel 级优化手段，只是本周期 vLLM 补的是"对不对"、SGLang 推的是"快不快"。

---

## 五、常驻专题进展

### 1. PD 分离（Prefill-Decode Disaggregation）
- **SGLang**：文档持续支持 **Mooncake / NIXL** 两种 KV 传输引擎；自带 **router**（多种路由策略，支持负载均衡与容错）；提供 `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK / INTRA_NODE_NVLINK` 自定义内存池用于 NVL72 / A100/H20/H100 的 KV 传输，另含 PD 资源配比相关的 SLO 感知论文（arXiv 2603.04716）。
- **vLLM**：v0.25 含 disaggregated prefill/decode/encode 与 PD KV transfer 的分布式修复（NCCL 相关）。
- **本周期**：无重大新功能进展，以既有能力稳定使用为主。
- 来源：https://docs.sglang.ai/advanced_features/pd_disaggregation.html

### 2. 最新推理架构演进
- **投机解码**：SGLang Spec V2 默认开启（重叠调度 +11% TPS）；GLM-5.2 **IndexShare MTP** 复用 DSA indexer top-k（长上下文草稿步 ~1.9x 加速）；**TopK-V2**（Lightning-TopK，选择而非排序）优化 80k 长上下文 DSA indexer。vLLM v0.25 动态推测解码兼容完整 CUDA graph。
- **注意力 / 量化**：SGLang DeepSeek V4 **FlashMLA 稀疏 prefill 默认**；vLLM NVFP4 KV cache + ModelOpt W4A16、KV 量化。
- **多模态推理架构**：SGLang 的 **EPD（Encoder-Prefill-Decode）分离**（lmsys 2026-01 博客）把 ViT 编码从语言 prefill 再拆一层，图像重负载场景 TTFT 降 6–8×；与 Step 3.7 Flash / Step3-VL-10B 的"视觉编码器 + 语言解码器"两段式架构思路一致。

### 3. 纯 PyTorch 实现 vs HuggingFace transformers 取舍
- **社区趋势**：持续"去 transformers 耦合"——vLLM v0.25 已让 Transformers 后端追平原生实现、并弃用 transformers v4（转向 v5）；Step 3.7 Flash 通过 `trust_remote_code=True` 走自研建模代码（仅 debug/验证需 transformers 5.0+）。
- **动机/好处**：更可控的算子、更低开销、便于定制 attention/采样、减少依赖与版本耦合、更易做极致优化（融合核、CUDA graph）。
- **代价**：需自行实现 tokenizer / 模型加载 / 并行。
- **一句话横向结论**：生产服务栈正从"调 transformers"走向"自研纯 PyTorch 内核 + transformers 仅作校验"，以换取可控与极致性能，代价是维护成本上移。

### 4. 阶跃星辰 Step 系列 · 国产模型推理适配
- **适配进度**：Step 3.7 Flash 在 vLLM（官方预建镜像、MTP 投机默认路径）与 SGLang（dev 镜像、EAGLE 多层投机）均已可用，三档精度（FP8/BF16/NVFP4）齐全；Step3-VL-10B 端侧 VLM 社区部署活跃。
- **与官方支持对照**：vLLM 侧更像"已合并进主干的官方支持"，SGLang 侧偏"dev 镜像紧跟"，两者成熟度都在第一梯队；Step 系列对两大框架的友好度已可比肩 DeepSeek / Qwen 等头部开源模型。

---

## 六、对比视角

- **NVFP4 成 2026 下半年主线**：vLLM v0.25.1 与 SGLang GLM-5.2 优化博客同步于 7/14 产出，一个补"正确性"（静默乱码）、一个推"吞吐"（500+ TPS）——两家都把 4-bit NVFP4 当生产量化核心战场。
- **Step 适配成熟度**：vLLM 对 Step 3.7 Flash 是"合并进主干 + 预建镜像"，SGLang 是"dev 镜像 + EAGLE"，前者更"正式"、后者更"紧跟"。
- **发布节奏**：模型发布常先于引擎调优约一个月（如 GLM-5.2 6/16 发布、SGLang 7/14 才出 500 TPS 调优），选型时别把"day-0 支持"等同于"生产最快"。

---

## 七、今日最值得关注（1 条）

**vLLM v0.25.1 的 #48330 混合 dtype 量化融合守卫**。所有跑 NVFP4 量化服务（尤其 Gemma4 / Qwen 系 FP32 RMSNorm 权重）的用户应**立即升级**——该 bug 会静默输出乱码且全程不报错，比崩溃更危险。同步建议阅读 SGLang 的 GLM-5.2 NVFP4 优化博客，把 Spec V2 / IndexShare MTP / TopK-V2 的吞吐经验搬过来。

---

## 八、备注

- vLLM 最近版本：v0.25.1（2026-07-14）；SGLang 最近稳定版：v0.5.15（2026-07-10），另有 v0.5.15.post1（2026-07-14）补丁。
- Step 系列近 72 小时无新模型发布，最新旗舰为 Step 3.7 Flash（2026-05-29）。
- 信息来源：GitHub Releases、lmsys 官方博客、阶跃星辰官方博客/HF、第三方技术博客（BestHub / LearnAIVisually / Cerevisor），均可溯源。
