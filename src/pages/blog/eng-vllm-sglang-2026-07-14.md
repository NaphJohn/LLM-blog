---
title: vLLM & SGLang 社区跟踪 · 2026-07-14（周二）
description: 联网检索时间：2026-07-14 09:00 GMT+8｜范围：近 24–72 小时优先，GitHub Releases / Discussions / 
pubDate: 2026-07-14
series: 大模型工程实践
lang: zh
layout: ../../layouts/BlogPost.astro
---

> 联网检索时间：2026-07-14 09:00 GMT+8｜范围：近 24–72 小时优先，GitHub Releases / Discussions / 官方博客 / 技术文章与 benchmark。
> 口径：所有版本号、日期、PR 编号、性能数字均来自下方「信息来源」，不杜撰。

## 一、今日概览

- **重点：vLLM v0.25.0 于 2026-07-12 发布，落在近 72 小时内**，是本期最大更新（558 commits / 232 贡献者 / 64 位新贡献者）。
- **SGLang 近 72 小时无新版本发布**，最新稳定版仍为 **v0.5.15（2026-07-10）**；主分支最近提交停留在 2026-07-06，最近一次实质社区内容为其官方博客《DSpark in SGLang》（2026-07-06）。

---

## 二、vLLM 社区动态（重点）

### 1. 版本发布 — vLLM v0.25.0 正式发布
- **摘要**：大版本更新（558 commits、232 贡献者、64 新面孔），定位为一次架构换代而非增量改进。
- **来源**：[GitHub Release v0.25.0](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
- **对部署/选型的影响**：含多项破坏性变更，升级前必须跑完整回归，尤其量化 / LoRA / 投机解码 / 多模态组合场景。

### 2. 重要 PR / 架构变更 — PagedAttention 被正式移除（PR #47361）
- **摘要**：曾让 vLLM 一战成名的 PagedAttention 被删除，V1 / MRv2 后端成为唯一标准执行路径，属破坏性变更。
- **来源**：[vLLM v0.25.0 Release Notes（#47361）](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
- **对部署/选型的影响**：依赖旧注意力路径的部署脚本需迁移；这是从「单一通用算法」到「可插拔多后端」的架构重写。

### 3. 新特性 — Model Runner V2（MRv2）成为所有稠密模型默认路径（PR #44443）
- **摘要**：MRv2 默认化，并补齐前缀缓存、多模态双向注意力、兼容完整 CUDA 图的动态投机解码、Mamba 混合模型前缀缓存等能力。
- **来源**：[v0.25.0 Release Notes（#44443）](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
- **对部署/选型的影响**：后续新优化优先沉淀到 MRv2，老路径不再维护；同一模型走新老路径的吞吐/显存/调度差异会越来越大。

### 4. 性能提升 — Transformers 后端追平原生 vLLM（PR #47187）+ 多项融合核提速
- **摘要**：Hugging Face Transformers 建模后端性能已与原生 vLLM 持平并支持 FP8 MoE；GLM-5.2 / DeepSeek Triton 融合核带来 1.9–3.3% E2E 吞吐，DSv4 `token_to_req_indices` 缓存加速 5–6×。
- **来源**：[v0.25.0 Release Notes（#47187 / #47474）](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
- **对部署/选型的影响**：在 HF 看到的新模型可直接塞给 vLLM 跑，新模型接入门槛降到接近零，无需等原生实现。

### 5. 新特性 — 流式解析引擎（Streaming Parser Engine，PR #46610）
- **摘要**：统一工具调用 / 推理（reasoning）解析框架，新增 Kimi k2.5 / k2.6 / k2.7、DeepSeek V4 等专用解析器。
- **来源**：[v0.25.0 Release Notes（#46610）](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
- **对部署/选型的影响**：Agent 场景下流式输出里的工具调用与思考过程能被正确识别，减少「JSON 被修坏 / reasoning 被吞」的脏坑。

### 6. 新特性 — 通用投机解码（异构词表 TLI，#38174）+ 新 drafter / 新模型
- **摘要**：通用投机解码支持异构词表（TLI）；新增 DSpark（#46995）、DFlash（#46770）drafters；新增 Hy3、GLM-5、DeepSeek-V3.2、MiniMax-M3（流水线并行 + NVFP4）等模型。
- **来源**：[v0.25.0 Release Notes（#38174 / #46995 / #46770）](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
- **对部署/选型的影响**：带 thinking budget 的推理模型（DeepSeek-R1、Qwen3 等）投机解码更可用；国产 / 新架构模型第一时间获得支持。

---

## 三、SGLang 社区动态

- **近 72 小时无重大更新。** 最新稳定版仍为 **v0.5.15（2026-07-10）**，主分支最近提交停留在 2026-07-06。
- **参考（上期已详述，本日不再重复铺陈）**：v0.5.15 要点包括 Spec V2 默认开启（+11% TPS）、GLM-5.2 在 Blackwell 上达 500+ tok/s/user（8×B300）、DeepSeek V4 FlashMLA 稀疏 prefill 默认开启（长上下文 >10% 吞吐）、原生 Web 搜索（ExaAI）、Breakable CUDA Graph 默认捕获路径。
- **最近一次实质社区内容**：官方博客《DSpark in SGLang: Speculative Decoding with Confidence-Driven, Variable-Length Verification》（2026-07-06）。
  - 来源：[SGLang 官方博客](https://sgl-project.github.io/)
  - 影响：与 vLLM 同期推进 DSpark 投机解码，是两边最清晰的横向对照主线（见第四节）。

---

## 四、原理 / 代码解读：vLLM 为什么删掉 PagedAttention，把 MRv2 设为默认？

PagedAttention 是 vLLM 的「招牌技术」——把操作系统虚拟内存分页的思路搬到了 GPU KV Cache 管理上，用非连续的分页块消除 KV Cache 碎片，从而在同一张卡上塞下更多并发请求。它曾是 vLLM 在推理框架里杀出重路的核心。

**为什么是现在删？** 删除的前提是替代方案已经成熟。v0.25.0 把 **Model Runner V2（MRv2）设为所有稠密模型的默认执行路径（#44443）**，而 MRv2 在上一版支持了量化模型后，这一版又补齐了关键能力：
- 前缀缓存（prefix caching）
- 多模态双向注意力（bidirectional attention）
- **兼容完整 CUDA 图的动态投机解码**（dynamic spec decode）
- Mamba 混合模型前缀缓存

**架构含义**：这不是增量改进，而是重写。PagedAttention 是「一个通用算法解决所有问题」；MRv2 是「可插拔后端框架」——你可以按场景换后端，每个后端在特定负载下比通用方案做得更好。配套地，旧版注意力实现被整体移除（#47361），V1 / MRv2 后端成为唯一标准路径。

**对实际部署的代价**（破坏性变更需重点关注）：
- PagedAttention 相关路径被移除，依赖旧路径的脚本要迁移；
- 移除了若干旧模型支持：Baichuan、Aquila、Grok、Tarsier/Tarsier2、AyaVision/MusicFlamingo、Mantis；
- 不再内部设置 `CUDA_VISIBLE_DEVICES`，改用显式 `device_ids` 参数——K8s / 多服务共用 GPU / Ray 多进程的启动方式需检查。

一句话：**vLLM 正从「推理加速器」转向「模型服务平台」**，代价是升级时要更严格地回归测试。

---

## 五、对比视角

- **投机解码是本期最清晰的横向对照**：vLLM v0.25.0 新增 DSpark / DFlash drafters 与异构词表通用投机解码（#38174）；SGLang 在 v0.5.15 默认开启 Spec V2（+11% TPS）并于 2026-07-06 发布 DSpark 博客（置信度驱动的变长验证）。两边都在把「置信度 / 变长验证」投机解码做深。
- **GLM-5.2 NVFP4 生产化两边都在推进**：vLLM 做了 GLM-5.2 调优；SGLang 在 8×B300 上达 500+ tok/s/user、4×GB300 上 450 tok/s/user（bs=1）。
- **架构取向不同**：vLLM 在 v0.25.0 移除 PagedAttention、全面切 MRv2；SGLang 继续以 RadixAttention 为核心并强化 prefix caching / 稀疏 prefill。

---

## 六、今日最值得关注

**vLLM v0.25.0 的架构换代（PagedAttention 移除 + MRv2 默认）**。这是一次重写而非修补，升级务必完整回归测试——尤其量化、LoRA、投机解码、多模态的组合场景，以及 `device_ids` 替代 `CUDA_VISIBLE_DEVICES` 的启动方式变更。若你的服务依赖旧注意力路径或已移除的模型（Baichuan/Aquila/Grok 等），需先评估迁移成本再升级。

---

## 信息来源（可溯源）

- vLLM v0.25.0 发布页与 Release Notes：https://github.com/vllm-project/vllm/releases/tag/v0.25.0
- vLLM 版本总览（含历史版本日期）：https://vllm.com.cn/releases ｜ https://libraries.io/pypi/vllm/versions
- vLLM v0.25.0 解读（PagedAttention 移除 / MRv2 默认）：https://m.toutiao.com/is/rY2sOPX8Tec/ ｜ https://www.163.com/dy/article/L1LN5D4R0519EA27.html
- SGLang 最新 Release（v0.5.15，2026-07-10）：https://github.com/sgl-project/sglang/releases/tag/v0.5.15
- SGLang 官方博客（DSpark 等，2026-07-06）：https://sgl-project.github.io/
- SGLang v0.5.15 发布要点（社区帖）：https://www.tweetlook.com/sgl_project
