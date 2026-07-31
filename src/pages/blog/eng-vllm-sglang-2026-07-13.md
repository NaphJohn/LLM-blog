---
title: vLLM & SGLang 社区跟踪 · 2026-07-13
description: 跟踪窗口：截至 2026-07-13 08:50 (GMT+8)，优先近 24–72 小时。
pubDate: 2026-07-13
series: 大模型工程实践
lang: zh
layout: ../../layouts/BlogPost.astro
---

> 跟踪窗口：截至 2026-07-13 08:50 (GMT+8)，优先近 24–72 小时。
> 说明：信息均来自可溯源的官方/社区来源，不杜撰版本号与数字；近 72 小时无新版本者已明确标注并附最近版本号。

## 一句话概览

- **vLLM 最新稳定版：v0.25.0**（GitHub Releases，约 2026-06-29）。近 72 小时**无新版本发布**，但过去 1–2 周社区有重大进展（DSpark 推测解码合入主干、腾讯 Hunyuan Hy3 即日支持）。
- **SGLang 最新稳定版：v0.5.15**（GitHub Releases，2026-07-10 22:58 UTC）。✅ **落在近 72 小时内**，为本日重点。

---

## vLLM 要点

> 近 72 小时无新版本发布（最近稳定版 v0.25.0，约 2026-06-29）。以下为近 1–2 周仍高度相关的社区进展。

1. **【重要 PR / 社区讨论】DSpark 半自回归推测解码合入主干（PR #46995）**
   - 摘要：DeepSeek 团队联合北大发布的 DSpark 框架（宣称线上推理提速 60–85%）已被社区移植进 vLLM，复用 SparseMLA 后端，支持前缀缓存与 FP8 KV cache。
   - 来源：vLLM 2026 第 27 周周报（prhub.com.cn，06.29–07.05）、DeepSeek-AI/DSpark 开源仓库。
   - 影响：DeepSeek-V4 系列部署可获得显著吞吐提升；但需关注复现边界（高度依赖硬件型号、框架补丁版本、上下文长度与并发设置）。

2. **【新特性 / 模型支持】腾讯 Hunyuan Hy3 即日支持（2026-07-07）**
   - 摘要：混元 Hy3（295B MoE / 21B 激活 / 256K 上下文，Apache 2.0）由 vLLM 与 SGLang 同步宣布支持；vLLM 支持其工具调用与推理解析，已在 NVIDIA 与 AMD 硬件验证。
   - 来源：Recsys Frontier AI 日报 2026-07-07。
   - 影响：国产大模型选型可直接上 vLLM，工具调用/推理链路开箱可用。

3. **【新特性】解析器与多模态扩展**
   - 摘要：DeepSeek V4 流式解析器（PR #45877）、Kimi K2 流式解析引擎（PR #46610）合入，多模态新增 LLaVA-OneVision-2（PR #44785）。
   - 来源：vLLM 2026 第 27 周周报。
   - 影响：多模态与 agent 场景的流式输出稳定性提升。

4. **【版本特性参考】v0.25.0 核心变化**
   - 摘要：据 GitHub Releases，v0.25.0 将 Model Runner V2 设为所有稠密模型默认路径、移除 PagedAttention、Transformers 后端提速至接近原生 vLLM 水平（含 558 commits / 232 贡献者）。
   - 来源：GitHub Releases（vllm-project/vllm）。
   - 影响：升级时需注意后端默认路径与依赖变化。

---

## SGLang 要点（v0.5.15，2026-07-10 发布）

1. **【版本发布 / 性能提升】GLM-5.2 NVFP4 生产级调优（Blackwell）**
   - 摘要：8×B300 达 **500+ tok/s/user**、4×GB300 达 **450 tok/s/user**（bs=1）。
   - 来源：GitHub Releases tag v0.5.15。
   - 影响：GLM-5.2 生产部署首选已验证的高吞吐配置，单卡成本/性能比清晰。

2. **【性能提升】Spec V2 成为默认**
   - 摘要：零开销调度使端到端 TPS **+11%**；IndexShare MTP 长上下文草稿步成本最高降 **1.9x**；Indexer prologue 融合（12 核函数合并为 4）解码提速约 **8%**。
   - 来源：v0.5.15 release notes（PR #29413 / #29959 / #27705）。
   - 影响：长上下文高并发场景默认即享吞吐红利，无需额外调参。

3. **【新特性】原生 Web 搜索、上下文并行与 MoE 路由**
   - 摘要：内置 Exa `web_search` 支持；Decode Context Parallelism 落地 MLA 模型（DeepSeek V3、Kimi K2）；新增 FlashInfer A2A for routed MoE。
   - 来源：v0.5.15 release notes。
   - 影响：agent/检索增强与超长序列部署能力补强。

4. **【新模型支持】**
   - 摘要：Hunyuan 3 (Hy3)、HRM-Text、LocateAnything-3B、Unlimited-OCR、JoyEcho 多轮 A/V、Qwen3.6 NVFP4 等。
   - 来源：v0.5.15 release notes。

5. **【DeepSeek V4 优化】**
   - 摘要：FlashMLA 稀疏 prefill 默认开启（长上下文 >10% 吞吐）、非分页 indexer（>5% e2e 吞吐）；JIT Kernel DSA indexer 支持 runtime k≤2048。
   - 来源：v0.5.15 release notes（PR #29775 / #29619 / #30274）。

6. **【社区讨论 / Bug 修复】智谱 GLM-5 Coding Agent 成果回流**
   - 摘要：智谱披露 GLM-5 系列在 Coding Agent 场景系统吞吐最高提升 **132%**、异常输出率降至万分之三以下；修复方案已通过 **PR #22811** 提交 SGLang（2026-07-10）。
   - 来源：智谱《Scaling Pain》技术博客报道（aastocks，2026-07-10）。
   - 影响：高并发 Coding Agent 场景收益显著，建议关注该 PR 落地节点。

---

## 横向对比

- **推测解码**：vLLM 在 06.29–07.05 周合入 DSpark 半自回归推测（#46995）；SGLang v0.5.15 已将 Spec V2 设为默认并给出 +11% TPS 实测。两者都押注"推测解码"作为吞吐主引擎 —— SGLang 已默认开启并量化收益，vLLM 仍在生态补齐阶段。
- **新硬件/模型**：两边均在 07-07 即支持腾讯 Hunyuan Hy3（295B MoE）；SGLang 额外提供 FP8 checkpoint，vLLM 侧重工具调用/推理解析。
- **DeepSeek V4**：SGLang v0.5.15 已默认开启 FlashMLA 稀疏 prefill（>10% 长上下文吞吐）；vLLM 通过 DSpark + SparseMLA 后端跟进，社区复现数据依赖具体硬件。

---

## 今日最值得关注

**SGLang v0.5.15（2026-07-10）默认开启 Spec V2 推测解码（端到端 TPS +11%、长上下文草稿成本降 1.9x），并带来 GLM-5.2 在 Blackwell 上 500+ tok/s/user 的生产级调优。** 对正在做 GLM-5.2 / DeepSeek-V4 推理部署、且关注单卡吞吐与成本的用户，这是本周最该升级验证的版本。
对比之下，vLLM 近 72 小时无新版本，DSpark 推测解码刚合入主干、尚未进稳定版 —— 生产落地建议先观望或基于 nightly 验证。

---

## 信息来源（可溯源）

- vLLM GitHub Releases：https://github.com/vllm-project/vllm/releases
- SGLang GitHub Releases（v0.5.15）：https://github.com/sgl-project/sglang/releases/tag/v0.5.15
- vLLM 2026 第 27 周周报（prhub.com.cn）：https://prhub.com.cn/vllm-project/vllm/reports/2026-06-29-to-2026-07-05
- Recsys Frontier AI 日报 2026-07-07：https://recsys-frontier.com/article/ai-daily-2026-07-07
- 智谱 GLM-5 Coding Agent 技术博客报道：https://www.aastocks.com/tc/stocks/news/aafn-con/now.1521700/industry-news/source
- DeepSeek-AI/DSpark 开源仓库：https://github.com/deepseek-ai/DSpark
