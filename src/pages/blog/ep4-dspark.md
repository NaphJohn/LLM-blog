---
title: DSpark 深度解析：半自回归起草 + 马尔可夫头 + 置信度调度
description: 拆解 DSpark（DeepSeek + 北大开源）的半自回归草稿、马尔可夫头顺序建模与置信度调度器，以及它在 vLLM / SGLang 上的落地。
pubDate: 2026-07-31
series: 推测解码手记
lang: zh
altLang: en
altHref: /en/blog/ep4-dspark
layout: ../../layouts/BlogPost.astro
---

## 1. 背景与来源

**DSpark** 由 **DeepSeek + 北京大学**于 2026-06-27 开源（MIT 协议，仓库 `deepseek-ai/DeepSpec`）。它与 DFlash 走的是**不同路线**：不靠扩散去噪，而是用**半自回归 + 马尔可夫头**来建顺序、用**置信度调度**来动态控验证长度。

## 2. 核心一：Semi-Autoregressive Drafter + Markov Head

- **半自回归（semi-autoregressive）**：以"块"为单位并行出块，块间可以并行，块内仍有顺序；
- **马尔可夫头（Markov head）**：在块内注入 token 间的顺序依赖，保证块内生成顺序一致、不至于乱序。

这与 DFlash 的扩散范式形成对照：DSpark 在"自回归 ↔ 纯并行"之间取折中——比纯自回归并行、比纯扩散更可控。

## 3. 核心二：Confidence Scheduler（置信度调度器）

这是 DSpark 最巧妙的一笔：

- 不固定 block size，而是**根据草稿置信度动态决定本次验证多少 token**；
- **高置信** → 多验证（一次多换几个 token）；**低置信** → 少验证（避免把整块都喂进去却大量被拒的浪费）。

相对 DFlash 固定 block size=16，DSpark 的验证长度随"当前该不该信草稿"自适应，进一步逼近 α 的理论上限。

## 4. 核心三：大模型并行验证，零质量损失

与 DFlash 一样，DSpark 也是**无损推测解码**：草稿块交给大模型一次并行前向验证，拒绝采样保证输出分布严格等于目标模型。

## 5. 效果与生态

| 指标 | 数值 / 状态 |
|---|---|
| V4 加速 | **57–85%**（据 DeepSeek / 北大发布资料） |
| 吞吐 | **+400%** |
| vLLM 落地 | PR **#46995** 在合入（复用稀疏 MLA 非因果索引、`DSparkSpeculator` 继承 `DFlash`、Triton 非因果 SWA 内核） |
| 多后端 | 同时支持 SGLang / OpenInfer |

## 6. 与 DFlash 的关系（预告）

DSpark 改的是"草稿顺序建模 + 验证长度调度"，DFlash 改的是"草稿并行范式 + 执行引擎"——二者**不在同一层，正交可叠加**。这正是 OpenInfer 能在 Qwen3-4B 上**同时**挂两套路径的原因。完整对比见 [DFlash vs DSpark](/blog/ep5-dflash-vs-dspark)。

> 本篇属于「推测解码手记」系列 Ep4。前情：[DFlash 深度解析](/blog/ep3-dflash)；对比篇：[DFlash vs DSpark](/blog/ep5-dflash-vs-dspark)。
