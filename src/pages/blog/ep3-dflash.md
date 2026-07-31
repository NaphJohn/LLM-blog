---
title: DFlash 深度解析：块扩散起草 + KV 注入 + 无空转流水线
description: 拆解 DFlash 的三大机制——块扩散草稿、Target 隐状态条件化（KV 注入）、Spec V2 overlap scheduler，以及它在 Qwen3 上的实测加速。
pubDate: 2026-07-31
series: 推测解码手记
lang: zh
altLang: en
altHref: /en/blog/ep3-dflash
layout: ../../layouts/BlogPost.astro
---

## 1. 背景与来源

**DFlash** 由 **Z Lab + SGLang + Modal** 在 2026-06-15 的博客中系统提出，目标是把推测解码在 Qwen3 系列上推进到 **6×**。它之所以能突破 Ep2 里说的"串行起草"瓶颈，是因为从草稿范式到执行引擎都重做了一遍。

## 2. 核心一：Block Diffusion Drafter（块扩散起草）

传统草稿模型是**自回归串行**的——一个一个猜。DFlash 改用**扩散（diffusion）**范式：

- 一次前向，并行预测**一整块**被 mask 的未来 token（默认 block size = 16，`--speculative-dflash-block-size`）；
- 直觉上，它不是"逐个猜下一个"，而是"一次性把一整块的可能都填上，再同时确认"；
- 代价是扩散去噪需要多步迭代，但相比自回归的 K 次串行前向，整体仍显著更省。

这一步直接绕开了 Ep2 的瓶颈 (a) 串行起草 和 (c) 块内顺序依赖——起草从"串行串 token"变成"并行出块"。

## 3. 核心二：Target 隐状态条件化 + KV 注入

光并行还不够，接受率 α 才是加速比的命门。DFlash 的关键招数是把**目标大模型的中间隐状态（hidden states）注入草稿模型的 KV 投影（跨层）**：

- 草稿不再是"盲猜"，而是**基于目标模型的特征做条件化**；
- KV injection 让草稿的提议与目标上下文高度对齐 → 接受率显著抬高。

这是 DFlash 高 α 的来源，也是它比 EAGLE-3 快约 2.5× 的重要原因。

## 4. 核心三：Spec V2 引擎 + overlap scheduler（执行层）

多数 SD 方案只改算法，DFlash 额外重做了执行引擎：

- 标准流水线上，host 端做清理 / KV 分配 与 GPU 计算之间存在**同步空转（bubble）**；
- **overlap scheduler** 把 host 任务与 GPU 计算**重叠**执行 → 再额外 **+33%** 吞吐。

这是 DFlash 与"纯算法"方案的分水岭：**既改草稿范式，又改执行引擎**。

## 5. 效果数据

| 指标 | 数值 |
|---|---|
| Qwen3-8B 最高加速 | **6×** |
| 对比 EAGLE-3 | 约快 **2.5×** |
| Blackwell 上 | 最高 **15×** |
| overlap scheduler 额外增益 | **+33%** 吞吐 |

## 6. 小结与下篇

DFlash 用"**块扩散并行起草 + KV 注入抬接受率 + 无空转流水线**"三连击，把 SD 从 2–3× 拉到 6×。下一篇我们看另一条路线 **DSpark**：它不靠扩散，而是用半自回归 + 马尔可夫头 + 置信度调度另辟蹊径。

> 本篇属于「推测解码手记」系列 Ep3。前情：[自回归草稿的天花板](/blog/ep2-arg-ceiling)；对比视角：[DFlash vs DSpark](/blog/ep5-dflash-vs-dspark)；下一篇：[DSpark 深度解析](/blog/ep4-dspark)。
