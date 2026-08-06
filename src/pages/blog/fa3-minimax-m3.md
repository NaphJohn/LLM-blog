---
title: 前沿架构解码手记（三）：MiniMax M2.7 → M3——稀疏注意力的跨越
description: 顺着路线图，第二家拆解 MiniMax。先看 M2.7 的 baseline（标准注意力、200K、纯文本、仅 API），再看 M3 如何用 MiniMax Sparse Attention（MSA）把上下文从 200K 拉到 1M、同时把 1M 下的每 token 算力压到 M2.7 的约 1/20（prefill 快约 9×、decode 快约 15×）。M3 还补齐原生多模态（文本+图像+视频）与 computer use，并开源权重。
pubDate: 2026-08-06
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa3-minimax-m3
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么看 MiniMax 的"演进"

Kimi K3 是"在 MLA 上加差分"，改动温和。MiniMax 给了更直观的一课：**同一个系列，如何从标准注意力一步步稀疏化**。先看清起点 M2.7，才懂 M3 的跨越有多大。

## 1. 起点：MiniMax M2.7（baseline）

- **标准 softmax 注意力**，无稀疏；
- **上下文 200K**，纯文本，无原生多模态；
- **仅 API 提供**，权重未开源；
- 训练数据以文本为主。

它是"上一代"稠密/标准注意力的代表：能力够用，但上下文与算力都到天花板。

## 2. 跨越：MiniMax M3 + MSA

M3 的核心创新是 **MiniMax Sparse Attention（MSA，稀疏注意力）**：

- **上下文从 200K → 1M**（5×）；
- 在 1M 上下文下，**每 token 算力约为 M2.7 的 1/20**；
- **prefill 快约 9×、decode 快约 15×**；
- 训练用 **100T token**、**交错（interleaved）数据**。

稀疏注意力的直觉：并非每个 query 都要和所有历史 key 算一遍。MSA 用一套稀疏选择机制，让每个位置只关注"真正相关"的少部分历史，从而在 1M 长度下把算力压住。

<div class="keybox">
<strong>要点：</strong>稀疏注意力的收益来自"少算"——M3 在 1M 长度下每 token 算力只有 M2.7 的 <strong>1/20</strong>，换来的是 prefill/decode 数量级提速。这是"上下文变长但成本不爆"的典型解法。
</div>

## 3. 不止注意力：原生多模态 + computer use

M3 并非只改注意力，还补齐了"可用性"：

- **原生多模态**：文本 + 图像 + 视频统一建模，不再是文本模型外挂视觉；
- **computer use**：能操作 GUI / 浏览器等界面，贴合 Agent 场景；
- **权重开源**：相对 M2.7 的"仅 API"，M3 开放权重，利于生态与二次开发。

## 4. 基准表现（官方口径）

| 基准 | M3 成绩 |
|---|---|
| SWE-Bench Pro | 59.0% |
| Terminal-Bench 2.1 | 66.0% |
| MCP Atlas | 74.2% |

这些偏"agentic / 工具调用 / 终端操作"的基准，恰好对应 M3 的 computer use 与长上下文定位。

## 5. 与 Kimi K3 的对照

- Kimi K3 是"**改注意力内部**（KDA 差分 + 门控）"，MLA 底子保留；
- MiniMax M3 是"**改注意力的计算范围**（稀疏化）"，从标准注意力直接跨到 MSA；
- 共同点：都为了**撑住 1M 上下文又不爆算力**；差异在动刀的位置不同。

> 下一章看 DeepSeek V4：它比稀疏更激进——直接**压缩 KV 本身**。
