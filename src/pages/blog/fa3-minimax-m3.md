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

### 1.1 前史：从 MiniMax-Text-01 的 Lightning Attention 到 M2.7 的"退回全注意力"

M2.7 并不是 MiniMax 第一次碰"非标准注意力"。更早的 **MiniMax-Text-01** 用过一种叫 **Lightning Attention** 的混合线性注意力（linear / retentive attention 变体，与标准 softmax 层交替堆叠），借此把上下文一口气推到 4M。线性注意力的好处是算力随长度**线性**增长、不平方爆炸；代价是表达能力与训练稳定性受限，长程精确检索、复杂推理上不如全注意力。

MiniMax 在后续迭代里做了一个关键转折：先回到**标准 softmax 全注意力**作为稳妥 baseline（即 M2.7 的 200K 标准注意力），把"超长上下文"的赌注留到更可控的方案上；再到 M3 用**稀疏注意力（MSA）**而非线性注意力去攻 1M。换句话说，M2.7 的"退回全注意力"不是倒退，而是把"长上下文"从一条风险高的技术路线（线性注意力），换到了一条更可控的路线（先全注意力打底、再稀疏化扩展），为 M3 的 MSA 扫清了地基。

> 这条"线性注意力探索 → 退回全注意力 → 稀疏化"的曲线，和 Kimi K3（在 MLA 上做差分）、DeepSeek V4（直接压 KV）构成了 2026 年"改注意力"的三种不同勇气。

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

### 2.1 工程侧：KV-Block-Major Prefill——承载 MSA 的 KV 布局

MSA 让每个 query 只挑少量 KV block 计算，注意力从"全连接"变成"块选择"。这对推理引擎的 KV 缓存布局提出了新要求。

标准 paged-attention 的 KV 是 **token-major**：page → token → head → dim。当注意力是块选择式的（每个 query 选不同的 KV block 集合），token-major 会让"取某个 query 选中的若干 block"变成跨 page 的零散 gather，prefill 时内存访问不连续、kernel 难写。

**KV-Block-Major** 把布局换成 **block → token → head → dim**：同一 block 内的 token 在内存里连续。这样每个 query 选中的若干 block 可以整块连续读出，gather 被压成少量大块拷贝，prefill kernel 既能 coalesce 内存访问、又能按 block 并行。正是这类布局支持，让 MSA 这类"块稀疏注意力"模型能在 1M 上下文下被 vLLM 等引擎直接服务，而不必重写整套注意力后端（所谓"Day-0 承载"的关键之一）。

> 一句话：MSA 改的是"算哪些"，KV-Block-Major 改的是"怎么存才让这种算法拉满吞吐"——算法与系统工程是同一枚硬币的两面。

### 2.2 稀疏注意力的来路：从 BigBird / Longformer 到 MSA

MSA 不是从石头里蹦出来的。块稀疏注意力的思想至少可追溯到 2020 年前后的几篇奠基工作：

- **Longformer**（2020）：用"滑动窗口 + 全局 token + 任务相关稀疏"组合，把注意力从 O(n²) 降到 O(n)，证明"不是每个 token 都要看全部历史"就能干活。
- **BigBird**（2020）：用"随机块 + 窗口块 + 全局块"三类稀疏模式，理论上等价 Turing 完备、复杂度仍为线性，是块稀疏注意力的代表之作。
- 后续 **Linear Attention / Retentive Network / Gated Linear Attention** 等则走另一条路：用核技巧或递归把注意力写成线性，但牺牲了部分表达力（见 1.1 的 Lightning Attention 复盘）。

MSA 站在这些肩膀上，但做了两处关键升级：**（1）选择是学习出来的**，不是固定窗口/随机模式，模型自己决定每块该看哪些历史块；**（2）与原生多模态、长上下文训练深度耦合**，而非事后贴上去的稀疏技巧。所以它既不是 Longformer 的固定窗口，也不是 BigBird 的随机块，而是"可学习块稀疏 + 端到端长上下文训练"的现代版本。

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
