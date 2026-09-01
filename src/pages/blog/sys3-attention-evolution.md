---
title: 推理系统基础设施手记（三）：MSA / CSA / HCA —— 三种注意力改造路线一张图看清
description: 2026 年长上下文模型的注意力改造有三条主流路线：MiniMax 的稀疏注意力 MSA、DeepSeek V4 的压缩稀疏注意力 CSA、以及重度压缩注意力 HCA。它们分别在哪动刀、压缩比多少、适合什么场景，用一张对比图 + 三张原理图讲清楚。
pubDate: 2026-08-26
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys3-attention-evolution
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么要区分这三者？

2026 年的前沿模型几乎都在改注意力，但动刀位置不同：

- **MSA（Sparse Attention）**：不改 KV，只改「看哪里」——稀疏选择；
- **CSA（Compressed Sparse Attention）**：先把 KV 压成块，再稀疏选择；
- **HCA（Heavily Compressed Attention）**：把 KV 极致压缩后做稠密注意力。

三者不是替代关系，而是**分层协作关系**。DeepSeek V4 甚至把三者（滑窗 + CSA + HCA）堆在一起。理解它们各自的优势边界，才能看懂为什么 V4 能在 1M token 下把 KV 压到 10%。

## 1. 一句话定位

| 机制 | 一句话 | 代表模型 | 核心动作 | 复杂度 |
|---|---|---|---|---|
| **MSA** | 只算重要的 attention 对，其他全跳过 | MiniMax M3 | 学习/固定稀疏模式 | ≪ O(n²) |
| **CSA** | 先把 KV 压缩成块，再对块做 top-k | DeepSeek V4-Pro | 4-token 块压缩 + FP4 索引器 | O(n·c) |
| **HCA** | 把 128 个 token 压成 1 个条目，再做稠密注意力 | DeepSeek V4 上层 | 128× 压缩 + 可微融合 | O(n·c') |

> 符号：n = 序列长度；c = 压缩后条目数（CSA 中 c ≈ n/4；HCA 中 c ≈ n/128）。

## 2. 原理对比图

<div class="fig">
<svg viewBox="0 0 680 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MSA、CSA、HCA 三种注意力改造对比">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">三种注意力改造：动刀位置与压缩比</text>

  <!-- MSA -->
  <rect x="20" y="50" width="300" height="110" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.2"/>
  <text x="35" y="75" font-size="14" font-weight="700" fill="#166534">MSA · 稀疏注意力</text>
  <text x="35" y="95" font-size="12" fill="#166534">MiniMax M3</text>
  <text x="35" y="115" font-size="12" fill="#444">KV 不压缩，只选重要位置 attend</text>
  <text x="35" y="135" font-size="12" fill="#444">稀疏模式学习 / 固定，跳过 90%+ 计算</text>

  <!-- CSA -->
  <rect x="360" y="50" width="300" height="110" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.2"/>
  <text x="375" y="75" font-size="14" font-weight="700" fill="#1e40af">CSA · 压缩稀疏注意力</text>
  <text x="375" y="95" font-size="12" fill="#1e40af">DeepSeek V4</text>
  <text x="375" y="115" font-size="12" fill="#444">每 4 token 压缩成 1 个 KV 块</text>
  <text x="375" y="135" font-size="12" fill="#444">FP4 索引器选 top-k 块，局部滑窗兜底</text>

  <!-- HCA -->
  <rect x="190" y="190" width="300" height="110" rx="8" fill="#fff7ed" stroke="#f59e0b" stroke-width="1.2"/>
  <text x="205" y="215" font-size="14" font-weight="700" fill="#92400e">HCA · 重度压缩注意力</text>
  <text x="205" y="235" font-size="12" fill="#92400e">DeepSeek V4 上层</text>
  <text x="205" y="255" font-size="12" fill="#444">每 128 token 压缩成 1 个条目</text>
  <text x="205" y="275" font-size="12" fill="#444">压缩后对全部条目做稠密注意力</text>

  <!-- relationships -->
  <line x1="170" y1="160" x2="260" y2="190" stroke="#6b7280" stroke-dasharray="4" marker-end="url(#arr)"/>
  <text x="190" y="182" font-size="11" fill="#6b7280">再加压缩</text>
  <line x1="510" y1="160" x2="420" y2="190" stroke="#6b7280" stroke-dasharray="4" marker-end="url(#arr)"/>
  <text x="430" y="182" font-size="11" fill="#6b7280">压缩比↑128×</text>

  <!-- summary box -->
  <rect x="20" y="330" width="640" height="70" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="355" font-size="12.5" fill="#1a1a1a"><tspan font-weight="700">协作关系：</tspan>近处 128 token 走滑窗（不压缩）+ 中距 CSA（4× 压缩 + top-k）+ 远端 HCA（128× 压缩）。</text>
  <text x="35" y="380" font-size="12" fill="#6b7280">MSA 省计算、CSA 省 KV 且保选择、HCA 省到极致——三者组合才撑起 1M token。</text>
</svg>
<p class="cap">图：MSA / CSA / HCA 的动刀位置与压缩比对比。</p>
</div>

## 3. MSA：只选重要的看

**核心思想**：标准注意力的 O(n²) 来自「每个 query 看所有 key」。MSA 通过学习或固定模式，让每个 query 只看一小部分 key。

```text
Full Attention:  Q_n × K_1...K_n      → n² 次点积
MSA:             Q_n × K_selected     → 只算被选中的位置
```

MiniMax M3 的做法是在注意力内部引入**可学习稀疏门控**：模型自己决定哪些历史位置值得看。由于不需要压缩 KV，实现相对直接，但稀疏索引本身有额外开销。

**优点**：不损失原始 KV 精度，保留远距离细粒度信息（只要被选到）。
**缺点**：稀疏索引不规则，GPU 访存不连续；超长序列下索引开销会变大。

## 4. CSA：先压缩，再稀疏选择

**核心思想**：把 KV 按 4-token 块压缩成一个「块向量」，再用轻量 FP4 索引器对所有块打分，选 top-k 块做精确注意力，最后用 128-token 滑窗保底局部细节。

```text
原始 KV:  [t1][t2][t3][t4] [t5][t6][t7][t8] ...  → 长度 n
压缩后:   [c1]            [c2]            ...  → 长度 n/4
FP4 索引: Q 与每个 c 做点积 → top-k 块
精确计算: Q 只 attend top-k 块 + 最近 128 token
```

**关键点**：
- 压缩比 m=4，不算极端，信息损失可控；
- FP4 索引器极快，top-k 选择只占总时间很小一部分；
- 滑窗分支保留局部精度，避免短程细节被压坏。

**优点**：KV Cache 直接降到 1/4，top-k 后算力也大幅下降；规则块压缩利于 GPU 并行。
**缺点**：需要额外训练压缩函数和索引器；压缩函数设计不好会丢远距离语义。

## 5. HCA：压到极端，再用稠密注意力

**核心思想**：把 128-token 大块融合成 1 个 KV 条目（压缩比 128:1），然后对所有压缩条目做**稠密注意力**。

```text
原始 KV:  128 token  →  1 个目录条目
1M token  →  ~8000 个目录条目
Q 对所有 8000 条目做 softmax attention
```

为什么压缩后反而用**稠密**不用稀疏？因为 8000 条目的规模下，稠密 Kernel 的访存连续、warp 利用率高，实际比不规则稀疏更快。HCA 负责给模型一个「远方历史的粗轮廓」，具体细节再交给 CSA 或滑窗去补。

**优点**：KV 压缩到极致（1/128），1M token 也能放进显存；稠密注意力稳定高效。
**缺点**：单 token 信息被高度抽象，不适合依赖远距离精确细节的任务；必须和 CSA/滑窗配合。

## 6. 三者协作：V4 的分层注意力

<div class="fig">
<svg viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="DeepSeek V4 分层注意力：滑窗+CSA+HCA">
  <rect x="0" y="0" width="680" height="220" fill="none"/>
  <text x="20" y="28" font-size="14" font-weight="700" fill="#1a1a1a">DeepSeek V4：近/中/远三层注意力协作</text>

  <rect x="20" y="55" width="180" height="60" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="110" y="82" font-size="13" fill="#047857" text-anchor="middle">最近 128 token</text>
  <text x="110" y="102" font-size="11" fill="#047857" text-anchor="middle">标准滑窗 · 不压缩</text>

  <rect x="220" y="55" width="200" height="60" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="320" y="82" font-size="13" fill="#1d4ed8" text-anchor="middle">中距历史</text>
  <text x="320" y="102" font-size="11" fill="#1d4ed8" text-anchor="middle">CSA · 4× 压缩 + top-k</text>

  <rect x="440" y="55" width="220" height="60" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="550" y="82" font-size="13" fill="#b45309" text-anchor="middle">远端历史</text>
  <text x="550" y="102" font-size="11" fill="#b45309" text-anchor="middle">HCA · 128× 压缩 + 稠密</text>

  <line x1="200" y1="85" x2="218" y2="85" stroke="#6b7280"/>
  <line x1="420" y1="85" x2="438" y2="85" stroke="#6b7280"/>

  <rect x="20" y="140" width="640" height="60" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="165" font-size="12.5" fill="#1a1a1a">Query 到不同距离的历史，走不同分支；最终输出相加或拼接。</text>
  <text x="35" y="185" font-size="12" fill="#6b7280">结果：1M token 下 KV Cache ≈ V3.2 的 10%，每 token 算力 ≈ 27%。</text>
</svg>
<p class="cap">图：V4 把「滑窗 + CSA + HCA」三层叠在一起，远近分工。</p>
</div>

## 7. 选型速查表

| 场景 | 推荐机制 | 原因 |
|---|---|---|
| 局部精确建模 | 滑窗 / Full Attention | 不压缩，细节完整 |
| 中距选择型长上下文 | CSA | 压缩+选择，平衡精度与效率 |
| 远端概览 / 粗粒度记忆 | HCA | 压缩到极致，稠密计算快 |
| 想自己实现一个轻量长上下文模型 | 先滑窗 + CSA，再按需加 HCA | CSA 比 HCA 更容易调 |
| 需要 1M+ token 的 Agent / 视频历史 | 三层组合（V4 思路） | 单层机制都有短板，组合才能覆盖 |

## 8. 常见坑

1. **不要把 HCA 换成 Sparse**：8000 条目规模下稠密 Kernel 更规则，sparse 索引开销会反超。
2. **CSA 的压缩函数必须可微**：简单 mean-pooling 会丢语义，需用 gated / learned compression。
3. **FP4 索引器要 clamp**：数值范围小，点积前不 clamp 容易溢出。
4. **HCA 必须和 CSA 范围对齐**：如果 HCA 压缩范围 ≠ CSA 可选范围，会出现「目录里没有、却想索引到」的空洞。

## 9. 与投资主线的关联

这三种注意力改造共同指向一个结论：**长上下文成本正在快速下降**。对具身智能而言，意味着机器人 VLA 可以「记住」更长视频历史、做更复杂的任务规划，而不必受限于板载显存。对应标的：端侧推理芯片（地平线 09660.HK）、机器人 ETF 华夏（562500）、以及液冷/算力基建（TrendForce 渗透率 53%→60%）。
