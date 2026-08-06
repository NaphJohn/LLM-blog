---
title: 前沿架构解码手记（四）：DeepSeek V4——极致压缩与高效训练
description: 路线图压轴。DeepSeek V4 走最激进的一条路：直接压缩 KV 本身——CSA（压缩稀疏注意力，4-token KV 块 + top-k）+ HCA（重度压缩注意力，128×）混合，外加 128-token 不压缩滑窗兜底；1M 上下文下每 token 算力约 V3.2 的 27%、KV 约 10%。训练侧换 Muon 优化器、OPD 蒸馏、FP8 训练 / FP4 专家参数。Pro 1.6T / Flash 284B（MoE），MIT 开源。
pubDate: 2026-08-06
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa4-deepseek-v4
layout: ../../layouts/BlogPost.astro
---

## 0. 压轴：最激进的"压缩"路线

前面两家：Kimi K3 改注意力**内部**（差分），MiniMax M3 改注意力**范围**（稀疏）。DeepSeek V4 更进一步——**直接压缩 KV Cache 本身**，并把训练也重做了一遍。这是路线图里改造最重的一站。

## 1. 注意力：CSA + HCA 混合

- **CSA（Compressed Sparse Attention，压缩稀疏注意力）**：把 KV 按 **4-token 块**压缩，再对块做 top-k 选择——既压缩又保留稀疏选择。
- **HCA（Heavily Compressed Attention，重度压缩注意力）**：压缩比高达 **128×**，用于处理"最远端、最该压缩"的历史。
- **不压缩滑窗分支**：最近 **128 token** 走普通注意力，保证局部精细建模不被压坏。

三者混合，形成"近处精确、远处压缩"的分层结构。

<div class="fig">
<svg viewBox="0 0 680 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="CSA+HCA 混合注意力结构">
  <rect x="0" y="0" width="680" height="240" fill="none"/>
  <text x="20" y="28" font-size="14" font-weight="700" fill="#1a1a1a">Query 位置 → 历史 KV 的分层处理</text>
  <rect x="20" y="60" width="160" height="50" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="100" y="90" font-size="13" fill="#047857" text-anchor="middle">最近 128 token</text>
  <rect x="200" y="60" width="160" height="50" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="280" y="90" font-size="13" fill="#1d4ed8" text-anchor="middle">CSA（4-token 块 + top-k）</text>
  <rect x="380" y="60" width="200" height="50" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="480" y="82" font-size="13" fill="#b45309" text-anchor="middle">HCA（重度压缩 128×）</text>
  <text x="480" y="100" font-size="11" fill="#b45309" text-anchor="middle">最远端历史</text>
  <line x1="180" y1="85" x2="198" y2="85" stroke="#6b7280"/>
  <line x1="360" y1="85" x2="378" y2="85" stroke="#6b7280"/>
  <rect x="20" y="140" width="640" height="70" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="40" y="166" font-size="12.5" fill="#1a1a1a">混合效果（官方口径，相对 V3.2 @ 1M）：</text>
  <text x="40" y="188" font-size="12.5" fill="#1a1a1a">每 token 算力 ≈ 27% · KV Cache ≈ 10% · 默认 1M 上下文</text>
  <text x="40" y="205" font-size="11" fill="#6b7280">近处精确（滑窗）+ 中距压缩（CSA）+ 远端极压（HCA）</text>
</svg>
  <p class="cap">图：DeepSeek V4 的分层注意力——近处不压缩，越远压得越狠。</p>
</div>

## 2. 训练侧：Muon + OPD + 低比特

- **Muon 优化器**替代 AdamW：对正交/矩矩阵用 Muon（动量 + 正交化），更省显存、收敛更稳，是 V4 训练降本的关键。
- **OPD（On-Policy Distillation，同策略蒸馏）**：用主模型在线蒸馏"领域专家"小模型，把专项能力压缩进统一模型，避免离线蒸馏的分布偏移。
- **FP8 训练 / FP4 专家参数**：MoE 专家权重用 FP4，进一步压训练/存储成本。
- **MegaMoE EP 通信**：专家并行通信优化，提速 1.5–1.73×；另有**磁盘 KV Cache**、TileLang DSL、昇腾 NPU 适配。

## 3. 两个版本与定价

| 版本 | 参数（MoE） | 定位 | 定价（每 M tokens） |
|---|---|---|---|
| **Flash** | 284B / 激活 13B | 轻量、低延迟 | 输入 1 元 / 输出 2 元（命中缓存输入 0.2 元） |
| **Pro** | 1.6T / 激活 49B | 强推理 | 输入 12 元 / 输出 24 元 |

两者默认 1M 上下文，最大输出约 384K；**MIT 许可开源**。

## 4. 基准（官方口径）

- Codeforces 评分 **3206**（V4-Pro-Max）；
- SWE-Verified **80.6%**；
- Toolathlon **51.8**；
- Putnam 2025 **满分**。

偏代码、agentic 工具调用、竞赛数学——与"极致压缩换来长上下文 + 强推理"的定位一致。

## 5. 三家合流：共同终点

把四篇串起来看：

| 路线 | 代表 | 注意力改造 | 1M 上下文代价 |
|---|---|---|---|
| 重设计 | Kimi K3 | KDA 差分 + 门控(3:1) + 残差 | 靠 MLA 省 KV |
| 稀疏化 | MiniMax M3 | MSA 稀疏选择 | 每 token 算力≈M2.7 的 1/20 |
| 压缩 | DeepSeek V4 | CSA+HCA 压 KV | 每 token 算力≈V3.2 的 27%、KV≈10% |

<div class="keybox">
<strong>结论：</strong>三家的差异在"动刀位置"（内部 / 范围 / KV 本身），但<strong>终点完全一致——MoE + 1M 上下文 + 注意力改造</strong>。2026 之后，稠密大模型基本退场，"长上下文 + 低 KV"成为前沿模型的及格线。
</div>

## 6. 投资视角

- **推理成本曲线继续下探**：KV 与每 token 算力被压到 1/10~1/20 量级，直接利好**端侧 / 机载实时模型**与**长程 Agent**——具身智能的成本拐点更近。
- **能力商品化加速**：三家全开源（MIT/Apache），"前沿模型能力"不再是壁垒，**壁垒转移到数据、工程、生态与场景**。
- **训练降本（Muon / FP4 / OPD）**意味着小团队也能训出强模型，利好**国产算力与垂直模型**生态。
