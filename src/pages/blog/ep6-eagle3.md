---
title: EAGLE-3 深度解析：扔掉 feature 约束，让草稿模型第一次"吃数据"变快
description: 从 feature 预测为什么是 scaling 枷锁讲起，拆解 EAGLE-3 的 Training-Time Test 与多层特征融合，配三代对比表与 scaling 曲线。
pubDate: 2026-08-04
series: 推测解码手记
lang: zh
altLang: en
altHref: /en/blog/ep6-eagle3
layout: ../../layouts/BlogPost.astro
---

## 1. 为什么要单独看 EAGLE-3

前面 Ep1–Ep5 把推测采样（SD）的轮廓、自回归草稿的天花板、以及 DFlash / DSpark 两条新路线都讲过了。但有一个名字反复出现——**EAGLE**——它其实是整个"特征层草稿"流派的开山祖，DFlash/DSpark 在对比时都以它为基线（"比 EAGLE-3 快约 2.5×"）。

EAGLE 系列三代：

- **EAGLE-1**（ICML 2024）：在 feature 层做自回归，借目标模型的顶层特征猜得更准；
- **EAGLE-2**（EMNLP 2024）：按置信度动态调度草稿树，把计算预算分配得更聪明；
- **EAGLE-3**（2025）：**扔掉 feature 预测约束**，让草稿模型第一次出现 scaling law。

EAGLE-3 最了不起的地方不是"又快了一点"，而是它**证明了草稿模型也能像大模型一样从更多数据里持续受益**——这是投机采样领域此前从没有过的现象。

## 2. 复习：EAGLE 是怎么猜得更准的

标准 SD 用一个小 LLM 当草稿，小模型和大模型分布差距大，接受率 α 低（约 1.6–1.9×）。

EAGLE-1 的洞察是：**与其用独立小模型去猜 token，不如直接复用目标模型的中间特征**。它在 feature 层做自回归——把目标模型顶层 feature 和"前移一步的 token"拼起来，喂给一个只有 1 层的草稿 Decoder，预测下一个 feature，再用目标模型的 LM head 转成 token。这样做到了约 3–4×。

EAGLE-2 发现草稿模型的置信度与实际接受率高度校准，于是**动态决定草稿树形状**（简单位置少分叉、难位置多分叉），把加速比推到约 4.2×。

## 3. feature 预测约束：为什么是 scaling 枷锁

EAGLE-1/2 的草稿模型训练时受**两个 loss** 约束：

- **feature loss**（SmoothL1）：让草稿输出逼近目标模型的顶层 feature；
- **token loss**（CrossEntropy）：让 token 猜对。

作者尝试把训练数据翻 8 倍时，发现加速比**几乎不涨**。原因在 feature loss：

> 草稿模型只有 1 层 Decoder，容量极小。feature loss 逼着它把大量容量花在"几何拟合"上——把输出向量在空间里摆得接近目标模型的 feature——而不是花在"怎么把 token 预测最准"这个最终目标上。

换句话说，feature 预测是个**过强的正则**：数据少时帮它泛化，数据多时反而成了枷锁。更糟的是，feature 约束还**把输入端也锁死**成"只能用顶层特征"（因为顶层 feature 和 next-token logits 一一对应），信息高度特化。

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#374151">图 1：草稿模型训练数据量 vs 加速比（LLaMA-Instruct 3.1 8B, MT-bench，示意）</text>
  <line x1="100" y1="40" x2="100" y2="250" stroke="#9ca3af" stroke-width="1"/>
  <line x1="100" y1="250" x2="540" y2="250" stroke="#9ca3af" stroke-width="1"/>
  <text x="92" y="254" font-size="10" fill="#6b7280" text-anchor="end">3.5</text>
  <text x="92" y="191" font-size="10" fill="#6b7280" text-anchor="end">4.5</text>
  <text x="92" y="127" font-size="10" fill="#6b7280" text-anchor="end">5.5</text>
  <text x="92" y="64" font-size="10" fill="#6b7280" text-anchor="end">6.5</text>
  <polyline points="120,218 220,212 330,212 450,212" fill="none" stroke="#3b82f6" stroke-width="2"/>
  <polyline points="120,142 220,117 330,98 450,85" fill="none" stroke="#eab308" stroke-width="2.5"/>
  <circle cx="120" cy="218" r="3" fill="#3b82f6"/><circle cx="220" cy="212" r="3" fill="#3b82f6"/><circle cx="330" cy="212" r="3" fill="#3b82f6"/><circle cx="450" cy="212" r="3" fill="#3b82f6"/>
  <circle cx="120" cy="142" r="3" fill="#eab308"/><circle cx="220" cy="117" r="3" fill="#eab308"/><circle cx="330" cy="98" r="3" fill="#eab308"/><circle cx="450" cy="85" r="3" fill="#eab308"/>
  <text x="120" y="268" font-size="10" fill="#6b7280" text-anchor="middle">1×</text>
  <text x="220" y="268" font-size="10" fill="#6b7280" text-anchor="middle">2×</text>
  <text x="330" y="268" font-size="10" fill="#6b7280" text-anchor="middle">4×</text>
  <text x="450" y="268" font-size="10" fill="#6b7280" text-anchor="middle">8×</text>
  <text x="320" y="286" font-size="10" fill="#6b7280" text-anchor="middle">训练数据量（相对 ShareGPT 的倍数）</text>
  <rect x="498" y="58" width="14" height="10" fill="#3b82f6"/><text x="518" y="67" font-size="10" fill="#374151">EAGLE / EAGLE-2（持平）</text>
  <rect x="498" y="78" width="14" height="10" fill="#eab308"/><text x="518" y="87" font-size="10" fill="#374151">EAGLE-3（随数据上升）</text>
  <text x="16" y="290" font-size="10" fill="#6b7280">去掉 feature 预测约束后，草稿模型第一次出现 scaling law：数据越多加速比越高，最高 6.5×。</text>
</svg>
</div>

## 4. EAGLE-3 的两个改动

两个设计互相配合，缺一不可。

### 4.1 扔掉 feature 预测，用 Training-Time Test 直接预测 token

核心洞察：**feature 预测是手段不是目的**——它的作用只是让单步训练能泛化到多步推理，代价是过度约束了输出。EAGLE-3 直接让草稿模型预测 token，并在**训练时就模拟多步生成过程**（把目标模型的 LM head、采样动作都接进草稿模型的训练回路）。这一步被称为 **Training-Time Test**：草稿模型在训练阶段就把"测试时会怎么一步步生成"练熟，从而消掉了 EAGLE 系列一直存在的"训练–推理分布不一致"。

<div class="fig">
<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#374151">图 2：EAGLE-2（带 feature 约束） vs EAGLE-3（Training-Time Test）</text>
  <text x="175" y="42" font-size="12" font-weight="700" fill="#1e3a8a" text-anchor="middle">EAGLE-2</text>
  <text x="505" y="42" font-size="12" font-weight="700" fill="#854d0e" text-anchor="middle">EAGLE-3</text>

  <rect x="60" y="58" width="230" height="38" rx="5" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="175" y="81" font-size="11" fill="#1e3a8a" text-anchor="middle">输入：feature f_t + 前移 token t₍ₜ₊₁₎</text>
  <line x1="175" y1="96" x2="175" y2="116" stroke="#9ca3af" stroke-width="1"/>
  <rect x="60" y="116" width="230" height="38" rx="5" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="175" y="139" font-size="11" fill="#1e3a8a" text-anchor="middle">草稿模型（1 层 Decoder）</text>
  <line x1="175" y1="154" x2="175" y2="174" stroke="#9ca3af" stroke-width="1"/>
  <rect x="60" y="174" width="230" height="42" rx="5" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="175" y="195" font-size="11" fill="#1e3a8a" text-anchor="middle">预测 f₍ₜ₊₁₎</text>
  <text x="175" y="210" font-size="10" fill="#1e3a8a" text-anchor="middle">（SmoothL1 拟合目标 feature）</text>
  <line x1="175" y1="216" x2="175" y2="236" stroke="#9ca3af" stroke-width="1"/>
  <rect x="60" y="236" width="230" height="34" rx="5" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="175" y="257" font-size="11" fill="#1e3a8a" text-anchor="middle">LM head → token</text>
  <text x="175" y="298" font-size="10" fill="#6b7280" text-anchor="middle">单步训练→泛化多步；feature 约束锁死容量与输入</text>

  <rect x="390" y="58" width="230" height="38" rx="5" fill="#fef9c3" stroke="#eab308"/>
  <text x="505" y="81" font-size="11" fill="#854d0e" text-anchor="middle">输入：多层融合 feature + prev token</text>
  <line x1="505" y1="96" x2="505" y2="116" stroke="#9ca3af" stroke-width="1"/>
  <rect x="390" y="116" width="230" height="38" rx="5" fill="#fef9c3" stroke="#eab308"/>
  <text x="505" y="139" font-size="11" fill="#854d0e" text-anchor="middle">草稿模型（容量可放大）</text>
  <line x1="505" y1="154" x2="505" y2="174" stroke="#9ca3af" stroke-width="1"/>
  <rect x="390" y="174" width="230" height="42" rx="5" fill="#fef9c3" stroke="#eab308"/>
  <text x="505" y="195" font-size="11" fill="#854d0e" text-anchor="middle">直接预测 token</text>
  <text x="505" y="210" font-size="10" fill="#854d0e" text-anchor="middle">（CrossEntropy loss）</text>
  <line x1="505" y1="216" x2="505" y2="236" stroke="#9ca3af" stroke-width="1"/>
  <rect x="390" y="236" width="230" height="46" rx="5" fill="#fef9c3" stroke="#eab308"/>
  <text x="505" y="256" font-size="10" fill="#854d0e" text-anchor="middle">训练时模拟多步生成：</text>
  <text x="505" y="271" font-size="10" fill="#854d0e" text-anchor="middle">把 LM head + 采样接进训练回路</text>
  <text x="505" y="298" font-size="10" fill="#6b7280" text-anchor="middle">训练即"演练"测试时多步生成，消掉分布差</text>
</svg>
</div>

### 4.2 多层特征融合（multi-layer feature fusion）

既然 feature 约束没了，输入端也获得自由——不再只能用顶层 feature，而是**融合目标模型的低 / 中 / 高层特征**，拿到更丰富的语义上下文喂给草稿模型。这一点和"直接预测 token"互为前提：没有 feature 约束，才敢换输入。

## 5. 三代对比表

| 维度 | EAGLE-1 | EAGLE-2 | EAGLE-3 |
|---|---|---|---|
| 草稿方式 | feature 层自回归 | 动态草稿树 | 直接 token + 多层融合 |
| 是否受 feature 约束 | 是 | 是 | **否** |
| 输入端 | 顶层 feature | 顶层 feature | 低/中/高层融合 |
| 是否出现 scaling law | 否 | 否 | **是** |
| 典型加速比 | ~3–4× | ~4.2× | **最高 6.5×** |

## 6. 效果数据

| 指标 | 数值 |
|---|---|
| 最高加速比 | **6.5×**（Vicuna 13B, HumanEval, T=0） |
| 相对 EAGLE-2 | 延迟再降约 **1.4×** |
| SGLang 吞吐（batch=64, H100） | **+1.38×** |
| 兼容 | 完全兼容 EAGLE-2 的草稿树 |

## 7. 小结

EAGLE 三代的演进一句话概括：

> EAGLE-1 解决"怎么借大模型信息猜得更准" → EAGLE-2 解决"怎么把计算预算分配得更聪明" → EAGLE-3 解决"怎么让草稿模型从更多数据持续受益"。

它给整个投机采样领域一个重要启示：**草稿模型不是只能是个小玩具，只要去掉错误的约束、给它更丰富的输入和更大的容量，它也能 scaling**。这也是为什么 DFlash / DSpark 在对比时都以 EAGLE-3 为基线——它定义了"特征层草稿"这条路线的天花板。

> 本篇属于「推测解码手记」系列 Ep6。前情：[DFlash 深度解析](../ep3-dflash)、[DSpark 深度解析](../ep4-dspark)、[DFlash vs DSpark](../ep5-dflash-vs-dspark)；回到系列首页见[推测解码手记](../)。
