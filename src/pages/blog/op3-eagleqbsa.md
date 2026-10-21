---
title: '算子讲解手记（三）：昇腾 EagleQBSA——量化×稀疏联合注意力，整网注意力最高加速 7.6×'
description: '长视频生成把注意力推到百万级序列：MindIE SD 在昇腾 950 上落地 EagleQBSA——QK 按块 INT8 + V/P 按通道 FP8 的分布感知混合精度量化，叠加 AdaBSA 块稀疏，再用 Cube→Vector 直搬与 Softmax 归约转置抠掉 Vector 瓶颈。Wan2.2 端到端 1.6×、MiniMax-H3 1.52×、单算子最高 14.7×，VBench 五维误差 <1%。'
pubDate: 2026-10-21
series: 算子讲解手记
lang: zh
altLang: en
altHref: /en/blog/op3-eagleqbsa
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

量化和稀疏是加速注意力的两条正交路线，但**直接叠会互相拆台**：稀疏选择改变数据分布，放大量化误差；低比特矩阵乘把 Cube 抠快之后，Softmax 所在的 Vector 侧又成了新瓶颈。昇腾 MindIE SD 的 EagleQBSA（Eagle Quant Block Sparse Attention）给出的答案是**分布感知的混合精度配方（QK 按块 INT8、V/P 按通道 FP8）+ 昇腾 950 的硬件亲和改造（Cube→Vector 直搬时转置、Softmax 归约列化、位运算近似 FP8 转换）**，最终整网注意力模块最高加速 7.6×，Wan2.2 端到端 1.6×、MiniMax-H3 端到端 1.52×，VBench 五维精度误差 <1%。

## 1. 背景：视频生成把注意力逼到墙角

DiT 类视频生成模型（Wan2.2、MiniMax-H3 等）在**时空联合的 token 序列**上做全注意力：5s、720p 级别的视频，patch 化之后序列长度轻易冲到**十万到百万级**。此时注意力的开销是双重的：

- **计算**：$O(N^2)$ 的 QK^T 与 PV 矩阵乘，长序列下吃掉绝大部分 FLOPs；
- **访存**：K/V 的读取量随序列平方增长，HBM 带宽先于算力饱和。

于是两类优化成为刚需——**稀疏注意力**（跳过不重要的 KV 连接）和**量化注意力**（QKV 压到 FP8/INT8 等低精度）。本篇文章拆解的 EagleQBSA，就是把两者在**算子级**联合起来的工程实践：基于 CANN 的 BlockSparseAttention 算子开发，面向昇腾 950 系列硬件，随 MindIE SD 发布。

## 2. 两条正交的加速路线

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="稀疏注意力与量化注意力示意">
  <defs>
    <marker id="arr31" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">稀疏 vs 量化：两种砍法，作用于注意力的不同维度</text>

  <!-- 左：稀疏 -->
  <text x="20" y="58" font-size="13" font-weight="700" fill="#047857">稀疏注意力：砍「连接数」</text>
  <g>
    <!-- 6x6 块矩阵 -->
    <rect x="30" y="70" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="64" y="70" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="98" y="70" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="132" y="70" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="166" y="70" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="200" y="70" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>

    <rect x="30" y="100" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="64" y="100" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="98" y="100" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="132" y="100" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="166" y="100" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="200" y="100" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>

    <rect x="30" y="130" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="64" y="130" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="98" y="130" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="132" y="130" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="166" y="130" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="200" y="130" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>

    <rect x="30" y="160" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="64" y="160" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="98" y="160" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="132" y="160" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="166" y="160" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="200" y="160" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
  </g>
  <text x="30" y="215" font-size="12" fill="#047857">■ 计算：只算重要块（注意力权重本就集中在少数连接）</text>
  <text x="30" y="235" font-size="12" fill="#6b7280">■ 跳过：不访问对应 KV，省计算也省带宽</text>
  <text x="30" y="262" font-size="11" fill="#6b7280">块粒度： MindIE SD 已集成自适应块稀疏 AdaBSA</text>

  <!-- 分隔线 -->
  <line x1="345" y1="50" x2="345" y2="280" stroke="#e5e7eb" stroke-width="1.5"/>

  <!-- 右：量化 -->
  <text x="375" y="58" font-size="13" font-weight="700" fill="#1d4ed8">量化注意力：砍「每个数的位宽」</text>

  <text x="375" y="92" font-size="12" fill="#1a1a1a">FP16 / BF16</text>
  <rect x="490" y="80" width="160" height="16" rx="3" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="600" y="93" font-size="10" fill="#1e40af" text-anchor="middle">16 bit</text>

  <text x="375" y="122" font-size="12" fill="#1a1a1a">FP8 (E4M3)</text>
  <rect x="490" y="110" width="80" height="16" rx="3" fill="#bfdbfe" stroke="#3b82f6"/>
  <text x="530" y="123" font-size="10" fill="#1e40af" text-anchor="middle">8 bit</text>

  <text x="375" y="152" font-size="12" fill="#1a1a1a">INT8</text>
  <rect x="490" y="140" width="80" height="16" rx="3" fill="#93c5fd" stroke="#3b82f6"/>
  <text x="530" y="153" font-size="10" fill="#1e40af" text-anchor="middle">8 bit</text>

  <text x="375" y="182" font-size="12" fill="#1a1a1a">FP4 / INT4</text>
  <rect x="490" y="170" width="40" height="16" rx="3" fill="#60a5fa" stroke="#3b82f6"/>
  <text x="510" y="183" font-size="10" fill="#1e40af" text-anchor="middle">4 bit</text>

  <text x="375" y="215" font-size="12" fill="#1a1a1a">收益：低精度矩阵乘算力 ↑</text>
  <text x="375" y="235" font-size="12" fill="#1a1a1a">显存占用 ↓ · 访存带宽压力 ↓</text>
  <text x="375" y="262" font-size="11" fill="#6b7280">代价：量化误差，与数据分布强相关</text>
</svg>
<p class="cap">图 1：稀疏与量化的正交性——一个减少「算哪些连接」，一个减少「每个连接用多少位」。二者管线独立，理论上收益可叠加。</p>
</div>

- **稀疏路线**在本站的上一篇前沿架构手记里见过模型侧版本（MiniMax M3 的 MSA，见 fa3）；EagleQBSA 用的是**推理侧算子级**的自适应块稀疏 AdaBSA——同一个思想，落点不同。
- **量化路线**的完整谱系见本系列 op2（FP8/INT4/AWQ·GPTQ/FP4 QAT/KV Cache 量化）。op2 讲「有哪些精度档位」，本篇讲「在一个真实算子里，每个矩阵该选哪一档、为什么」。

## 3. 直接叠为什么会翻车

稀疏和量化管线天然正交，但工程上直接组合有两个坑：

**算法层：稀疏选择改变数据分布，进一步放大量化误差。**
注意力的 P = softmax(QK^T) 是**逐块动态**的：稀疏 mask 每块不同，被保留块的 score 分布在 mask 前后并不一致。如果缩放因子按稀疏前的分布算，量化误差会被 softmax 的非线性进一步放大，直接冲击视觉模型的生成质量（画面崩坏、主体漂移）。

**实现层：低比特矩阵乘提速后，Vector 侧成为新瓶颈。**
QK/PV 矩阵乘交给 Cube 单元的低精度通路之后，Softmax（求 max、求和、指数、归一化）、P 的量化、反量化这些 **Vector 单元**的访存与计算占比反而上升。收益不再取决于矩阵乘吞吐，而取决于 Vector 链路能不能跟上。

EagleQBSA 的两大设计正对应这两个坑：**分布感知的混合精度量化**（解算法层）、**昇腾 950 硬件亲和实现**（解实现层）。

## 4. 分布感知的量化配方：QK INT8 / PV FP8

对视频生成任务整网 QKV 矩阵的实测分析揭示了三个关键事实，直接决定了量化格式的选型：

**事实一：QK 在同一块内部分布集中（时空局部性）。**
尽管 QK 在整个序列上分布差异较大，但同一块内部的局部序列数据分布相对集中——因此 QK 适合**在序列维度按块量化**（per-block scale）。

**事实二：QK 用 INT8 比 FP8 误差小一个量级。**
对整网 QK 分别做 FP8 和 INT8 量化注意力计算，对比 BF16 基线：**QK FP8 量化的单算子误差是 INT8 的 3~4 倍**。INT8 按块量化的细粒度 scale 恰好补上了 FP8 动态范围大、但尾数精度在块内不均的短板。

**事实三：V 分布离散、动态范围超 INT8 适配能力，P 跟随 V。**
V 矩阵数据分布离散，动态范围超出 INT8 的适配区间，适合 **FP8 按通道量化**（per-channel scale，反量化步骤效率高）；P 矩阵跟随 V 采用 FP8，并以 FP8 完成 PV 矩阵乘。实测 **P/V INT8 量化相比 P/V FP8，单算子精度误差增大 2~3 个数量级**。

最终配方一句话：**QK INT8（按块）/ PV FP8（V 按通道、P 跟随）的混合精度量化**，叠加 AdaBSA 块稀疏。量化公式的代码骨架：

```python
# QK: 序列维度按块 INT8（块内分布集中，时空局部性）
q_int8, q_scale = quant_per_block_int8(q, block=128)   # scale: [num_blocks]
k_int8, k_scale = quant_per_block_int8(k, block=128)
scores = (q_int8 @ k_int8.T) * (q_scale * k_scale)     # INT8 矩阵乘 + 反量化 scale

# AdaBSA 块稀疏 mask：只保留重要块，其余块不访问 KV
scores = apply_block_sparse_mask(scores, sparsity=0.63)

# Softmax（Vector 侧，昇腾 950 上做归约列化优化，见第 5 节）
p = softmax(scores)

# P 量化到 FP8（位运算近似转换，见第 5 节），V 按通道 FP8
p_fp8 = quant_fp8_bitwise(p)
v_fp8, v_scale = quant_per_channel_fp8(v)              # scale: [channels]
out = (p_fp8.to_fp16() @ v_fp8.to_fp16()) * v_scale    # FP8 PV 矩阵乘 + 反量化
```

<div class="fig">
<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="EagleQBSA 计算流程：QK INT8、块稀疏、Softmax、P FP8、PV FP8">
  <defs>
    <marker id="arr32" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">EagleQBSA 计算流程：蓝色 = Cube 低精度矩阵乘，琥珀色 = Vector 侧</text>

  <!-- Q K V 输入 -->
  <rect x="20" y="60" width="70" height="40" rx="6" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="55" y="84" font-size="13" fill="#1a1a1a" text-anchor="middle">Q / K / V</text>
  <text x="55" y="116" font-size="10" fill="#6b7280" text-anchor="middle">BF16 输入</text>

  <!-- 量化盒子 -->
  <rect x="120" y="60" width="110" height="40" rx="6" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="175" y="78" font-size="12" fill="#1e40af" text-anchor="middle">QK 按块 INT8</text>
  <text x="175" y="95" font-size="10" fill="#1e40af" text-anchor="middle">块内分布集中</text>

  <rect x="120" y="150" width="110" height="40" rx="6" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="175" y="168" font-size="12" fill="#1e40af" text-anchor="middle">V 按通道 FP8</text>
  <text x="175" y="185" font-size="10" fill="#1e40af" text-anchor="middle">分布离散、动态范围大</text>

  <!-- QK matmul -->
  <rect x="265" y="60" width="110" height="40" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
  <text x="320" y="78" font-size="12" fill="#1e40af" text-anchor="middle">INT8 QK^T</text>
  <text x="320" y="95" font-size="11" fill="#1e40af" text-anchor="middle">Cube 矩阵乘</text>

  <!-- dequant + sparse -->
  <rect x="265" y="115" width="110" height="40" rx="6" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="320" y="133" font-size="12" fill="#92400e" text-anchor="middle">反量化 × AdaBSA</text>
  <text x="320" y="150" font-size="10" fill="#92400e" text-anchor="middle">块稀疏 mask</text>

  <!-- Softmax -->
  <rect x="410" y="115" width="110" height="40" rx="6" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="465" y="133" font-size="12" fill="#92400e" text-anchor="middle">Softmax</text>
  <text x="465" y="150" font-size="10" fill="#92400e" text-anchor="middle">归约列化优化</text>

  <!-- P FP8 -->
  <rect x="410" y="180" width="110" height="40" rx="6" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="465" y="198" font-size="12" fill="#92400e" text-anchor="middle">P → FP8</text>
  <text x="465" y="215" font-size="10" fill="#92400e" text-anchor="middle">位运算近似转换</text>

  <!-- PV matmul -->
  <rect x="555" y="150" width="105" height="40" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
  <text x="607" y="168" font-size="12" fill="#1e40af" text-anchor="middle">FP8 PV</text>
  <text x="607" y="185" font-size="11" fill="#1e40af" text-anchor="middle">Cube 矩阵乘</text>

  <!-- 输出 -->
  <rect x="555" y="225" width="105" height="40" rx="6" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="607" y="249" font-size="13" fill="#1a1a1a" text-anchor="middle">输出 O</text>

  <!-- 箭头 -->
  <line x1="90" y1="80" x2="118" y2="80" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr32)"/>
  <line x1="90" y1="170" x2="118" y2="170" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr32)"/>
  <line x1="230" y1="80" x2="263" y2="80" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr32)"/>
  <line x1="320" y1="100" x2="320" y2="113" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr32)"/>
  <line x1="375" y1="135" x2="408" y2="135" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr32)"/>
  <line x1="465" y1="155" x2="465" y2="178" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr32)"/>
  <line x1="520" y1="200" x2="553" y2="185" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr32)"/>
  <line x1="175" y1="190" x2="175" y2="225" stroke="#4b5563" stroke-width="1.5"/>
  <line x1="175" y1="225" x2="607" y2="225" stroke="#4b5563" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="607" y1="225" x2="607" y2="192" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr32)"/>
  <line x1="607" y1="190" x2="607" y2="223" stroke="none"/>

  <text x="20" y="290" font-size="11" fill="#6b7280">要点：QK 与 PV 两段矩阵乘全部走低精度 Cube 通路；Vector 侧只保留 Softmax 与 P 量化，且被专项优化（见图 3）。</text>
</svg>
<p class="cap">图 2：EagleQBSA 的计算流程。QK INT8 / PV FP8 的混合精度不是拍脑袋，而是第 4 节三个实测事实的直接结果。</p>
</div>

## 5. 昇腾 950 硬件亲和：把 Vector 侧抠到极致

低精度量化把矩阵乘交给 Cube 之后，Vector 单元的访存与计算成为新瓶颈。昇腾 950 系列提供了两个关键硬件能力：**Cube 与 Vector 之间可直接搬运数据**（无需绕回内存）、**寄存器级（Regbase）Vector 开发能力**。EagleQBSA 的三个优化全部建立在这两点上：

**优化一：搬运的同时做转置，Softmax 归约「列化」。**
算子在从 Cube 向 Vector 搬运矩阵乘结果的同时完成矩阵转置。Softmax 本来需要对**一行数据**做归约（求最大值、求和）——一次只处理一行，数据并行程度低；转置之后，行归约变成**多列数据的两两归约**（树形折半），Vector 的并行度被最大限度压榨出来。

<div class="fig">
<svg viewBox="0 0 680 290" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Softmax 归约优化：行归约转为多列两两归约">
  <defs>
    <marker id="arr33" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Softmax 归约优化：Cube→Vector 搬运时转置，把行归约变成多列两两归约</text>

  <!-- 左：原始布局，行归约 -->
  <text x="20" y="56" font-size="13" font-weight="700" fill="#92400e">优化前：按行归约（并行度低）</text>
  <rect x="30" y="70" width="200" height="26" rx="3" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="130" y="87" font-size="11" fill="#92400e" text-anchor="middle">row i：max / sum 顺序归约一行</text>
  <rect x="30" y="100" width="200" height="26" rx="3" fill="#f9fafb" stroke="#e5e7eb"/>
  <rect x="30" y="130" width="200" height="26" rx="3" fill="#f9fafb" stroke="#e5e7eb"/>
  <rect x="30" y="160" width="200" height="26" rx="3" fill="#f9fafb" stroke="#e5e7eb"/>
  <line x1="130" y1="96" x2="130" y2="126" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3"/>
  <text x="30" y="205" font-size="11" fill="#6b7280">Vector 一次归约只吃一行数据，</text>
  <text x="30" y="222" font-size="11" fill="#6b7280">lane 利用率上不去</text>

  <!-- 中：转置 -->
  <text x="270" y="56" font-size="13" font-weight="700" fill="#1a1a1a">搬运时转置</text>
  <rect x="285" y="80" width="60" height="90" rx="6" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="315" y="120" font-size="11" fill="#1e40af" text-anchor="middle">S^T</text>
  <text x="315" y="138" font-size="10" fill="#1e40af" text-anchor="middle">直搬</text>
  <line x1="230" y1="128" x2="283" y2="125" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr33)"/>

  <!-- 右：列化两两归约 -->
  <text x="390" y="56" font-size="13" font-weight="700" fill="#047857">优化后：多列两两归约（树形折半）</text>
  <rect x="400" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <rect x="426" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <rect x="452" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <rect x="478" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <rect x="504" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <rect x="530" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <text x="556" y="120" font-size="11" fill="#047857">col 0..N</text>
  <!-- 两两归约树 -->
  <line x1="411" y1="70" x2="411" y2="52" stroke="#10b981" stroke-width="1.5"/>
  <line x1="437" y1="70" x2="437" y2="52" stroke="#10b981" stroke-width="1.5"/>
  <line x1="424" y1="52" x2="424" y2="40" stroke="#10b981" stroke-width="1.5"/>
  <line x1="463" y1="70" x2="463" y2="52" stroke="#10b981" stroke-width="1.5"/>
  <line x1="489" y1="70" x2="489" y2="52" stroke="#10b981" stroke-width="1.5"/>
  <line x1="476" y1="52" x2="476" y2="40" stroke="#10b981" stroke-width="1.5"/>
  <line x1="424" y1="40" x2="476" y2="40" stroke="#10b981" stroke-width="1.5"/>
  <text x="600" y="45" font-size="11" fill="#047857">两两</text>
  <text x="600" y="60" font-size="11" fill="#047857">折半</text>
  <text x="400" y="195" font-size="11" fill="#6b7280">同一时刻多列并行做 max/sum 两两归约，</text>
  <text x="400" y="212" font-size="11" fill="#6b7280">log₂N 步完成，Vector 算力被压满</text>

  <text x="20" y="262" font-size="11" fill="#6b7280">注：Softmax = max 归约 → 指数 → sum 归约 → 归一化；归约列化后两步归约都受益。</text>
</svg>
<p class="cap">图 3：Softmax 的归约列化。转置在搬运途中「免费」完成，换来的是 Vector 归约并行度的质变。</p>
</div>

**优化二：P 矩阵量化用位运算近似 FP16→FP8。**
相比标准的 FP16→FP8 数值转换（指数偏移 + 尾数截断 + 溢出处理），EagleQBSA 通过**位运算**实现近似转换，开销更低，进一步缓解 Vector 计算瓶颈。

**优化三：Regbase 寄存器级开发。**
配合昇腾 950 的 Regbase 能力，Vector 侧 kernel 直接在寄存器级组织数据，减少中间落地的访存。

## 6. 性能与精度实测

**单算子性能**（B=1、N=40、D=128，覆盖 8K–128K 序列长度）：

| 对比基准 | 配置 | 加速比 |
|---|---|---|
| BF16 块稀疏 BSA 算子 | 同等稀疏率 | **~1.6×** |
| 非稀疏浮点注意力 FaScore | 64K 序列、0.9 稀疏率 | **最高 14.7×** |

注意第二行的含义：**稀疏率 0.9 时相对稠密浮点注意力最高 14.7×**——这是「稀疏砍连接 × 量化砍位宽」两重收益叠加后的单算子理论上限区间；同稀疏率下相对 BF16 BSA 的 1.6× 才是纯量化贡献。

**端到端性能**（仅替换自注意力模块为 EagleQBSA，线性层不变）：

| 模型 | 任务 | 稀疏率 | 端到端加速 |
|---|---|---|---|
| Wan2.2-T2V-A14B | 文生视频 | 0.63 | **1.64×** |
| Wan2.2-I2V-A14B | 图生视频 | 0.63 | **1.61×** |
| MiniMax-H3 | 文生音视频 | 0.8 | **1.52×** |

整网注意力模块最高加速 **7.6×**，5s 视频生成任务端到端 1.5~1.6×——注意力在整网中的时间占比决定了端到端收益的上限，这也是「算子很快、整网打折」的典型分布。

**生成质量**（VBench 五维：主体一致性 / 成像质量 / 美学质量 / 整体一致性 / 运动平滑度）：

- 两类任务相对 BF16 基线的**整体精度误差均 <1%**；
- 文生视频用 48 条 `t2v_sora` 文本输入评测；
- 相同提示词下，EagleQBSA 与基线的画面主体和风格保持一致；
- ⚠️ 官方口径：结果仅反映本次测试配置，不代表所有内容和模型均无精度损失。

## 7. 串联：这篇在站内知识地图的哪个位置

- **op2（量化部署谱系）→ op3**：op2 给出 FP8/INT4/FP4 的档位地图，op3 用真实算子演示了「档位选择是数据分布问题」——QK 块内集中所以 INT8 按块，V 离散大动态范围所以 FP8 按通道，P 跟随 V。**量化没有全局最优，只有 per-matrix 最优。**
- **fa3（MiniMax M3 稀疏注意力）↔ op3**：同一个「稀疏」思想，fa3 是模型侧改架构（训练时就稀疏），op3 是推理侧改算子（训练不变、即插即用）。两条路线的适用场景完全不同。
- **fw9（Wan2.2 双专家 MoE）→ op3**：fw9 拆了 Wan2.2 的模型结构与 vllm-omni 引擎；op3 展示同一模型在昇腾侧的注意力加速路径。视频生成的系统优化正在成为独立战场。
- **sys 系列（KV Cache / 注意力优化）**：EagleQBSA 只处理**计算与访存中的注意力部分**；KV Cache 压缩、分层缓存是系统层的另一条正交线，与本文互为表里。

## 8. 总结

EagleQBSA 的方法论可以压缩成三句话：

1. **配方看分布**：QK 按块 INT8 + V/P 按通道 FP8，每个矩阵的量化格式由实测数据分布决定，而不是全局一刀切；
2. **瓶颈会转移**：Cube 低精度化之后，Vector 的 Softmax 与量化成为新瓶颈——优化永远要跟着瓶颈走；
3. **硬件亲和是放大器**：Cube→Vector 直搬转置、归约列化、位运算量化转换，这些「非算法」细节贡献了从 1.6× 到 7.6× 之间的差距。

对国内算力栈的从业者，这篇还有一个信号意义：昇腾 950 + MindIE SD 已经把「量化×稀疏联合注意力」做成了开箱即用的算子特性（随 MindIE SD 发布），视频生成场景的国产推理优化从「能跑」进入「抠倍速」阶段。

---

**系列导航**

- 上一篇：**op2 模型量化部署——从 FP16 到 FP4 的精度压缩谱系**（量化档位的地图，本篇是「真实算子里的档位选择」）；
- 同系列起点：**op1 MLA 算子级实现**（算子视角的另一个样本：压 KV 宽度）；
- 相关篇目：**fa3 MiniMax M3 稀疏注意力**（模型侧的稀疏，与本篇推理侧的稀疏对照读）、**fw9 Wan2.2 双专家 MoE 与 vllm-omni**（同一模型的系统侧拆解）、**sys9 KV Cache 全景**（系统层的正交优化线）。

*资料来源：本文整理自昇腾社区 / MindIE SD 公开技术文章《加速 7.6 倍！昇腾 EagleQBSA 量化稀疏注意力优化实践》（CANN 算子源码：gitcode.com/cann/ops-transformer → attention/block_sparse_attention；开源 PR：gitcode.com/Ascend/MindIE-SD/pull/608）。所有性能与精度数据以原文披露的测试配置为准，不代表普适结论。*
