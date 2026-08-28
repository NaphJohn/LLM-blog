---
title: 前沿架构解码手记（六）：Gated DeltaNet——Conv1D 先抓局部、Gated Delta Rule 再压全局的线性注意力层
description: 拆解 Gated DeltaNet（GDN）层的完整架构——算子 1 因果卷积 Conv1D 负责局部上下文与位置感知、算子 2 递归状态更新执行带衰减门控 α 的 Gated Delta Rule，以 O(n·d²) 线性复杂度压缩全局历史。结合原始论文（Yang et al., ICLR 2025）与 Qwen3.5/3.6/3.8 实现交叉验证，澄清四个易错点：型号命名、完整公式、无 RoPE、状态为 d_v×d_k 矩阵。
pubDate: 2026-08-26
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa6-gated-deltanet
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么单聊这一层

fa1–fa4 讲的是**怎么把注意力改轻**（稀疏 / 压缩 / 混合 SSM）。但这一篇要聊的 Gated DeltaNet（GDN）走得更远——它**不是注意力**，而是一类**线性注意力 / 递归状态空间**层，核心靠一个固定大小的「状态矩阵」在每次 token 上做增量更新，把全局历史压进 O(d²) 的定长张量里。

它的层结构非常干净，只有两个算子：

- **算子 1 · Causal Conv1D**：在 Q/K/V 投影之后，对三个分支各做一层深度可分离因果卷积 + SiLU，提供**局部上下文与位置感知**。
- **算子 2 · Recurrent State Update**：接收卷积后的 Q′/K′/V′ 与数据依赖的衰减 / 写入门控 αₜ、βₜ，执行 **Gated Delta Rule**，更新固定大小的 `ssm_state` 矩阵并输出。

两者形成「**先局部卷积、后全局线性注意力**」的层级：Conv1D 解决短程依赖和位置，Recurrent State 以 O(n·d²) 把长历史压缩成定长状态。下面这张图就是它的完整链路。

<figure class="arch-fig">
<svg viewBox="0 0 680 540" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gated DeltaNet 层架构：Causal Conv1D 后接带 Gated Delta Rule 的递归状态更新">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#8B4513"/>
    </marker>
    <marker id="arrowB" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#1d4ed8"/>
    </marker>
    <style>
      .box{fill:#fff7ed;stroke:#c2410c;stroke-width:1.5}
      .box2{fill:#eff6ff;stroke:#1d4ed8;stroke-width:1.5}
      .gate{fill:#fef9c3;stroke:#a16207;stroke-width:1.5}
      .ssm{fill:#ecfdf5;stroke:#047857;stroke-width:1.5}
      .lab{font:600 13px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .sub{font:11px -apple-system,'PingFang SC',sans-serif;fill:#475569}
      .form{font:700 14px -apple-system,'PingFang SC',sans-serif;fill:#7c2d12}
      .conn{font:10px -apple-system,'PingFang SC',sans-serif;fill:#6b7280}
      .cell{font:600 12px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
    </style>
  </defs>

  <!-- 输入 -->
  <rect class="ssm" x="270" y="14" width="140" height="36" rx="8"/>
  <text class="lab" x="340" y="37" text-anchor="middle">xₜ（当前 token）</text>
  <line x1="340" y1="50" x2="340" y2="68" stroke="#8B4513" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- QKV 投影 -->
  <rect class="box" x="200" y="68" width="280" height="44" rx="8"/>
  <text class="lab" x="340" y="91" text-anchor="middle">Q / K / V 投影（Linear）· 无 RoPE</text>
  <text class="sub" x="340" y="118" text-anchor="middle">位置信息交给 Conv1D，不引入 RoPE</text>

  <!-- 三分支到 Conv1D -->
  <path d="M300,112 C250,128 200,135 130,150" fill="none" stroke="#8B4513" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="340" y1="112" x2="340" y2="150" stroke="#8B4513" stroke-width="1.5" marker-end="url(#arrow)"/>
  <path d="M380,112 C430,128 480,135 550,150" fill="none" stroke="#8B4513" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 算子1 Conv1D -->
  <rect class="box" x="30" y="150" width="620" height="118" rx="10"/>
  <text class="lab" x="340" y="174" text-anchor="middle">算子 1 · Causal Conv1D（深度可分离因果卷积 + SiLU）</text>
  <text class="sub" x="340" y="193" text-anchor="middle">维护 conv_state 滑动窗口 → 局部上下文 + 位置感知</text>
  <rect class="gate" x="55" y="206" width="170" height="46" rx="6"/>
  <text class="cell" x="140" y="228" text-anchor="middle">Q′ = Conv1D(Q)</text>
  <text class="sub" x="140" y="244" text-anchor="middle">+ SiLU</text>
  <rect class="gate" x="255" y="206" width="170" height="46" rx="6"/>
  <text class="cell" x="340" y="228" text-anchor="middle">K′ = Conv1D(K)</text>
  <text class="sub" x="340" y="244" text-anchor="middle">+ SiLU</text>
  <rect class="gate" x="455" y="206" width="170" height="46" rx="6"/>
  <text class="cell" x="540" y="228" text-anchor="middle">V′ = Conv1D(V)</text>
  <text class="sub" x="540" y="244" text-anchor="middle">+ SiLU</text>

  <!-- 三分支到 算子2 -->
  <path d="M140,252 C110,272 100,290 90,300" fill="none" stroke="#1d4ed8" stroke-width="1.5" marker-end="url(#arrowB)"/>
  <line x1="340" y1="268" x2="340" y2="300" stroke="#1d4ed8" stroke-width="1.5" marker-end="url(#arrowB)"/>
  <path d="M540,252 C570,272 580,290 590,300" fill="none" stroke="#1d4ed8" stroke-width="1.5" marker-end="url(#arrowB)"/>

  <!-- 算子2 Recurrent State Update -->
  <rect class="box2" x="30" y="300" width="620" height="172" rx="10"/>
  <text class="lab" x="340" y="324" text-anchor="middle">算子 2 · Recurrent State Update（Gated Delta Rule）</text>

  <!-- 门控 -->
  <rect class="gate" x="55" y="340" width="200" height="44" rx="6"/>
  <text class="cell" x="155" y="362" text-anchor="middle">门控 αₜ, βₜ（数据依赖）</text>
  <text class="sub" x="155" y="378" text-anchor="middle">α 遗忘 · β 写入</text>

  <!-- 状态矩阵 -->
  <rect class="ssm" x="455" y="340" width="180" height="44" rx="6"/>
  <text class="cell" x="545" y="358" text-anchor="middle">Sₜ：每 head 一个</text>
  <text class="cell" x="545" y="375" text-anchor="middle">d_v × d_k 矩阵（定长）</text>

  <!-- 公式 -->
  <text class="form" x="340" y="420" text-anchor="middle">Sₜ = Sₜ₋₁ · αₜ ( I − βₜ kₜ kₜᵀ ) + βₜ vₜ kₜᵀ</text>
  <text class="sub" x="340" y="440" text-anchor="middle">δ-rule 写入 − 衰减门控 αₜ 遗忘历史 · 复杂度 O(n·d²)</text>

  <!-- 递归环路：Sₜ → 下一时刻 Sₜ₋₁ -->
  <path d="M635,362 C660,400 660,460 545,470 L95,470 C30,470 30,410 95,388" fill="none" stroke="#047857" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#arrowB)"/>
  <text class="conn" x="345" y="490" text-anchor="middle">↑ 递归传递：Sₜ 作为下一时刻 Sₜ₋₁，定长状态跨 token 滚动</text>

  <!-- 输出 -->
  <line x1="340" y1="472" x2="340" y2="492" stroke="#8B4513" stroke-width="2" marker-end="url(#arrow)"/>
  <rect class="ssm" x="270" y="492" width="140" height="34" rx="8"/>
  <text class="lab" x="340" y="514" text-anchor="middle">oₜ（层输出）</text>
</svg>
<figcaption>图：Gated DeltaNet 层两级算子。算子 1 在 Q/K/V 投影后各做一层深度可分离因果卷积 + SiLU（维护 conv_state 滑动窗口）；算子 2 接收卷积后的 Q′/K′/V′ 与数据依赖门控 αₜ、βₜ，执行 Gated Delta Rule 更新定长状态矩阵 Sₜ 并输出。Conv1D 管局部 + 位置，Recurrent State 管全局线性注意力。</figcaption>
</figure>

## 1. 算子 1：Causal Conv1D（局部 + 位置）

位于 Q/K/V 投影**之后**，对 Q、K、V 三个分支**分别**做：

- **深度可分离因果卷积（Depthwise Causal Conv1D）**：在每个通道上独立做一维因果卷积，卷积核只在当前及之前的位置滑动，保证因果性（看不到未来）。
- **SiLU 激活**：卷积后接 SiLU。
- **conv_state 滑动窗口**：训练 / 推理时维护一个定长滑动窗口状态，使得递归步进只需缓存最近 `kernel_size` 个位置，不随序列增长。

它的作用有两层：

1. **短程依赖建模**：卷积天然聚合邻近 token，弥补线性注意力「只看全局状态、对局部不敏感」的短板。
2. **位置感知（替代 RoPE）**：GDN 层**完全不使用 RoPE**，位置信息就靠这层因果卷积的局部感受野来提供——所以文档里「Causal Conv1D 类似位置编码或短程依赖建模」的判断是准确的。

## 2. 算子 2：Recurrent State Update（Gated Delta Rule）

这是 GDN 的核心。它接收卷积后的 Q′/K′/V′，以及两个**数据依赖**的门控信号：

- **αₜ（衰减门控，∈ (0,1)）**：控制对历史状态的指数遗忘强度。
- **βₜ（写入门控）**：控制当前 token 写入状态的强度。

**完整公式**（含衰减门控 αₜ）：

```
Sₜ = Sₜ₋₁ · αₜ ( I − βₜ kₜ kₜᵀ )  +  βₜ vₜ kₜᵀ
```

逐项读：

- `Sₜ₋₁ · αₜ`：先把上一时刻状态整体乘衰减门控，**历史被按 αₜ 比例遗忘**（这是朴素 Delta Rule 缺的那一项）。
- `αₜ ( I − βₜ kₜ kₜᵀ )`：在遗忘后的状态上，再做一次「沿当前 key 方向的 Delta 校正」——`βₜ kₜ kₜᵀ` 是从旧状态里减去「与当前 key 相关的旧预测」的投影，腾出空间。
- `+ βₜ vₜ kₜᵀ`：把当前 value 按写入门控 βₜ 写进状态，作为「key → value」的外积。

> 注意：文档里写的 `Sₜ = Sₜ₋₁ − βₜ(Sₜ₋₁φ(kₜ))φ(kₜ)ᵀ + βₜ vₜ φ(kₜ)ᵀ` 是**不包含 αₜ 衰减门控**的朴素 Delta Rule 形式，与原始论文一致；但 GDN 的**完整**公式必须包含 αₜ 衰减项，否则就退化成了没有遗忘机制的线性注意力。

## 3. 位置编码：无 RoPE

GDN 层**不使用旋转位置编码（RoPE）**。整层的绝对 / 相对位置线索都来自：

- 算子 1 的**因果卷积感受野**（局部顺序）；
- 递归状态 `Sₜ` 本身按时间顺序滚动（隐式时序）。

这正是 GDN 能保持线性复杂度、又不过度依赖位置编码注入的原因。

## 4. 状态维度：是矩阵，不是向量

`ssm_state` 的形状是**每个注意力头一个 `d_v × d_k` 矩阵**（而非单向量）：

- `d_k`：key 维度（外积 `kₜ kₜᵀ` 的一边）；
- `d_v`：value 维度（外积 `vₜ kₜᵀ` 的另一边）；
- 矩阵规模与序列长度**无关**，因此状态大小恒定——这是 O(n·d²) 线性复杂度的根基。

文档里「状态矩阵（或向量）」的表述略模糊，应明确为**矩阵**：每个头持有一张 `d_v × d_k` 的定长表，递归地在上面做「遗忘 + 校正 + 写入」。

## 5. 复杂度：为什么是线性

- 朴素注意力：每 token 要与全部历史做点积，复杂度 **O(n²·d)**。
- GDN 递归状态：每个 token 只做「读状态 + 更新定长矩阵 + 写状态」，状态大小恒为 `d_v·d_k`，复杂度 **O(n·d²)**。

当序列很长（n ≫ d）时，`O(n·d²)` 显著低于 `O(n²·d)`，这正是 GDN 能撑长上下文、且 KV Cache 不随序列爆炸的根本原因——它的「记忆」是一张定长矩阵，而不是逐 token 的 KV 列表。

## 6. 与 Qwen3.5 / 3.6 / 3.8 的关系（型号勘误）

文档里出现的 **`qwen3.8-27B` 这一确切型号如今确认真实存在**——早期本文判断其「不存在」有误，已据 2026-08-28 仓库快照扫描更正（交叉验证见（七）Qwen3.8 双 checkpoint 对比）。Qwen3.8 系列主档是 **2.4T-A95B 这类 MoE 模型**，同时也存在 **27B 稠密** 这一档 checkpoint。

但有一点是确凿的：**Gated DeltaNet 架构（GDN 层）确实是 Qwen3.5 / 3.6 / 3.8 系列的核心组件**，是这一系列能在长上下文下保持线性推理成本的关键模块之一。所以——

- ⚠️ 更正：「Qwen3.8-27B 确实存在」（27B 稠密 checkpoint，2026-08-28 仓库扫描确认；此前「不存在」判断已撤销）
- ✅ 正确：「GDN 是 Qwen3.5/3.6/3.8 系列（含 2.4T-A95B MoE 与 27B 稠密）的核心架构组件」

## 7. 文档交叉验证后的四处修正小结

结合原始论文（Yang et al., *Gated DeltaNet*, ICLR 2025）与 Qwen3.5/3.6/3.8 系列实现代码交叉验证，整体描述基本正确，四处需修正：

| # | 修正点 | 文档原表述 | 正确表述 |
|---|---|---|---|
| 1 | 型号名称 | `qwen3.8-27B` | 27B 稠密 checkpoint 真实存在（2026-08-28 仓库扫描确认，此前「不存在」判断已撤销）；GDN 是 Qwen3.5/3.6/3.8 核心组件 ✅ |
| 2 | 完整公式 | 仅含 β 的 Delta Rule | 须加衰减门控 α：`Sₜ = Sₜ₋₁·αₜ(I − βₜ kₜkₜᵀ) + βₜ vₜkₜᵀ` |
| 3 | 位置编码 | 提了一句 Conv1D「类似位置编码」 | 准确——GDN **完全无 RoPE**，位置靠因果卷积 + 状态滚动 |
| 4 | 状态维度 | 「状态矩阵（或向量）」略模糊 | 明确为**每 head 一个 `d_v × d_k` 矩阵**（定长，与序列长无关） |

<div class="warnbox">
<strong>一句话记忆：</strong>Gated DeltaNet = Conv1D 抓局部/位置 + Gated Delta Rule 压全局；状态是每头一张定长 d_v×d_k 矩阵，靠 α 遗忘、β 写入，全程无 RoPE、复杂度 O(n·d²)。它是 Qwen3.5/3.6/3.8 系列线性推理成本的核心支柱之一。
</div>

---

*下一篇预告：GDN 与 Mamba / 线性注意力家族（GLA、RWKV、RetNet）的横向对比——它们都在「用定长状态替代 KV 列表」，但状态更新规则完全不同。*
