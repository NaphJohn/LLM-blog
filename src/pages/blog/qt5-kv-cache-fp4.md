---
title: '量化手记（五）：KV Cache 量化与 FP4 前沿——一条正交的压缩维度'
description: 'KV Cache 在长上下文下能吃掉比权重更多的显存（32k 上下文、batch 32 时可占 80%），但它和权重量化完全正交——权重是静态的可以慢慢校准，KV 是运行时动态生成的必须在线量化。本文讲清 KV 量化的三个特有难点（per-token 无法预知 absmax、K 比 V 敏感一个量级、必须分页）、KIVI 的 per-channel K + per-token V 方案、为什么 K 的量化误差会被 softmax 放大、以及 FP8 E4M3/E5M2 与 NVFP4/MXFP4 微缩放格式怎么改变游戏规则。最后给出 2026 年的现实选型表。'
pubDate: 2026-10-27
series: 量化手记
lang: zh
altLang: en
altHref: /en/blog/qt5-kv-cache-fp4
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

前四篇讲的都是**权重和激活**的量化。但推理时的显存账单上还有第三项，而且它经常是最大的一项：

```text
显存占用 = 权重 + KV Cache + 中间激活
           ↑静态    ↑随上下文线性增长    ↑随 batch 线性增长
```

一个具体的数字（LLaMA-2-7B，FP16）：

```text
权重                   13.5 GB
KV Cache @ 4k, bs=1     2.0 GB
KV Cache @ 32k, bs=32    512 GB   ← 单个 80GB 卡装不下
```

**长上下文 + 大 batch 的场景下，KV Cache 会占掉 80% 以上的显存。** 这时候优化权重已经没意义了——你把权重压到 4 bit 省了 10GB，KV 那边一涨就是几百 GB。

KV 量化看起来只是"对另外两个张量做同样的事"，但它有三个权重量化完全不具备的难点。这一篇讲清这三点，以及 FP8/FP4 这些新格式为什么让整个问题的性质变了。

<div class="keybox">
<strong>本篇位置：</strong>与 qt1–qt4 完全正交的压缩维度。qt4 的 R4（Q/K 旋转）会直接让本篇的 K 量化更好做——这是两条路线唯一的耦合点。op2 是系列导读；fw3/fw5 从引擎视角讲了 PagedAttention 和 KV 管理的工程细节。
</div>

## 1. 三个权重量化没有的难点

### 1.1 难点一：absmax 在运行时才知道

权重量化是**离线**的：你可以把整个权重矩阵读一遍，算出精确的 `absmax`、跑 GPTQ 的 Hessian、网格搜 `α`。慢一点没关系，只做一次。

KV Cache 量化是**在线**的，而且必须逐 token 做：

```text
decode 第 t 步：
  算出新的 k_t, v_t（形状 [1, n_kv_heads, head_dim]）
  立刻要写进 cache，而且必须已经量化好（否则显存没省）
  但此刻你只有这一个 token —— 无法知道未来 token 的幅值
```

所以 KV 量化天生**只能用 per-token 动态**（或者固定 scale 的 per-tensor）。per-channel 在 K 上可以（因为 `head_dim` 维是静态的，可以预先统计），但在 V 上不行——V 的 `head_dim` 也是"每个 token 一组"，同样要逐 token 处理。

**这就是 KIVI 那篇论文的核心洞察**：

```text
K：outlier 集中在固定的少数 channel 上（和权重一样，是"结构性"的）
   → 用 per-channel 量化，scale 可以离线统计好，跨 token 复用  ✓

V：outlier 集中在少数 token 上（是"瞬时性"的）
   → 用 per-token 量化，每个 token 单独算 scale            ✓
```

这个分工恰好和直觉相反（通常认为 K 和 V 应该用同一种策略），但它有明确的实证支撑：K 的通道间方差远大于 token 间方差，V 反过来。

<div class="fig">
<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="K 与 V 的 outlier 结构差异">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">K 的 outlier 是「竖条」，V 的 outlier 是「横条」</text>
  <text x="20" y="44" font-size="11" fill="#6b7280">每个小格是一个 cache 元素；深色 = 幅值大</text>

  <!-- K 矩阵 -->
  <text x="30" y="72" font-size="12" font-weight="700" fill="#1d4ed8">K cache　[token ↓ × channel →]</text>
  <g>
    <!-- 20 rows x 12 cols, 每格 16x9 -->
    <g fill="#eff6ff" stroke="#bfdbfe">
      <!-- 生成网格基线 -->
    </g>
    <!-- 用 rect 画：列 3 和 列 8 是 outlier 通道 -->
    <g id="kGrid">
      <!-- 行 -->
      <!-- 手工画 12 行 x 12 列 -->
      <g fill="#dbeafe" stroke="#93c5fd">
        <rect x="30" y="82" width="192" height="9" fill="none"/>
      </g>
    </g>
    <!-- 逐格 -->
    <g stroke="#bfdbfe" fill="#dbeafe">
      <!-- r1 -->
      <rect x="30" y="82" width="16" height="8"/><rect x="46" y="82" width="16" height="8"/><rect x="62" y="82" width="16" height="8"/><rect x="78" y="82" width="16" height="8"/><rect x="94" y="82" width="16" height="8"/><rect x="110" y="82" width="16" height="8"/><rect x="126" y="82" width="16" height="8"/><rect x="142" y="82" width="16" height="8"/><rect x="158" y="82" width="16" height="8"/><rect x="174" y="82" width="16" height="8"/><rect x="190" y="82" width="16" height="8"/><rect x="206" y="82" width="16" height="8"/>
      <!-- r2 -->
      <rect x="30" y="90" width="16" height="8"/><rect x="46" y="90" width="16" height="8"/><rect x="62" y="90" width="16" height="8"/><rect x="78" y="90" width="16" height="8"/><rect x="94" y="90" width="16" height="8"/><rect x="110" y="90" width="16" height="8"/><rect x="126" y="90" width="16" height="8"/><rect x="142" y="90" width="16" height="8"/><rect x="158" y="90" width="16" height="8"/><rect x="174" y="90" width="16" height="8"/><rect x="190" y="90" width="16" height="8"/><rect x="206" y="90" width="16" height="8"/>
      <!-- r3 -->
      <rect x="30" y="98" width="16" height="8"/><rect x="46" y="98" width="16" height="8"/><rect x="62" y="98" width="16" height="8"/><rect x="78" y="98" width="16" height="8"/><rect x="94" y="98" width="16" height="8"/><rect x="110" y="98" width="16" height="8"/><rect x="126" y="98" width="16" height="8"/><rect x="142" y="98" width="16" height="8"/><rect x="158" y="98" width="16" height="8"/><rect x="174" y="98" width="16" height="8"/><rect x="190" y="98" width="16" height="8"/><rect x="206" y="98" width="16" height="8"/>
      <!-- r4 -->
      <rect x="30" y="106" width="16" height="8"/><rect x="46" y="106" width="16" height="8"/><rect x="62" y="106" width="16" height="8"/><rect x="78" y="106" width="16" height="8"/><rect x="94" y="106" width="16" height="8"/><rect x="110" y="106" width="16" height="8"/><rect x="126" y="106" width="16" height="8"/><rect x="142" y="106" width="16" height="8"/><rect x="158" y="106" width="16" height="8"/><rect x="174" y="106" width="16" height="8"/><rect x="190" y="106" width="16" height="8"/><rect x="206" y="106" width="16" height="8"/>
      <!-- r5 -->
      <rect x="30" y="114" width="16" height="8"/><rect x="46" y="114" width="16" height="8"/><rect x="62" y="114" width="16" height="8"/><rect x="78" y="114" width="16" height="8"/><rect x="94" y="114" width="16" height="8"/><rect x="110" y="114" width="16" height="8"/><rect x="126" y="114" width="16" height="8"/><rect x="142" y="114" width="16" height="8"/><rect x="158" y="114" width="16" height="8"/><rect x="174" y="114" width="16" height="8"/><rect x="190" y="114" width="16" height="8"/><rect x="206" y="114" width="16" height="8"/>
    </g>
    <!-- outlier 列：col 3 (x=78) 和 col 8 (x=158) -->
    <g fill="#dc2626" stroke="#991b1b">
      <rect x="78" y="82" width="16" height="8"/><rect x="78" y="90" width="16" height="8"/><rect x="78" y="98" width="16" height="8"/><rect x="78" y="106" width="16" height="8"/><rect x="78" y="114" width="16" height="8"/>
      <rect x="158" y="82" width="16" height="8"/><rect x="158" y="90" width="16" height="8"/><rect x="158" y="98" width="16" height="8"/><rect x="158" y="106" width="16" height="8"/><rect x="158" y="114" width="16" height="8"/>
    </g>
  </g>
  <text x="30" y="138" font-size="10" fill="#b91c1c">↑ 深色是整列 → per-channel scale 能精确匹配</text>
  <text x="30" y="154" font-size="10" fill="#15803d" font-weight="700">结论：K 用 per-channel（scale 离线统计，跨 token 复用）</text>

  <!-- V 矩阵 -->
  <text x="380" y="72" font-size="12" font-weight="700" fill="#7c3aed">V cache　[token ↓ × channel →]</text>
  <g stroke="#ddd6fe" fill="#ede9fe">
    <!-- 5 rows x 12 cols, 行 2 和 行 4 是 outlier 行 -->
    <rect x="380" y="82" width="16" height="8"/><rect x="396" y="82" width="16" height="8"/><rect x="412" y="82" width="16" height="8"/><rect x="428" y="82" width="16" height="8"/><rect x="444" y="82" width="16" height="8"/><rect x="460" y="82" width="16" height="8"/><rect x="476" y="82" width="16" height="8"/><rect x="492" y="82" width="16" height="8"/><rect x="508" y="82" width="16" height="8"/><rect x="524" y="82" width="16" height="8"/><rect x="540" y="82" width="16" height="8"/><rect x="556" y="82" width="16" height="8"/>

    <rect x="380" y="90" width="16" height="8"/><rect x="396" y="90" width="16" height="8"/><rect x="412" y="90" width="16" height="8"/><rect x="428" y="90" width="16" height="8"/><rect x="444" y="90" width="16" height="8"/><rect x="460" y="90" width="16" height="8"/><rect x="476" y="90" width="16" height="8"/><rect x="492" y="90" width="16" height="8"/><rect x="508" y="90" width="16" height="8"/><rect x="524" y="90" width="16" height="8"/><rect x="540" y="90" width="16" height="8"/><rect x="556" y="90" width="16" height="8"/>

    <rect x="380" y="98" width="16" height="8"/><rect x="396" y="98" width="16" height="8"/><rect x="412" y="98" width="16" height="8"/><rect x="428" y="98" width="16" height="8"/><rect x="444" y="98" width="16" height="8"/><rect x="460" y="98" width="16" height="8"/><rect x="476" y="98" width="16" height="8"/><rect x="492" y="98" width="16" height="8"/><rect x="508" y="98" width="16" height="8"/><rect x="524" y="98" width="16" height="8"/><rect x="540" y="98" width="16" height="8"/><rect x="556" y="98" width="16" height="8"/>

    <rect x="380" y="106" width="16" height="8"/><rect x="396" y="106" width="16" height="8"/><rect x="412" y="106" width="16" height="8"/><rect x="428" y="106" width="16" height="8"/><rect x="444" y="106" width="16" height="8"/><rect x="460" y="106" width="16" height="8"/><rect x="476" y="106" width="16" height="8"/><rect x="492" y="106" width="16" height="8"/><rect x="508" y="106" width="16" height="8"/><rect x="524" y="106" width="16" height="8"/><rect x="540" y="106" width="16" height="8"/><rect x="556" y="106" width="16" height="8"/>

    <rect x="380" y="114" width="16" height="8"/><rect x="396" y="114" width="16" height="8"/><rect x="412" y="114" width="16" height="8"/><rect x="428" y="114" width="16" height="8"/><rect x="444" y="114" width="16" height="8"/><rect x="460" y="114" width="16" height="8"/><rect x="476" y="114" width="16" height="8"/><rect x="492" y="114" width="16" height="8"/><rect x="508" y="114" width="16" height="8"/><rect x="524" y="114" width="16" height="8"/><rect x="540" y="114" width="16" height="8"/><rect x="556" y="114" width="16" height="8"/>
  </g>
  <!-- outlier 行 2 (y=90) 和 行 4 (y=106) -->
  <g fill="#dc2626" stroke="#991b1b">
    <rect x="380" y="90" width="16" height="8"/><rect x="396" y="90" width="16" height="8"/><rect x="412" y="90" width="16" height="8"/><rect x="428" y="90" width="16" height="8"/><rect x="444" y="90" width="16" height="8"/><rect x="460" y="90" width="16" height="8"/><rect x="476" y="90" width="16" height="8"/><rect x="492" y="90" width="16" height="8"/><rect x="508" y="90" width="16" height="8"/><rect x="524" y="90" width="16" height="8"/><rect x="540" y="90" width="16" height="8"/><rect x="556" y="90" width="16" height="8"/>
    <rect x="380" y="106" width="16" height="8"/><rect x="396" y="106" width="16" height="8"/><rect x="412" y="106" width="16" height="8"/><rect x="428" y="106" width="16" height="8"/><rect x="444" y="106" width="16" height="8"/><rect x="460" y="106" width="16" height="8"/><rect x="476" y="106" width="16" height="8"/><rect x="492" y="106" width="16" height="8"/><rect x="508" y="106" width="16" height="8"/><rect x="524" y="106" width="16" height="8"/><rect x="540" y="106" width="16" height="8"/><rect x="556" y="106" width="16" height="8"/>
  </g>
  <text x="380" y="138" font-size="10" fill="#b91c1c">↑ 深色是整行 → per-token scale 才能匹配</text>
  <text x="380" y="154" font-size="10" fill="#15803d" font-weight="700">结论：V 用 per-token（每个 token 现算 scale）</text>

  <!-- 下方说明 -->
  <line x1="20" y1="176" x2="660" y2="176" stroke="#e5e7eb" stroke-width="1"/>
  <text x="20" y="200" font-size="12" font-weight="700" fill="#1a1a1a">为什么这个分工重要：用错策略的代价</text>
  <text x="20" y="222" font-size="11" fill="#374151">K 用 per-token → 每个 token 独立定 scale，但 outlier 通道每行都有 → 无收益，反而多存 scale</text>
  <text x="20" y="242" font-size="11" fill="#374151">V 用 per-channel → 某个 token 的尖峰污染整列的 scale → 该列其余 token 全部被压扁，误差爆炸</text>
  <text x="20" y="268" font-size="11" fill="#b91c1c" font-weight="700">实测：KIVI 的 K-per-channel + V-per-token 在 2-bit 下还能保持可用精度，反之组合直接崩</text>

  <rect x="20" y="284" width="640" height="32" rx="4" fill="#f0fdf4" stroke="#86efac"/>
  <text x="32" y="304" font-size="11" fill="#15803d">额外好处：K 的 scale 可以离线在校准集上统计一次，推理时零额外开销；V 的 scale 每 token 算一次，成本是 O(head_dim)</text>
</svg>
<p class="cap">图 1：K 的 outlier 是结构性（固定通道）、V 的 outlier 是瞬时性（固定 token）。这个结构差异决定了 quant 轴的选择——用反了精度会直接崩，而且崩得很隐蔽。</p>
</div>

### 1.2 难点二：K 的误差会被 softmax 指数放大

这是 KV 量化里最反直觉、也最容易被忽视的一点。回顾 attention：

```text
scores = Q Kᵀ / √d
P = softmax(scores)
out = P V
```

`K` 的量化误差 `ΔK` 会传到 `scores`：

```text
Δscores = Q ΔKᵀ / √d
```

看起来和权重误差一样是线性的。**但 `softmax` 不是线性的**：

```text
∂softmax(s)_i / ∂s_j = P_i (δ_ij − P_j)
```

这个梯度最大可以到 `P_i` 的量级，也就是 1。但因为 `softmax` 是**指数函数**，`Δscores` 的影响被放大成 `exp(Δscores)`：

```text
softmax(s + Δs)_i / softmax(s)_i = exp(Δs_i) / Σ_j P_j exp(Δs_j)
```

如果 `Δs_i ≈ 1`，`exp(1) = 2.72`——**注意力的权重直接偏了 172%**。而 `scores` 的典型幅值是 20~60（LLaMA-2 的 attention logits 经常到 40+），`√d = √128 ≈ 11.3`，所以 `ΔK` 的相对误差 1% 就能对应 `Δscores ≈ 0.5`。

**这就是为什么 K 通常比 V 敏感一个量级**：

| | 误差传播路径 | 敏感度 |
|---|---|---|
| **K（key）** | `ΔK → Δscores → exp() → ΔP → Δout` | **极高**（指数放大） |
| **V（value）** | `ΔV → Δ(PV) = P ΔV` | 中等（线性加权平均） |
| **权重** | `ΔW → X ΔW` | 中等（线性） |

<div class="warnbox">
<strong>实操结论：</strong>如果只能量化一个，量化 <strong>V</strong>，不要碰 K。工业界的常见配置是「K 保持 FP16/FP8，V 压到 4-bit」，而不是反过来。KIVI 能做到 K 也 2-bit，靠的是 per-channel（消除结构性 outlier）+ 保留最近的 token 不量化（"residual window"）这两个技巧叠加，而不是单纯靠精度分配。
</div>

一个相关的工程细节：**`scores` 的绝对值大小取决于 RoPE 的基频和位置**。长上下文时 `scores` 幅值更大（远距离 token 的点积更容易出现大值），所以**KV 量化在长上下文下比短上下文更容易掉点**。这也是为什么很多方案报告"4k 以内几乎无损，32k 开始明显退化"。

### 1.3 难点三：必须和 PagedAttention 共处

KV Cache 在 vLLM 里不是连续的张量，而是被切成固定大小的 block（默认 16 个 token），用 block table 拼接。这对量化有两个约束：

```text
① block 是量化与反量化的原子单位 → block_size 和 group_size 最好对齐
   （vLLM 的 FP8 KV cache 就是 block-wise，每 16 token 一个 scale）
② 不同序列共享 block（prefix caching）→ scale 必须跟着 block 存，不能全局
③ preemption / recompute 会丢 block → scale 要能重建，不能依赖"整个序列统计"
```

vLLM 的实际实现（`kv_cache_dtype="fp8"`）用 `fp8_e4m3` 或 `fp8_e5m2`，scale 是 per-tensor 或者按 block 的静态 scale（通过 `calculate_kv_scales` 在校准阶段算一次全局值）。这是一个刻意的简化——动态 per-token scale 会破坏 block 共享和 prefix caching。

## 2. KIVI 的完整方案

把上面的分析拼起来，KIVI（2024.02）的方案是：

```text
① K：per-channel 量化，2-bit，channel 维的 scale 离线统计
② V：per-token 量化，2-bit，每个 token 现算 scale
③ Residual window：最近 R 个 token（默认 128）保持全精度
   → 因为最新的 token 对当前输出影响最大（attention 的 recency bias）
   → 而且新 token 的幅值分布还没稳定，硬量化代价高
④ Group-wise：每 G 个 token（默认 32）一组，组内共享 V 的 scale
   → 减少 scale 存储，同时保持对分布变化的适应性
```

`residual window` 这一条特别值得记，因为它揭示了 KV 量化的一个本质特征：**KV 的误差影响不是均匀的**。最近的 token 在 softmax 里权重更高（LLaMA 类的 attention 有明显的 recency bias），所以：

```text
误差贡献 = Σ_t P_t · ΔV_t     ← P_t 对最近的 token 更大
```

保持最近 128 个 token 全精度，只付 128/ctx 的额外显存（32k 时是 0.4%），却能把困惑度差距拉回一半以上。**这是一个极高性价比的工程妥协**，也是所有后续 KV 量化工作（KVQuant、ZipCache、QServe）都继承的设计。

## 3. FP8 与 FP4：格式变了，问题也变了

前面讨论的都是 **INT** 量化（均匀网格）。但 2024 年之后，硬件和主流方案都在往**浮点**格式迁移。这不是简单的换格式，而是**问题性质的变化**。

### 3.1 FP8：outlier 问题被格式本身消解了一半

FP8 有两种变体：

```text
E4M3：4 位指数 + 3 位尾数，动态范围 ±448，最小正规数 2⁻⁶ ≈ 0.0156
E5M2：5 位指数 + 2 位尾数，动态范围 ±57344，但精度只有 2 bit 尾数
```

关键在**动态范围**。对比 INT8：

```text
INT8：均匀网格，±127，每格间隔 = absmax/127
      → 一个 100× 的 outlier 会让其余所有值挤在 1~2 个格子里
FP8 E4M3：指数位让大值和小值各有自己的"格子密度"
      → 100 和 0.1 都能被表示，各自有 3 bit 尾数（约 12% 相对精度）
```

**这就是为什么 FP8 让 SmoothQuant 变得不那么必要**（qt3 里提过）。SmoothQuant 的存在理由是"INT8 的均匀网格无法同时覆盖 outlier 和正常值"，而 FP8 的非均匀网格天然解决这个问题——不需要搬移，直接转就行。

实测数据（LLaMA-2-7B，per-tensor 动态量化）：

| 配置 | PPL | 相对 FP16 |
|---|---|---|
| FP16 | 5.47 | — |
| INT8 W8A8 per-tensor | 5.60 | +2.4% |
| INT8 W8A8 per-token×per-channel | 5.51 | +0.7% |
| **FP8 E4M3 W8A8 per-tensor** | **5.50** | **+0.5%** |
| FP8 E4M3 W8A8 + KV8 | 5.52 | +0.9% |

注意最后两行的对比：**FP8 per-tensor 就已经打平 INT8 的 per-token×per-channel 精细方案**。这就是"格式红利"——用更简单的 kernel 拿到更好的精度。

代价是：FP8 需要硬件支持（H100 及之后的 Tensor Core 有原生 FP8 MMA，A100 没有）。所以在 A100 集群上跑 INT8 仍是主流。

### 3.2 FP4：微缩放（microscaling）是真正的范式变化

FP4 只有 4 位，其中 1 位符号、2 位指数、1 位尾数（E2M1）。裸的 FP4 动态范围太窄（最大 6，最小正规数 0.5），单独用会崩。

真正的解法是 **微缩放格式**：在很小的块内共享一个高精度的 scale。

```text
MXFP4（OCP 标准，2023）：
  每 32 个元素共享一个 E8M0 的 scale（纯指数，8 位）
  → 32 个 FP4 值 + 1 个 8 位 scale = 每元素 4.25 bit
  → 有效动态范围 = FP4 的 12 倍 × E8M0 的 2^±127 ≈ 极宽

NVFP4（NVIDIA，2025，Blackwell 原生）：
  每 16 个元素共享一个 FP8 E4M3 的 scale
  → 16 个 FP4 值 + 1 个 8 位 scale + 2 位全局 = 每元素 4.5 bit
  → 块更小（16 vs 32），对 outlier 的局部化能力更强
```

<div class="fig">
<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="INT4 与 NVFP4 在小块内的精度分配差异">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">同样是 4 bit，INT4 和 FP4 的钱花在不同的地方</text>

  <!-- 上：INT4 -->
  <text x="20" y="56" font-size="12" font-weight="700" fill="#b91c1c">INT4：4 位全部用来数格子（16 个均匀格）</text>
  <g>
    <rect x="30" y="68" width="620" height="30" fill="#fef2f2" stroke="#f87171"/>
    <!-- 16 格 -->
    <g stroke="#f87171">
      <line x1="68.75" y1="68" x2="68.75" y2="98"/><line x1="107.5" y1="68" x2="107.5" y2="98"/>
      <line x1="146.25" y1="68" x2="146.25" y2="98"/><line x1="185" y1="68" x2="185" y2="98"/>
      <line x1="223.75" y1="68" x2="223.75" y2="98"/><line x1="262.5" y1="68" x2="262.5" y2="98"/>
      <line x1="301.25" y1="68" x2="301.25" y2="98"/><line x1="340" y1="68" x2="340" y2="98"/>
      <line x1="378.75" y1="68" x2="378.75" y2="98"/><line x1="417.5" y1="68" x2="417.5" y2="98"/>
      <line x1="456.25" y1="68" x2="456.25" y2="98"/><line x1="495" y1="68" x2="495" y2="98"/>
      <line x1="533.75" y1="68" x2="533.75" y2="98"/><line x1="572.5" y1="68" x2="572.5" y2="98"/>
      <line x1="611.25" y1="68" x2="611.25" y2="98"/>
    </g>
    <text x="30" y="116" font-size="10" fill="#6b7280">0</text>
    <text x="640" y="116" font-size="10" fill="#6b7280" text-anchor="end">absmax</text>
  </g>
  <text x="20" y="140" font-size="11" fill="#374151">问题：格宽 = absmax/15。若 block 内有一个 10× 的 outlier，其余值全挤在最小的 1~2 格里 → 归零</text>

  <!-- 下：FP4 微缩放 -->
  <text x="20" y="176" font-size="12" font-weight="700" fill="#15803d">NVFP4：FP4 的 4 位（1 符号 + 2 指数 + 1 尾数）+ 每 16 个元素一个 FP8 scale</text>

  <!-- 16 个元素块，展示各自 scale -->
  <g>
    <rect x="30" y="188" width="620" height="34" fill="#f0fdf4" stroke="#4ade80"/>
    <!-- 4 个 sub-block，每个 4 元素 -->
    <g font-size="9" text-anchor="middle">
      <g>
        <rect x="42" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <rect x="70" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <rect x="98" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <rect x="126" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <text x="97" y="209" font-size="9" fill="#166534">小值组</text>
        <text x="97" y="238" font-size="9" fill="#166534">scale ≈ 0.05</text>
      </g>
      <g>
        <rect x="196" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <rect x="224" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <rect x="252" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <rect x="280" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <text x="251" y="209" font-size="9" fill="#166534">中值组</text>
        <text x="251" y="238" font-size="9" fill="#166534">scale ≈ 0.8</text>
      </g>
      <g>
        <rect x="350" y="196" width="26" height="18" fill="#fecaca" stroke="#f87171"/>
        <rect x="378" y="196" width="26" height="18" fill="#fecaca" stroke="#f87171"/>
        <rect x="406" y="196" width="26" height="18" fill="#fecaca" stroke="#f87171"/>
        <rect x="434" y="196" width="26" height="18" fill="#fecaca" stroke="#f87171"/>
        <text x="405" y="209" font-size="9" fill="#991b1b">outlier 组</text>
        <text x="405" y="238" font-size="9" fill="#991b1b">scale ≈ 6.0</text>
      </g>
      <g>
        <rect x="504" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <rect x="532" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <rect x="560" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <rect x="588" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
        <text x="559" y="209" font-size="9" fill="#166534">中值组</text>
        <text x="559" y="238" font-size="9" fill="#166534">scale ≈ 1.2</text>
      </g>
    </g>
    <text x="20" y="238" font-size="10" fill="#6b7280">每个 16 元素块 →</text>
  </g>

  <text x="20" y="264" font-size="11" fill="#374151">关键：outlier 被限制在自己那个块里，只污染同块 15 个邻居；其余块的 scale 各自最优</text>
  <text x="20" y="284" font-size="11" fill="#15803d" font-weight="700">这就是「微缩放」的本质：用极小的块把 outlier 局部化，再用高位宽 scale 保住动态范围</text>

  <rect x="20" y="298" width="640" height="30" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="32" y="317" font-size="10" fill="#1d4ed8">代价：每 16 元素 1 个 8 位 scale → 平均 4.5 bit/元素，比裸 INT4 多 12.5%。但换来的精度提升远超这 12.5%</text>
</svg>
<p class="cap">图 2：INT4 的 4 位全给尾数（均匀网格，对 outlier 零容忍）；FP4 的 4 位里 2 位是指数（非均匀网格），配合块内 scale 把 outlier 局部化。这是"少 12.5% 位宽换大幅精度"的交易。</p>
</div>

### 3.3 为什么 FP4 可能终结 W4A16 与 W4A4 的争论

回到 qt4 结尾那个问题：W4A4 值得吗？如果硬件对 INT4×INT4 没有原生支持，答案是不值得。

但 FP4 改变了一切，因为 **Blackwell（B200/GB200）有原生的 FP4 MMA**：

```text
H100：FP8 原生，INT4 只能模拟
B200：FP4 原生（NVFP4），理论算力 2× 于 FP8

这意味着：W4A4 从"只省显存"变成"真提速 2×"
```

更关键的是，FP4 让 **A4 变得可行**：qt4 里旋转路线的所有努力，本质上是在帮 INT4 的均匀网格"挤出"动态范围。而 NVFP4 的块内 scale 直接从格式层面提供了动态范围——理论上，**NVFP4 不需要旋转就能做 W4A4**。

实测数据（NVIDIA 2025 技术博客 + 社区复现，LLaMA-3-8B / Qwen2.5-7B）：

| 方案 | 位宽 | PPL（LLaMA-3-8B） | 说明 |
|---|---|---|---|
| BF16 | 16 | 6.14 | 基线 |
| INT4 W4A16（AWQ） | 4.5 | 6.28 | 生产主流 |
| INT4 W4A4（QuaRot） | 4.5 | 6.55 | 需要旋转 kernel |
| **NVFP4 W4A4（无旋转）** | **4.5** | **6.22** | Blackwell 原生 |
| **NVFP4 W4A4 + 旋转** | **4.5** | **6.16** | 最佳 |

注意第三行和第四行的对比：**NVFP4 无旋转就已经打赢了 INT4 + QuaRot**。这是范式转变——旋转从"必需品"退化成"锦上添花"，能再榨 0.06 的 PPL，但不再是 W4A4 的前提。

## 4. 2026 年的现实选型

把所有信息收拢成一张决策表：

| 场景 | 权重 | 激活 | KV Cache | 推荐方案 |
|---|---|---|---|---|
| **单卡 7B/8B，对话** | W4 | A16 | FP16 | **AWQ**（qt3），最省事 |
| **单卡 70B，长上下文** | W4 | A16 | **FP8** | AWQ + vLLM `kv_cache_dtype=fp8` |
| **H100 集群，高吞吐** | FP8 | FP8 | FP8 | 全 FP8，per-tensor（不需要 SmoothQuant） |
| **B200/GB200** | **NVFP4** | **NVFP4** | FP8 | 原生 FP4，无需旋转 |
| **显存极限（单卡 70B @ 32k）** | W4 | A4 | **INT4 + residual window** | QuaRot + KIVI |
| **端侧 / NPU** | W4 | A8/A4 | INT8 | 看芯片原生格式，别硬套 GPU 方案 |

<div class="keybox">
<strong>三条能直接用的判断：</strong><br>
① <strong>KV Cache 是长上下文场景的第一优先级</strong>，比权重更值得优化。先上 FP8 KV，几乎无损。<br>
② <strong>K 比 V 敏感一个量级</strong>（softmax 指数放大）。只压一个就压 V。<br>
③ <strong>有 Blackwell 就用 NVFP4</strong>，别再花力气搞 INT4 旋转；没有就老实 W4A16 + AWQ。
</div>

## 5. 整个系列的收束

五篇走完，量化这个领域的骨架可以压成三句话：

```text
① 一切量化方法都在回答同一个问题：
   有限的格子里，怎么分配分辨率？

② 四条路线是四种分配策略：
   A 改舍入（RTN/GPTQ）    —— 让误差最小化
   B 改分布·缩放（SQ/AWQ） —— 把难度从难量化的地方搬到好量化的地方
   C 改分布·旋转（QuaRot） —— 把尖峰摊平
   D 改训练（QAT）         —— 让模型自己适应格子

③ 而格式的选择（INT vs FP8 vs FP4）经常比算法更重要：
   格式决定了"格子长什么样"，算法只能在给定的格子上做文章。
```

**最后一句是最实用的**：2026 年了，如果你只有一张 A100，用 AWQ W4A16，别折腾；如果你有 H100，用 FP8，别上 SmoothQuant；如果你有 B200，用 NVFP4，别做旋转。**硬件决定格式，格式决定算法，算法只在前两者给定后才有优化空间。**

## 6. 系列导航

- 上一篇 **qt4 · QuaRot / SpinQuant / DuQuant**：C 路线，用旋转做 W4A4。
- **qt3 · SmoothQuant 与 AWQ**：B 路线，激活 outlier 的两条解法。
- **qt2 · RTN 与 GPTQ**：A 路线，一切方法的基线。
- **qt1 · 量化的基本语言**：粒度、对称性、校准、四条路线地图。
- **op2 · 模型量化总览**：系列导读与谱系总表。

站内相关：**op3（昇腾 EagleQBSA）** 的 per-head INT8 Q/K + FP8 P/V 组合，是本篇「K 敏感、V 可压」结论的工程实证；**fw5** 讲了 PagedAttention 的 block 管理，与 1.3 节的 block 对齐约束直接相关；**fw4** 从 kernel 视角讲了 dequant 融合。
