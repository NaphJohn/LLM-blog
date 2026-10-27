---
title: '量化手记（三）：SmoothQuant 与 AWQ——激活 outlier 的两条相反解法'
description: 'LLM 激活里存在少数幅值大 50~100 倍的 outlier 通道，这是 W8A8 和 W4A16 掉点的真正元凶。SmoothQuant 把 outlier 从激活「搬」到权重（数学上等价，per-channel 缩放），AWQ 反其道而行——不搬走，而是给显著通道「放大再量化」（保护式缩放）。两者都依赖同一个恒等式 X·diag(s)⁻¹·diag(s)·W，区别在 s 怎么算、往哪边推。附完整推导、网格搜索伪代码、α 参数怎么调、以及与 GPTQ 的关系。'
pubDate: 2026-10-27
series: 量化手记
lang: zh
altLang: en
altHref: /en/blog/qt3-smoothquant-awq
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

SmoothQuant 和 AWQ 解决的是**同一个问题**，用的是**同一个数学工具**，但**方向完全相反**：

```text
Y = X W = (X · diag(s)⁻¹) · (diag(s) · W)
          ↑ 缩小激活         ↑ 放大权重
```

- **SmoothQuant**：`s` 取"激活最大值 / 权重最大值"的比值，**把难度从激活推到权重**——因为权重的分布本来就均匀好量化，激活的 outlier 才是麻烦。
- **AWQ**：`s` 取"激活显著性 ^α"，方向反过来——**把显著通道的权重放大后再量化**，等于给重要通道更高的有效分辨率，代价是让这些通道的激活看起来更极端。

一个"削平激活"，一个"放大权重"，最终目标都是让 4-bit 网格覆盖的信息更均匀。理解这个对称性，两个方法就都不需要背了。

<div class="keybox">
<strong>本篇位置：</strong>qt1 图 2 的 B 路线——改分布（缩放），不动舍入。与 qt2 的 A 路线正交：可以先 AWQ 定 scale、再 GPTQ 定舍入。下一篇 qt4 用旋转做同类的事，但能做到 W4A4。
</div>

## 1. 问题的根源：激活 outlier

### 1.1 现象

LLM 的激活里有一类特殊通道：**绝大多数 token 上值都接近 0，但极少数 token 上会爆到几十甚至上百**。这是 LLM 独有的、和 CV 模型很不一样的特性（LLM.int8() 论文最早系统记录了它）。

<div class="fig">
<svg viewBox="0 0 680 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="激活 outlier 通道的分布">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">为什么一个通道就能毁掉整层的量化</text>

  <!-- 左：通道幅值谱 -->
  <text x="20" y="54" font-size="12" font-weight="700" fill="#1d4ed8">各通道的激活幅值 |X[:, i]| 的最大值</text>
  <line x1="30" y1="200" x2="330" y2="200" stroke="#9ca3af" stroke-width="1.5"/>
  <line x1="30" y1="70" x2="30" y2="200" stroke="#9ca3af" stroke-width="1.5"/>
  <text x="14" y="74" font-size="10" fill="#6b7280">120</text>
  <text x="18" y="200" font-size="10" fill="#6b7280">0</text>

  <!-- 大量小柱 -->
  <g fill="#bfdbfe" stroke="#60a5fa">
    <rect x="36" y="190" width="7" height="10"/><rect x="46" y="188" width="7" height="12"/>
    <rect x="56" y="191" width="7" height="9"/><rect x="66" y="186" width="7" height="14"/>
    <rect x="76" y="192" width="7" height="8"/><rect x="86" y="189" width="7" height="11"/>
    <rect x="96" y="187" width="7" height="13"/><rect x="106" y="191" width="7" height="9"/>
    <rect x="116" y="185" width="7" height="15"/>
  </g>
  <!-- outlier 柱 -->
  <rect x="126" y="76" width="9" height="124" fill="#dc2626" stroke="#991b1b"/>
  <text x="130" y="70" font-size="10" fill="#b91c1c" text-anchor="middle">118</text>
  <!-- 更多小柱 -->
  <g fill="#bfdbfe" stroke="#60a5fa">
    <rect x="140" y="189" width="7" height="11"/><rect x="150" y="192" width="7" height="8"/>
    <rect x="160" y="187" width="7" height="13"/><rect x="170" y="190" width="7" height="10"/>
    <rect x="180" y="193" width="7" height="7"/><rect x="190" y="188" width="7" height="12"/>
    <rect x="200" y="191" width="7" height="9"/><rect x="210" y="186" width="7" height="14"/>
    <rect x="220" y="190" width="7" height="10"/><rect x="230" y="192" width="7" height="8"/>
  </g>
  <!-- 第二个 outlier -->
  <rect x="240" y="128" width="9" height="72" fill="#dc2626" stroke="#991b1b"/>
  <text x="244" y="122" font-size="10" fill="#b91c1c" text-anchor="middle">68</text>
  <g fill="#bfdbfe" stroke="#60a5fa">
    <rect x="254" y="189" width="7" height="11"/><rect x="264" y="191" width="7" height="9"/>
    <rect x="274" y="187" width="7" height="13"/><rect x="284" y="190" width="7" height="10"/>
    <rect x="294" y="192" width="7" height="8"/><rect x="304" y="188" width="7" height="12"/>
    <rect x="314" y="191" width="7" height="9"/>
  </g>
  <text x="30" y="222" font-size="11" fill="#6b7280">通道 index →</text>
  <text x="30" y="242" font-size="11" fill="#b91c1c">■ 约 0.1% 的通道，幅值是其余通道的 50~100×</text>

  <!-- 右：量化后 -->
  <text x="400" y="54" font-size="12" font-weight="700" fill="#b91c1c">per-tensor INT8 量化后</text>

  <!-- 网格示意 -->
  <text x="400" y="84" font-size="11" fill="#6b7280">scale = 118 / 127 ≈ 0.93</text>
  <line x1="410" y1="150" x2="640" y2="150" stroke="#9ca3af" stroke-width="1.5"/>
  <g stroke="#d1d5db">
    <line x1="410" y1="145" x2="410" y2="155"/><line x1="448" y1="145" x2="448" y2="155"/>
    <line x1="486" y1="145" x2="486" y2="155"/><line x1="524" y1="145" x2="524" y2="155"/>
    <line x1="562" y1="145" x2="562" y2="155"/><line x1="600" y1="145" x2="600" y2="155"/>
    <line x1="638" y1="145" x2="638" y2="155"/>
  </g>
  <text x="410" y="172" font-size="10" fill="#6b7280">0</text>
  <text x="638" y="172" font-size="10" fill="#6b7280" text-anchor="middle">118</text>

  <!-- 正常值塌缩 -->
  <rect x="410" y="184" width="10" height="26" fill="#bfdbfe" stroke="#60a5fa"/>
  <text x="432" y="202" font-size="11" fill="#1a1a1a">正常通道（幅值 ~1）只能落在最开始 1 个格内</text>
  <text x="432" y="220" font-size="11" fill="#b91c1c">→ 全部量化成 0 或 1，信息几乎全丢</text>

  <text x="400" y="256" font-size="12" fill="#6b7280">per-channel 也救不了 W8A8：</text>
  <text x="400" y="276" font-size="12" fill="#6b7280">矩阵乘 X·W 要求两边 scale 能合并成一次 epilogue 缩放。</text>
  <text x="400" y="296" font-size="12" fill="#6b7280">激活 per-channel + 权重 per-channel 时，外积会变成 per-token × per-channel 的二维 scale 矩阵，</text>
  <text x="400" y="312" font-size="12" fill="#6b7280">Tensor Core 的 INT8 通路无法吸收（这一条是 SmoothQuant 的出发点）。</text>
</svg>
<p class="cap">图 1：左图是真实 LLM 激活的典型形态——极少数通道幅值远超其余。右图说明 per-tensor 量化为什么必然失败：scale 必须覆盖最大值，于是 99.9% 的正常值被挤到一两个量化格里。注意右下角那句：这不是"加细粒度"就能解决的，INT8 矩阵乘的硬件契约才是真约束。</p>
</div>

### 1.2 为什么不能"给激活也用 per-channel"

直觉方案：激活按通道量化（每个特征维度一个 scale），权重也 per-channel，不就解决了吗？两个障碍：

**(1) 硬件契约。** INT8 GEMM 的硬件通路要求两侧是"整数矩阵 × 整数矩阵 + int32 累加"，缩放只能在 epilogue 做一次。如果激活是 per-token、权重是 per-channel，结果矩阵的每个元素需要 `s_x[t] · s_w[c]` 两个 scale——这变成二维索引，无法在 epilogue 用一次 broadcast 完成。要么退化成 FP16 计算，要么每个元素单独缩放，两边都失去意义。

**(2) per-token 动态 scale 有运行时开销。** 每个 token 都要先求一次 absmax，是一个额外的 reduction kernel，在 decode（每步只有 1 个 token、已经是延迟敏感）上尤其昂贵。

所以必须在**量化之前**就把分布改好，让 per-tensor（或 per-token）的粗粒度 scale 也能覆盖住。这就是 B 路线的动机。

## 2. SmoothQuant：把 outlier 从激活搬到权重

### 2.1 核心恒等式

SmoothQuant 只有一个公式：

```text
Y = X W = (X · diag(s)⁻¹) · (diag(s) · W)
```

`diag(s)` 是对角矩阵，`s ∈ R^in` 是每个输入通道的缩放因子。这里是**逐通道（按输入维度）** 缩放：第 `i` 个输入通道的激活除以 `s_i`，对应的权重行乘以 `s_i`。乘积不变。

关键在于：**`s_i` 可以从激活里"取走"难度，塞给权重。**

```text
s_j = max(|X_j|)^α / max(|W_j|)^(1-α)        α ∈ [0, 1]
```

- `α = 0`：`s_j = 1/max|W_j|` → 完全不动，退化为原始（激活难量化）
- `α = 1`：`s_j = max|X_j|` → 把激活的最大值归一化到 1（激活好量化了，但权重变得难量化）
- `α = 0.5`：迁移一半，两边都"中等难度"

```python
import torch

def smoothquant_scale(X, W, alpha=0.5):
    """
    X: [T, in] 校准激活   W: [in, out] 权重
    返回逐输入通道的平滑因子 s，以及变换后可直接量化的 (X', W')
    """
    # 每个输入通道的激活最大值 / 权重最大值
    act_max = X.abs().amax(dim=0).clamp(min=1e-5)     # [in]
    w_max   = W.abs().amax(dim=0)                     # [in]
    w_max   = w_max.clamp(min=1e-5)

    s = (act_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)

    X_smooth = X / s.unsqueeze(0)      # 激活除以 s
    W_smooth = W * s.unsqueeze(1)      # 权重乘以 s
    return s, X_smooth, W_smooth
```

### 2.2 α 为什么是 0.5

`α=0.5` 是 SmoothQuant 论文的默认值，理由是**均衡**：让激活的动态范围和权重的动态范围在量化后"难度相当"。

<div class="fig">
<svg viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="α 参数的迁移效果">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">α 控制「难度」在激活和权重之间的分配</text>

  <!-- 三根刻度条 -->
  <!-- α=0 -->
  <text x="20" y="60" font-size="12" font-weight="700" fill="#b91c1c">α = 0（不平滑）</text>
  <text x="20" y="80" font-size="11" fill="#6b7280">激活动态范围</text>
  <rect x="120" y="70" width="440" height="14" rx="3" fill="#fecaca" stroke="#ef4444"/>
  <text x="340" y="81" font-size="10" fill="#991b1b" text-anchor="middle">0 ~ 118，跨 2 个数量级 → 量化崩</text>
  <text x="20" y="102" font-size="11" fill="#6b7280">权重动态范围</text>
  <rect x="120" y="92" width="90" height="14" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <text x="222" y="103" font-size="10" fill="#047857">0 ~ 0.05，均匀好量化</text>

  <!-- α=0.5 -->
  <text x="20" y="142" font-size="12" font-weight="700" fill="#047857">α = 0.5（默认）</text>
  <text x="20" y="162" font-size="11" fill="#6b7280">激活动态范围</text>
  <rect x="120" y="152" width="200" height="14" rx="3" fill="#fef3c7" stroke="#f59e0b"/>
  <text x="332" y="163" font-size="10" fill="#854F0B">0 ~ 11，降到 1/10</text>
  <text x="20" y="184" font-size="11" fill="#6b7280">权重动态范围</text>
  <rect x="120" y="174" width="200" height="14" rx="3" fill="#fef3c7" stroke="#f59e0b"/>
  <text x="332" y="185" font-size="10" fill="#854F0B">0 ~ 0.55，涨 10× —— 两边都还能用 INT8 覆盖</text>

  <!-- α=1 -->
  <text x="20" y="224" font-size="12" font-weight="700" fill="#1d4ed8">α = 1（全推给权重）</text>
  <text x="20" y="244" font-size="11" fill="#6b7280">激活已归一化到 1，但权重动态范围涨到 0~5.9，反而成为新的 outlier 源</text>
</svg>
<p class="cap">图 2：α 不是"越大越好"。α=1 只是把问题从一边搬到另一边，而权重的 outlier 会被 GEMM 放大到所有输出通道上，后果往往更糟。实践里 α 在 0.4~0.7 之间，按模型和层做网格搜索（PPL 或输出 MSE）的效果最稳。</p>
</div>

### 2.3 落地细节：融合到 LayerNorm / 前一个线性层

SmoothQuant 的工程美感在于：**`diag(s)⁻¹` 这个除法不需要单独一次 kernel**，它可以被吸收到前一层的输出里。

```text
标准 Transformer 块：
    x → LayerNorm → (X W_q, X W_k, X W_v, X W_gate) →

SmoothQuant 后：
    x → LayerNorm → ×diag(s)⁻¹ → (X' W_q·diag(s), ...)
       ↑ 融合：LN 的 γ 直接乘上 1/s，零额外开销
```

具体做法是把 `1/s` 折进 LayerNorm 的权重 `γ`：

```python
# 融合前：h = LN(x) * gamma + beta
# 融合后：h = (LN(x) / s) * gamma + beta  →  gamma' = gamma / s
gamma_new = gamma / s
# 之后所有权重行都乘 s：W_new = W * s.unsqueeze(1)
```

这样推理图上**完全没有多出来的算子**，只有权重的数值变了。这是 SmoothQuant 能被 vLLM / TensorRT-LLM / SGLang 广泛支持的原因——它是一个**纯离线变换**，部署侧零成本。

<div class="warnbox">
<strong>注意一个实现陷阱：</strong>残差连接会破坏这个融合。如果 <code>x</code> 同时流向 LN 之外（残差支路），那么"除以 s"就没法只加在 LN 上——它会影响残差累加。所以 SmoothQuant 的 <code>s</code> 必须按"LN 之后的激活"统计，并且只作用于 LN 的输出，不能动残差流本身。这也是为什么 SmoothQuant 主要用在 pre-norm Transformer（LN 在支路上）而不是 post-norm。
</div>

### 2.4 效果与局限

**效果**：LLaMA / OPT 系列在 W8A8 下 PPL 损失通常在 **0.05~0.2**，相比 per-tensor 朴素的 W8A8（PPL 涨几个点）是数量级改善。这是 W8A8 在数据中心部署能成立的直接原因。

**局限**：

1. **它做不到 W4A8 / W4A4。** scale 迁移只能把难度拉平，不能凭空增加分辨率。4-bit 激活的信息量不足以承载 outlier 通道，α 怎么调都不够。
2. **它假设 outlier 通道在用校准集统计后是稳定的。** 实测中 outlier 通道的"身份"（哪个 index）相当稳定，这才让它可行。但**幅值**会随输入变化，尤其是长上下文下 attention 的 softmax 输出分布会漂移。
3. **对 attention 的 QK^T 不适用。** 那里的 outlier 出在 softmax 之后（P 矩阵），是逐 token 动态的，SmoothQuant 的静态 scale 处理不了——这个问题在 qt5（KV Cache 量化）和 op3（EagleQBSA 的 P 跟随 V）里会再遇到。

## 3. AWQ：反过来，给显著通道"放大再量化"

### 3.1 思路的反转

AWQ（Activation-aware Weight Quantization，MLSys 2024 Best Paper）的出发点和 SmoothQuant 不同：它瞄准的是 **W4A16**（只量化权重，激活保持 FP16），所以"激活的位宽"不是问题，问题是**权重的 4-bit 精度不够覆盖所有通道**。

它的核心观察：**权重的重要性不只看权重本身的大小，而要看"经过它的激活有多大"。**

```text
对输出 Y = Σ_i X[:, i] · W[i, :]

第 i 个输入通道对 Y 的贡献强度 ∝ ||X[:, i]|| · ||W[i, :]||
```

所以即使某个权重值很小，如果它对应的激活通道很大，它量化出的误差也会被放大。**这类通道必须保护。**

### 3.2 保护的方式：等价缩放挡不住 absmax，但能改变分配

AWQ 用的恒等式和 SmoothQuant 完全一样：

```text
Y = X W = (X · diag(s)⁻¹) · (diag(s) · W)
```

但方向相反：**AWQ 让 `s_i > 1`，即把显著通道的权重放大、激活缩小。**

这里必须把一个常见的误讲讲清楚。很多博客说"放大权重 → 相对误差不变 → 所以无损"，这是错的：group 内 `absmax` 会变，`Δ` 也会变，误差不是简单的不变。真实机制更精细，也是 AWQ 论文里最关键的一段论证。

**先看 group 量化的误差结构。** 一个 group 共享 `Δ = absmax(W_group) / (2^(b-1) − 1)`，任意元素的绝对误差上界是 `Δ/2`。注意这个误差**对所有元素相同**——它由 group 内的最大值决定，和元素自己的大小无关。

于是：

- **大权重**（接近 absmax 的元素）：`Δ/2` 相对它自己的幅值很小 → 相对误差低。
- **小权重**：`Δ/2` 相对它自己的幅值很大 → 相对误差高。

**而 AWQ 的观察恰恰是：显著通道往往权重不大，但激活很大。** 这类通道正好落在"相对误差最高"的位置上。

**放大 `s_i` 之后发生了什么？** 分两种情况：

```text
情况 A：被放大的通道成为 group 内新的 absmax
  → Δ 变大 s 倍，group 内所有元素的绝对误差都变大 s 倍
  → 该通道自己：绝对误差 ↑s 但反量化时 ÷s → 净效果 = 不变
  → 其余通道：绝对误差 ↑s，毫无补偿 → 变差   ✗ 得不偿失

情况 B：group 内原本就有更大的权重（absmax 由别的通道决定）
  → Δ 基本不变
  → 被放大的通道：相对幅值从 w 变成 s·w，占用的格数 ×s
     绝对误差仍是 Δ/2，反量化 ÷s 后 → 误差 ÷s   ✓ 净赚
  → 其余通道：Δ 不变，误差不变
```

**所以 AWQ 的净收益取决于"显著通道的权重是否本来就不是 group 内的最大值"。** 实测中这一条成立得相当好：LLM 权重的分布是"少数大值 + 长尾小值"，而激活显著的通道通常落在长尾里。`α` 的网格搜索本质上就是在找一个**既能把显著通道拉出长尾、又不至于把它们推成新 absmax** 的平衡点。

<div class="fig">
<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="AWQ 在 group 内重新分配量化分辨率">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">同一个 group（128 个权重共享一个 Δ）：Δ 由最大权重锚定，与缩放无关</text>

  <!-- 上：缩放前 -->
  <text x="20" y="58" font-size="12" font-weight="700" fill="#6b7280">缩放前</text>
  <line x1="90" y1="98" x2="620" y2="98" stroke="#9ca3af" stroke-width="1.5"/>
  <g stroke="#d1d5db">
    <line x1="90" y1="92" x2="90" y2="104"/><line x1="166" y1="92" x2="166" y2="104"/>
    <line x1="242" y1="92" x2="242" y2="104"/><line x1="318" y1="92" x2="318" y2="104"/>
    <line x1="394" y1="92" x2="394" y2="104"/><line x1="470" y1="92" x2="470" y2="104"/>
    <line x1="546" y1="92" x2="546" y2="104"/><line x1="620" y1="92" x2="620" y2="104"/>
  </g>
  <text x="90" y="120" font-size="10" fill="#6b7280" text-anchor="middle">0</text>
  <text x="620" y="120" font-size="10" fill="#6b7280" text-anchor="middle">absmax = 2.5</text>

  <!-- 通道 A：大权重，非显著 -->
  <circle cx="602" cy="98" r="6" fill="#9ca3af" stroke="#4b5563" stroke-width="1.5"/>
  <text x="602" y="84" font-size="10" fill="#374151" text-anchor="middle">通道 A：w=2.48</text>
  <text x="602" y="136" font-size="10" fill="#6b7280" text-anchor="middle">占据末尾格，本身误差就小</text>

  <!-- 通道 B：小权重，显著 -->
  <circle cx="238" cy="98" r="6" fill="#dc2626"/>
  <text x="238" y="84" font-size="10" fill="#b91c1c" text-anchor="middle">通道 B：w=0.60（激活大 40×）</text>
  <text x="238" y="136" font-size="10" fill="#b91c1c" text-anchor="middle">只占 1.9 个格 → 量化到 0.71，相对误差 19%</text>

  <text x="20" y="158" font-size="11" fill="#6b7280">Δ = 2.5 / 7 ≈ 0.357，group 内所有元素的绝对误差上界都是 Δ/2 ≈ 0.179</text>

  <!-- 分隔 -->
  <line x1="20" y1="176" x2="660" y2="176" stroke="#e5e7eb" stroke-width="1.5"/>

  <!-- 下：缩放后 -->
  <text x="20" y="204" font-size="12" font-weight="700" fill="#047857">s_B = 4 缩放后（A 的 s ≈ 1，不参与缩放）</text>
  <line x1="90" y1="244" x2="620" y2="244" stroke="#9ca3af" stroke-width="1.5"/>
  <g stroke="#d1d5db">
    <line x1="90" y1="238" x2="90" y2="250"/><line x1="166" y1="238" x2="166" y2="250"/>
    <line x1="242" y1="238" x2="242" y2="250"/><line x1="318" y1="238" x2="318" y2="250"/>
    <line x1="394" y1="238" x2="394" y2="250"/><line x1="470" y1="238" x2="470" y2="250"/>
    <line x1="546" y1="238" x2="546" y2="250"/><line x1="620" y1="238" x2="620" y2="250"/>
  </g>
  <text x="90" y="266" font-size="10" fill="#6b7280" text-anchor="middle">0</text>
  <text x="620" y="266" font-size="10" fill="#6b7280" text-anchor="middle">absmax 仍是 2.5</text>

  <!-- A 不动 -->
  <circle cx="602" cy="244" r="6" fill="#9ca3af" stroke="#4b5563" stroke-width="1.5"/>
  <text x="602" y="230" font-size="10" fill="#374151" text-anchor="middle">A 不变</text>

  <!-- B 放大 -->
  <circle cx="610" cy="226" r="6" fill="#047857"/>
  <text x="560" y="212" font-size="10" fill="#047857" text-anchor="middle">B：0.60×4 = 2.40</text>
  <text x="400" y="292" font-size="11" fill="#047857">量化 2.40 → 2.5，绝对误差 0.10；除以 s_B=4 还原 → 误差 0.025</text>
  <text x="400" y="312" font-size="11" fill="#047857">相对 w=0.60 误差从 19% 降到 4.2%，而 group 内 Δ 完全没变 ← 净收益</text>
</svg>
<p class="cap">图 3：AWQ 的收益不是"网格变密了"，而是"网格没变，但显著通道在网格里占的格数变多了"。前提是 group 内的 absmax 由另一个不显著的通道锚定——所以 AWQ 的 <code>α</code> 必须做网格搜索，推出过头就会把自己变成新的 absmax，收益立刻反转（对应上文情况 A）。</p>
</div>

这也解释了 AWQ 论文里那个反直觉的消融结论：**只保护约 1% 的通道就接近饱和收益**。因为问题的规模本来就不大——真正需要救的通道很少（图 1 里也只有约 0.1%），只要精准定位，把它们的权重拉出长尾即可。


### 3.3 显著性怎么算：`s = mean(|X|)^α`

AWQ 的 scale 不是像 SmoothQuant 那样用 `max`，而是用一个**幂次化的平均**：

```text
s_j = mean(|X_j|)^α       α ∈ [0, 1]，默认 0.5
```

三个设计选择各有理由：

**为什么用 `mean` 而不是 `max`？** `max` 是单个 token 的极值，噪声极大且对校准集敏感；`mean` 是整条通道的平均能量，稳定得多。而且 AWQ 保护的是"**整体上经常出现大激活**"的通道，而不是"偶尔爆一次"的通道——后者可能是噪声。

**为什么加幂 `α`？** 这是经验性的"软化"。直接线性使用 `mean(|X|)` 会让少数极显著通道获得过大的 `s`（比如 100×），导致它们的权重被推到量化范围外；`α = 0.5` 用平方根压缩动态范围。论文里 α 在 0~1 上做网格搜索，0.5 附近最稳。

**为什么 `s` 要归一化？** AWQ 最后的 `s` 会除以它的几何平均（`s / geometric_mean(s)`），保证整体权重尺度不变。否则所有 `s > 1` 时权重整体变大，4-bit 网格会整体偏移。

```python
import torch

def awq_scale(X, W, alpha=0.5, group_size=128):
    """
    AWQ：算显著性 → 幂次化 → 归一化 → 等价缩放到权重。
    X: [T, in] 校准激活   W: [in, out]
    """
    # ① 逐输入通道的平均激活幅值（显著性）
    sig = X.abs().mean(dim=0).clamp(min=1e-5)        # [in]

    # ② 幂次化：压缩动态范围，避免极端 s
    s = sig.pow(alpha)

    # ③ 按 group 内归一化（AWQ 实际按 group 做，让每组几何均值为 1）
    if group_size > 0:
        sg = s.reshape(-1, group_size)
        s = (sg / sg.prod(dim=1, keepdim=True).pow(1.0 / group_size)).reshape(-1)
    else:
        s = s / s.prod().pow(1.0 / s.numel())

    # ④ 等价缩放：权重乘 s（要量化的那侧），激活除 s（无损那侧）
    W_scaled = W * s.unsqueeze(1)
    return s, W_scaled
```

<div class="warnbox">
<strong>AWQ 与 SmoothQuant 的 <code>s</code> 方向：别记反了。</strong><br>
两者公式相同（<code>X/s</code> 和 <code>W*s</code>），但语义相反：<br>
· <strong>SmoothQuant</strong>：<code>s</code> 来自 <code>act_max / w_max</code>，<strong>大激活通道得到大 <code>s</code> → 激活被大幅缩小 → outlier 被搬走</strong>。目的是让<strong>激活</strong>能压进 INT8。<br>
· <strong>AWQ</strong>：<code>s</code> 来自 <code>mean(|X|)^α</code>，<strong>大激活通道得到大 <code>s</code> → 权重被大幅放大 → 获得更细分辨率</strong>。目的是让<strong>权重</strong>在 4-bit 下更准。<br>
<strong>同样是"大激活通道 → 大 s"，但一个是在救激活，一个是在救权重。</strong>因为 AWQ 里激活不量化，所以"激活被缩小"不构成损失；而在 SmoothQuant 里激活要被量化，"激活被缩小"正是救命的那一步。
</div>

### 3.4 AWQ 的实际卖点：不依赖反向传播

AWQ 论文强调的另一个优势是**它不需要 GPTQ 那样的二阶信息，也不需要 QAT 的梯度**，scale 搜索是一个**离散网格搜索**：

```text
对每个 α ∈ {0, 0.05, ..., 1.0}:
    用 s(α) 缩放权重 → 量化 → 反量化
    测量输出误差 ||X·W − X·Ŵ||²
选误差最小的 α
```

因为 `α` 只是一维标量，20 次搜索就够，单卡几分钟。这让 AWQ 的**校准成本比 GPTQ 低一个量级**，而且对超参不敏感（不存在 GPTQ 那种 `damp` 调不好就崩的风险）。

它的核心保护机制还有一个细节值得记：**AWQ 只保护约 1% 的显著通道，就能挽回大部分精度损失**。论文的消融显示，保护 0.1%~1% 的通道已接近饱和收益。这和"激活 outlier 占比极小"（图 1）的事实完全对应——问题的规模本来就不大，只要精准定位。

## 4. 三种方法的横向对比

现在可以把 qt2 和 qt3 的方法放在一起看：

| 维度 | RTN | GPTQ | SmoothQuant | AWQ |
|---|---|---|---|---|
| **改什么** | 无（基线） | 只改舍入 | 改分布（削激活） | 改分布（放大权重） |
| **需要校准数据** | 否 | 是（求 H） | 是（求 act_max） | 是（求 mean\|X\|） |
| **需要反向传播** | 否 | 否 | 否 | 否 |
| **目标格式** | 任意 | W4A16 / W4A8 | **W8A8** | W4A16 |
| **计算成本** | 秒级 | 分钟~小时 | 秒级 | 分钟级 |
| **核心公式** | `round(W/s)` | `+ H⁻¹ 补偿` | `s = amax_X^α / amax_W^(1-α)` | `s = mean\|X\|^α` |
| **能否叠加** | — | ✅ 与 AWQ/SQ 叠加 | ✅ 与 GPTQ 叠加 | ✅ 与 GPTQ 叠加 |
| **典型 PPL 损失（LLaMA-7B）** | W4: 0.5+ | W4: 0.1~0.3 | W8A8: 0.05~0.2 | W4: 0.1~0.25 |

**怎么组合？** 生产里最常见的是 **AWQ 定 scale → GPTQ 定舍入**：

```text
① 用 mean(|X|)^α 算出 s
② 权重乘 s、激活除 s（等价变换）
③ 对缩放后的权重跑 GPTQ（此时分布更均匀，H 条件数更好）
④ 部署时把 s 折进前一层
```

这样既拿到了 AWQ 的"保护显著通道"，又拿到了 GPTQ 的"误差补偿"。缺点是推理 kernel 要同时支持两套逻辑，只有部分引擎（如 vLLM 的部分 kernel）做了融合。

## 5. 选型建议

<div class="keybox">
<strong>一条判断链：</strong><br>
① 只量化权重（W4A16，decode 瓶颈）→ <strong>AWQ</strong>（快、稳、无需调参），追求极致精度再加 GPTQ<br>
② 权重和激活都量化（W8A8，prefill 瓶颈）→ <strong>SmoothQuant</strong>（这是它的唯一主场）<br>
③ 有 FP8 硬件 → 直接 FP8，通常不需要 SmoothQuant（E4M3 的动态范围比 INT8 大得多，outlier 容忍度更高）<br>
④ 要做到 W4A8 / W4A4 → 这两个都不够，看 qt4
</div>

一个常被忽略的点：**SmoothQuant 在 FP8 时代地位下降了。** 因为 FP8 E4M3 有 4 位指数、动态范围约 ±448，而 INT8 只有 ±127 且均匀分布——FP8 对 outlier 的天然容忍度高得多。很多团队实测发现"直接用 FP8 per-tensor 就够，不需要 SmoothQuant"。SmoothQuant 现在的价值主要在**只有 INT8 通路的硬件**（老 GPU、部分国产芯片）。

反过来，**AWQ 的地位在上升**：端侧、单卡部署、显存受限场景是当下的主流需求，而 W4A16 正好是这些场景的最优解。

## 6. 系列导航

- 上一篇 **qt2 · RTN 与 GPTQ**：A 路线，只改舍入。
- 下一篇 **qt4 · QuaRot / SpinQuant / DuQuant**：C 路线，用旋转把 outlier 真正"摊平"，这是 W4A4 的唯一出路。
- **qt5 · KV Cache 量化与 FP4**：另一条正交的压缩维度。
- **qt1 · 量化的基本语言**：α、粒度、对称性的基础。

站内相关：**op3（昇腾 EagleQBSA）** 的"P 跟随 V 用 FP8 按通道量化"是这里结论的一个实证——视频生成里 V 的分布超出 INT8 适配能力，只能上 FP8；**op2** 是系列导读；**fw4** 从引擎视角讲了 W8A8/W4A16 的工程落地差异。
