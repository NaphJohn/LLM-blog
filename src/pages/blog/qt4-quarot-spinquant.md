---
title: '量化手记（四）：QuaRot / SpinQuant / DuQuant——用旋转把 outlier 摊平，W4A4 的唯一出路'
description: 'SmoothQuant 把 outlier 搬走、AWQ 把显著通道放大，两者都没真正消灭 outlier，所以都做不到 W4A4。旋转路线换了个角度：用一个正交矩阵 R 把激活和权重同时转一下，模型输出数学不变，但那个 118 的 outlier 会被摊成 128 个 10。本文讲清旋转为什么"免费"、只能插在计算图的哪四个位置（R1–R4）、SwiGLU 的逐元素乘为什么让 R3 必须在线算、Q/K 旋转为什么白送，再对比 QuaRot（手挑 Hadamard）、SpinQuant（Cayley 学旋转）、DuQuant（块对角 + zigzag 置换）三代方案，最后回答一个现实问题：W4A4 今天能上线吗。'
pubDate: 2026-10-27
series: 量化手记
lang: zh
altLang: en
altHref: /en/blog/qt4-quarot-spinquant
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

前三篇的方法都在**对抗** outlier：RTN 无视它、GPTQ 用 Hessian 补偿它、SmoothQuant 把它搬到权重、AWQ 给显著通道开小灶。它们的共同上限是 **W4A16**——激活不动，所以激活里的 outlier 不构成问题；但只要你想量化激活（W4A8 / W4A4），outlier 就无法回避。

旋转路线换了个完全不同的角度：

```text
不搬走 outlier，而是把它「摊平」到所有通道上。

x    = [118, 2, 1, 0, ...]        ← 一个通道吃掉全部能量
x·R  = [10.4, 10.4, ..., 10.4]    ← 128 个通道平均分
```

`R` 是正交矩阵，`R Rᵀ = I`。把它塞进 `Y = XW` 的中间：

```text
Y = X W = (X R)(Rᵀ W)
          ↑ 转激活   ↑ 转权重，抵消掉
```

模型输出的数学**完全不变**（不是近似，是恒等式），但激活和权重的分布被重新打散了。118 变成 10.4 之后，per-token INT4 量化突然就可行了——动态范围压缩了 √128 ≈ 11 倍。

<div class="keybox">
<strong>本篇位置：</strong>qt1 图 2 的 C 路线——改分布（旋转），不动舍入、不搬移、不放大。它是唯一能支撑 <strong>W4A4</strong>（权重和激活都 4-bit）的路线。代价是每层多几个 fused kernel，以及要改模型的权重布局。qt5 会讲它的正交搭档：KV Cache 量化。
</div>

## 1. 旋转为什么能"无损"消掉 outlier

### 1.1 第一步：把正交矩阵塞进矩阵乘法

对任意正交矩阵 `R`（`R Rᵀ = I`）：

```text
Y = X W
  = X (R Rᵀ) W
  = (X R) (Rᵀ W)
  = X̃  W̃
```

`X̃ = X R` 是旋转后的激活，`W̃ = Rᵀ W` 是反向旋转后的权重。因为 `R Rᵀ` 精确等于单位阵，这一步**没有引入任何误差**。它只是一个重新参数化（reparameterization）。

关键在于：`W̃` 是**离线**算好的。推理时权重已经是旋转过的版本，激活只需要在进入 GEMM 之前乘一次 `R`——如果 `R` 选得好（见 1.3），这次乘法的成本可以做到接近零。

### 1.2 第二步：一个 outlier 会被摊成 d 份

设 `x` 只在第 5 个通道有一个巨大的值 `M`，其余通道接近 0。取 `R` 为归一化 Hadamard 矩阵，则：

```text
x̃[k] = Σ_i x[i] · H[i, k] ≈ M · H[5, k] = ± M/√d
```

**每一个输出通道都拿到了 `M/√d`。** 对 `d = 128`，118 变成 10.4。量化的 `absmax` 从 118 降到 10.4，量化步长 `Δ` 同步缩小 11 倍，也就是**有效精度提升约 3.5 bit**。

这就是旋转的全部魔力：它不消灭能量，只是把能量从 1 个通道均摊到 `d` 个通道。

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="正交旋转把 outlier 摊平到所有通道">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">同一个向量，转一下，动态范围压 11 倍</text>

  <text x="20" y="54" font-size="12" font-weight="700" fill="#b91c1c">旋转前：x</text>
  <line x1="30" y1="215" x2="320" y2="215" stroke="#9ca3af" stroke-width="1.5"/>
  <line x1="30" y1="70" x2="30" y2="215" stroke="#9ca3af" stroke-width="1.5"/>
  <text x="8" y="74" font-size="10" fill="#6b7280">120</text>
  <text x="18" y="215" font-size="10" fill="#6b7280">0</text>

  <g fill="#fecaca" stroke="#f87171">
    <rect x="38" y="211" width="6" height="4"/><rect x="47" y="210" width="6" height="5"/>
    <rect x="56" y="212" width="6" height="3"/><rect x="65" y="211" width="6" height="4"/>
    <rect x="74" y="210" width="6" height="5"/><rect x="83" y="211" width="6" height="4"/>
    <rect x="92" y="212" width="6" height="3"/><rect x="101" y="210" width="6" height="5"/>
  </g>
  <rect x="110" y="72" width="8" height="143" fill="#dc2626" stroke="#991b1b"/>
  <text x="114" y="66" font-size="10" fill="#b91c1c" text-anchor="middle">118</text>
  <g fill="#fecaca" stroke="#f87171">
    <rect x="126" y="211" width="6" height="4"/><rect x="135" y="212" width="6" height="3"/>
    <rect x="144" y="210" width="6" height="5"/><rect x="153" y="211" width="6" height="4"/>
    <rect x="162" y="213" width="6" height="2"/><rect x="171" y="210" width="6" height="5"/>
    <rect x="180" y="211" width="6" height="4"/><rect x="189" y="212" width="6" height="3"/>
    <rect x="198" y="210" width="6" height="5"/><rect x="207" y="211" width="6" height="4"/>
    <rect x="216" y="212" width="6" height="3"/><rect x="225" y="213" width="6" height="2"/>
    <rect x="234" y="210" width="6" height="5"/><rect x="243" y="211" width="6" height="4"/>
    <rect x="252" y="212" width="6" height="3"/><rect x="261" y="210" width="6" height="5"/>
  </g>
  <text x="288" y="238" font-size="10" fill="#6b7280">128 个通道</text>
  <text x="20" y="264" font-size="11" fill="#b91c1c" font-weight="700">absmax = 118 → INT4 步长 15.7 → 除 outlier 外全部归零</text>

  <path d="M 336 150 L 372 150" stroke="#1d4ed8" stroke-width="2" marker-end="url(#a4)"/>
  <defs><marker id="a4" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L6,3 L0,6 z" fill="#1d4ed8"/></marker></defs>
  <text x="354" y="142" font-size="11" fill="#1d4ed8" text-anchor="middle" font-weight="700">× R</text>

  <text x="384" y="54" font-size="12" font-weight="700" fill="#15803d">旋转后：x·R（归一化 Hadamard）</text>
  <line x1="392" y1="215" x2="660" y2="215" stroke="#9ca3af" stroke-width="1.5"/>
  <line x1="392" y1="70" x2="392" y2="215" stroke="#9ca3af" stroke-width="1.5"/>
  <text x="376" y="106" font-size="10" fill="#6b7280">12</text>
  <text x="382" y="215" font-size="10" fill="#6b7280">0</text>

  <g fill="#bbf7d0" stroke="#4ade80">
    <rect x="400" y="92" width="6" height="123"/><rect x="409" y="94" width="6" height="121"/>
    <rect x="418" y="91" width="6" height="124"/><rect x="427" y="95" width="6" height="120"/>
    <rect x="436" y="92" width="6" height="123"/><rect x="445" y="93" width="6" height="122"/>
    <rect x="454" y="95" width="6" height="120"/><rect x="463" y="91" width="6" height="124"/>
    <rect x="472" y="94" width="6" height="121"/><rect x="481" y="92" width="6" height="123"/>
    <rect x="490" y="95" width="6" height="120"/><rect x="499" y="93" width="6" height="122"/>
    <rect x="508" y="91" width="6" height="124"/><rect x="517" y="94" width="6" height="121"/>
    <rect x="526" y="92" width="6" height="123"/><rect x="535" y="95" width="6" height="120"/>
    <rect x="544" y="93" width="6" height="122"/><rect x="553" y="91" width="6" height="124"/>
    <rect x="562" y="94" width="6" height="121"/><rect x="571" y="92" width="6" height="123"/>
    <rect x="580" y="95" width="6" height="120"/><rect x="589" y="93" width="6" height="122"/>
    <rect x="598" y="91" width="6" height="124"/><rect x="607" y="94" width="6" height="121"/>
    <rect x="616" y="92" width="6" height="123"/><rect x="625" y="95" width="6" height="120"/>
    <rect x="634" y="93" width="6" height="122"/><rect x="643" y="91" width="6" height="124"/>
  </g>
  <line x1="392" y1="96" x2="660" y2="96" stroke="#15803d" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="658" y="88" font-size="10" fill="#15803d" text-anchor="end">absmax ≈ 10.4 = 118/√128</text>

  <text x="384" y="264" font-size="11" fill="#15803d" font-weight="700">absmax = 10.4 → INT4 步长 1.4 → 每个通道都还活着</text>
</svg>
<p class="cap">图 1：旋转不减少能量，只重新分配能量。左右两图的总能量（L2 范数）完全相同——这正是"数学不变"的直观体现。</p>
</div>

### 1.3 Hadamard：唯一又好又快的选择

`R` 可以随便挑一个正交矩阵吗？理论上可以——随机正交矩阵同样能打散 outlier（高维集中现象）。但工程上只有一个答案：**归一化 Hadamard 矩阵**。

理由有三个：

```text
① 正交：H Hᵀ = I，满足"免费"的前提
② 快：d = 128 时，快速 Walsh-Hadamard 变换（FWHT）只要 log₂(128) = 7 轮蝶形
       乘加量 O(d log d) = 128×7 ≈ 900 次，对比 GEMM 的 d² = 16384 次，可忽略
③ 几乎无浮点乘法：蝶形只有加减，最后一次性乘 1/√d。定点友好，数值误差极小
```

`H` 的构造是 Kronecker 幂：`H_128 = H_2^⊗7`，其中 `H_2 = (1/√2)[[1,1],[1,-1]]`。

```python
import math, torch

def fwht(x):
    """归一化 Hadamard 变换。x: [..., d]，d 必须是 2 的幂。O(d log d)，只有加减。"""
    y = x.clone()
    d = x.shape[-1]
    h = 1
    while h < d:
        y = y.view(*y.shape[:-1], d // (2 * h), 2, h)
        a, b = y[..., 0, :].clone(), y[..., 1, :].clone()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.reshape(*x.shape[:-1], d)
        h *= 2
    return y / math.sqrt(d)

# 验证正交性
H = torch.eye(128)
H = fwht(H)                       # 对单位阵逐行变换即得 H 本身
assert torch.allclose(H @ H.T, torch.eye(128), atol=1e-5)
```

这个读写模式是标准蝶形网络，可以完全在 shared memory 里做完，也可以融进前一个算子的 epilogue。**这就是为什么旋转路线的运行时开销能压到 1~3%。**

### 1.4 但要小心：旋转只在"线性部分"免费

这是整个路线最容易踩的坑。恒等式 `Y = (XR)(RᵀW)` 成立的前提是：`R` 的插入点前后**都是线性映射**。

一旦中间夹了非线性，`R` 就出不来了：

```text
✓ 线性：  (X R)(Rᵀ W) = X W

✗ RMSNorm 的 γ：  (x ⊙ γ) · W  ≠  x · (γ ⊙ ...)   ← γ 是逐元素乘，不能穿过旋转
✗ SwiGLU：        (R g) ⊙ (R u) ≠ R (g ⊙ u)      ← 逐元素积不跟旋转交换
✗ RoPE：          RoPE(x R) ≠ RoPE(x) R          ← 位置编码的 2D 块结构对不上
```

所以旋转不能"想插哪就插哪"。能插的位置**只有四个**，而且每个位置的代价不一样。

## 2. 四个插入位置：R1–R4

<div class="fig">
<svg viewBox="0 0 680 520" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Transformer 块中四个旋转插入位置">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">旋转只能插在这四个位置（以 LLaMA 类 decoder block 为例）</text>

  <rect x="20" y="36" width="14" height="10" fill="#dbeafe" stroke="#1d4ed8"/>
  <text x="40" y="45" font-size="10" fill="#6b7280">离线折叠（权重改一下，运行时零开销）</text>
  <rect x="280" y="36" width="14" height="10" fill="#ffedd5" stroke="#ea580c"/>
  <text x="300" y="45" font-size="10" fill="#6b7280">在线 kernel（每次前向都要算）</text>

  <g font-size="11" text-anchor="middle">
    <rect x="90" y="62" width="230" height="26" rx="4" fill="#f3f4f6" stroke="#6b7280"/>
    <text x="205" y="79" fill="#374151">残差流 x（已在旋转基下）</text>

    <rect x="90" y="106" width="230" height="26" rx="4" fill="#fff" stroke="#6b7280"/>
    <text x="205" y="123" fill="#374151">RMSNorm(x) · γ</text>

    <rect x="20" y="146" width="44" height="22" rx="11" fill="#dbeafe" stroke="#1d4ed8"/>
    <text x="42" y="161" font-size="11" font-weight="700" fill="#1d4ed8">R1</text>
    <rect x="90" y="144" width="230" height="26" rx="4" fill="#fff" stroke="#1d4ed8" stroke-width="1.5"/>
    <text x="205" y="161" fill="#1d4ed8">q/k/v/gate/up_proj</text>
    <text x="336" y="152" font-size="10" fill="#1d4ed8">γ 夹在中间，折叠公式：</text>
    <text x="336" y="167" font-size="10" fill="#1d4ed8">W' = W · diag(γ) · R · diag(1/γ)</text>

    <rect x="90" y="184" width="230" height="26" rx="4" fill="#fff" stroke="#6b7280"/>
    <text x="205" y="201" fill="#374151">RoPE（作用在 head_dim 的 2D 对上）</text>

    <rect x="20" y="224" width="44" height="22" rx="11" fill="#ffedd5" stroke="#ea580c"/>
    <text x="42" y="239" font-size="11" font-weight="700" fill="#ea580c">R4</text>
    <rect x="90" y="222" width="230" height="26" rx="4" fill="#fff" stroke="#ea580c" stroke-width="1.5"/>
    <text x="205" y="239" fill="#ea580c">Q ← Q·R，K ← K·R（同一个 R）</text>
    <text x="336" y="231" font-size="10" fill="#ea580c">QKᵀ = Q R Rᵀ Kᵀ = Q Kᵀ → 分数不变，</text>
    <text x="336" y="246" font-size="10" fill="#ea580c">靠 softmax 内部抵消，不需要补偿项</text>

    <rect x="90" y="262" width="230" height="26" rx="4" fill="#fff" stroke="#6b7280"/>
    <text x="205" y="279" fill="#374151">softmax(QKᵀ/√d) · V</text>

    <rect x="20" y="302" width="44" height="22" rx="11" fill="#dbeafe" stroke="#1d4ed8"/>
    <text x="42" y="317" font-size="11" font-weight="700" fill="#1d4ed8">R2</text>
    <rect x="90" y="300" width="230" height="26" rx="4" fill="#fff" stroke="#1d4ed8" stroke-width="1.5"/>
    <text x="205" y="317" fill="#1d4ed8">o_proj（逐 head 旋转）</text>
    <text x="336" y="309" font-size="10" fill="#1d4ed8">P·(V R) = (P V) R → 补偿进 W_o：</text>
    <text x="336" y="324" font-size="10" fill="#1d4ed8">W_o' = W_o · R（按 head 分块对角）</text>

    <rect x="90" y="340" width="230" height="26" rx="4" fill="#f3f4f6" stroke="#6b7280"/>
    <text x="205" y="357" fill="#374151">⊕ 残差相加</text>

    <rect x="90" y="384" width="230" height="26" rx="4" fill="#fff" stroke="#6b7280"/>
    <text x="205" y="401" fill="#374151">RMSNorm₂(x) · γ₂</text>

    <rect x="90" y="422" width="230" height="26" rx="4" fill="#fff" stroke="#6b7280"/>
    <text x="205" y="439" fill="#374151">SiLU(gate(x)) ⊙ up(x)</text>
    <text x="336" y="430" font-size="10" fill="#b91c1c">↑ 逐元素积：旋转到此为止</text>
    <text x="336" y="445" font-size="10" fill="#b91c1c">  必须在这里现算一次 Hadamard</text>

    <rect x="20" y="462" width="44" height="22" rx="11" fill="#ffedd5" stroke="#ea580c"/>
    <text x="42" y="477" font-size="11" font-weight="700" fill="#ea580c">R3</text>
    <rect x="90" y="460" width="230" height="26" rx="4" fill="#fff" stroke="#ea580c" stroke-width="1.5"/>
    <text x="205" y="477" fill="#ea580c">fwht(h) → down_proj</text>
    <text x="336" y="469" font-size="10" fill="#ea580c">可融进 SiLU⊙up 的 epilogue，</text>
    <text x="336" y="484" font-size="10" fill="#ea580c">每层 1 次；d=11008 非 2 幂需补齐</text>
  </g>

  <g stroke="#9ca3af" stroke-width="1.2">
    <path d="M205,88 L205,106"/><path d="M205,132 L205,144"/>
    <path d="M205,170 L205,184"/><path d="M205,210 L205,222"/>
    <path d="M205,248 L205,262"/><path d="M205,288 L205,300"/>
    <path d="M205,326 L205,340"/><path d="M205,366 L205,384"/>
    <path d="M205,410 L205,422"/><path d="M205,448 L205,460"/>
  </g>
</svg>
<p class="cap">图 2：四个插入位置。蓝色 = 可离线折叠进权重（运行时零成本），橙色 = 必须在线计算。R3 与 R4 是后两个，也是真正的成本所在。</p>
</div>

### 2.1 R1：残差流——折叠进所有权重

这是最重要的一个旋转，因为它一次性解决了 `q/k/v/gate/up` 五个投影层输入的 outlier。

**残差流一旦被旋转，就必须"保持旋转"**——从 embedding 到最后一层的 `lm_head`，整条残差通路都活在旋转基下。所以不是转一次，而是把所有相关权重都改一遍：

```python
# 入口侧（吃残差流的层）：y = x @ W.T，其中 x 已按 x → x·R 旋转
# γ 夹在中间不能直接穿过旋转，要把它一起折进去
Dg, Dgi = torch.diag(gamma), torch.diag(1.0 / gamma)
W_in_new = W_in @ Dg @ R @ Dgi        # [out, in]，注意乘序

# 出口侧（写残差流的层）：要求 y' = y·R  ⇒  W_out_new = Rᵀ @ W_out
W_out_new = R.T @ W_out

# 边界：embedding 入口和 lm_head 出口
E_new    = E    @ R                   # [vocab, d]
W_lm_new = R.T  @ W_lm                # [vocab, d]，lm_head 一般不量化但也要转
```

<div class="warnbox">
<strong>最容易被讲错的一步：</strong>很多人以为"RMSNorm 对旋转不变，所以 γ 也不变"。前半句对（<code>||Rx||₂ = ||x||₂</code>，RMS 只看范数），后半句错——<code>γ</code> 是逐元素乘，<code>(Rx) ⊙ γ ≠ R(x ⊙ γ)</code>。正确做法是上面那行 <code>W · diag(γ) · R · diag(1/γ)</code>：把 γ 的逆拉回输入侧、把 γ 和 R 一起折进权重。少写一个 diag，模型直接崩，而症状是"精度莫名其妙掉一大截"，极难 debug。
</div>

### 2.2 R2：注意力输出——折进 o_proj

注意力输出 `PV` 也有 outlier（某些 head 的某些通道特别大）。旋转每个 head 的 `head_dim`（= 128，正好是 2 的幂）：

```text
P·(V R) = (P V) R     ← P 左乘，R 右乘，作用在不同维度上，可交换
```

`R` 穿过 `P` 之后被 `o_proj` 吸收：

```text
(PV R) W_o = (PV) (R W_o)   ⇒   W_o' = R · W_o   ← 按 head 分块对角
```

注意这里 `R` 是**分块对角**的：`blockdiag(R_hd, R_hd, ..., R_hd)`，共 `n_heads` 块。因为 `o_proj` 的输入维是 `n_heads × head_dim`，每个 head 独立旋转。

### 2.3 R3：SwiGLU 之后——唯一的在线旋转

`down_proj` 的输入 `h = SiLU(gate(x)) ⊙ up(x)` 是所有中间激活里 outlier 最严重的（因为 SiLU 的负半轴被截断、正半轴无上界，天然产生偏斜）。

但 `⊙` 是逐元素积：

```text
(R g) ⊙ (R u)  ≠  R (g ⊙ u)
```

**旋转不能穿过它。** 所以必须在 `h` 上现算一次 Hadamard：

```python
h = F.silu(gate_out) * up_out     # [B, T, 11008]
h = fwht(h)                        # 在线旋转，唯一无法折叠的一处
y = h @ down_proj.T
```

> **一个常见误解**：能不能让 `gate` 和 `up` 的输出各自先旋转一次，省掉 `h` 上的 FWHT？不行。`(R g) ⊙ (R u) ≠ R (g ⊙ u)`，逐元素积这一步永远阻断交换。所以哪怕 `gate`/`up` 的输出已经转到某个基下，`h` 上的那一次旋转**无法避免**。这也解释了为什么 QuaRot 的运行时开销几乎全压在 R3 上。

两个工程细节：

1. **`d = 11008` 不是 2 的幂**（LLaMA-2 的 intermediate size）。QuaRot 的做法是 **pad 到 16384**（2¹⁴），多出来的通道填 0，旋转完再截断。padding 让这一步的计算量涨 49%，所以有些实现改用手工构造的 11008×11008 近似 Hadamard，或用 128×128 块对角 + 跨块置换（这就是 DuQuant 的思路，见 3.3）。
2. **FWHT 可以融进 `up_proj` 的 epilogue**：`h` 算完立刻在 shared memory 里做完蝶形，不需要落回 HBM 再读一遍。这一步融合做不做，决定了额外开销是 1% 还是 10%。

### 2.4 R4：Q/K 旋转——白送的

这一处非常优雅。RoPE 之后对 `Q` 和 `K` 施加**同一个** `R`（作用在 `head_dim` 上）：

```text
scores = Q R · (K R)ᵀ = Q R Rᵀ Kᵀ = Q Kᵀ
```

分数**逐比特不变**（在精确算术下）。没有补偿项，没有额外权重改动，`V` 也不受影响。

它的收益是：`Q` 和 `K` 的激活幅值被打散，于是 **KV Cache 可以更激进地量化**——这正是 qt5 要展开的内容。R4 是唯一的"免费午餐"，QuaRot 论文里 W4A4KV4 的 KV4 部分就靠它。

## 3. 三代方案：QuaRot → SpinQuant → DuQuant

同样一个 `R`，怎么选是核心问题。三代方案的演进，本质上是**对 `R` 的搜索空间逐步放开**。

| 代 | 方案 | `R` 怎么来 | 训练需求 | 关键洞察 |
|---|---|---|---|---|
| 1 | **QuaRot**（2024.04） | 手工指定：Hadamard（+ 随机置换） | 无（纯 PTQ） | 证明"随机旋转就够"，把 W4A4 从不可行变可行 |
| 2 | **SpinQuant**（2024.05） | Cayley 优化在流形上学 | 需校准集上训 `R`（几小时） | 旋转的**选择**显著影响精度，学出来的比手挑好 |
| 3 | **DuQuant**（2024.10） | 块对角旋转 + zigzag 通道置换 | 无（纯 PTQ） | 全局旋转会造出**新 outlier**；要按块局部处理 |

### 3.1 QuaRot：证明随机旋转就够

QuaRot 的核心贡献不是"发明旋转"（这个想法在量化圈早有），而是**把四个插入点全部找齐并做对了融合**，从而第一次让 LLaMA-2-7B 的 W4A4（含 KV4）在 PTQ 下接近 FP16。

它的 `R` 选择非常朴素：

```text
R1 = 随机置换 P₁ · Hadamard H        （残差流，d = 4096）
R2 = 每个 head 一个 H                （head_dim = 128）
R3 = Hadamard H                      （intermediate，pad 到 16384）
R4 = 每个 head 一个 H                （Q/K，head_dim = 128）
```

随机置换 `P` 的作用：单独用 Hadamard 时，`H` 的行有固定结构，某些输入模式可能被映射到"恰好都同号"从而叠加成新的尖峰。加一个随机置换打破这个规律性（相当于把 outlier 位置随机化）。**这是一个很实用的小技巧**：`x → permute(x) → H → ...`。

QuaRot 的实测（LLaMA-2-7B，wiki-text PPL）：

```text
FP16 基线                5.47
RTN W4A4（无旋转）        ~2000   ← 完全崩掉
QuaRot W4A4              6.10    ← 只差 0.63
QuaRot W4A4 + KV4        6.19
```

对比一下 qt3 里 AWQ 的 W4A16 是 ~5.6——**QuaRot 用一半的位宽，只多付 0.5 的 PPL**。这就是旋转路线存在的理由。

### 3.2 SpinQuant：旋转的"选择"比"存在"更重要

SpinQuant 的出发点是：QuaRot 用的是**固定**的 Hadamard，但 `R` 的选取自由度极高（`d×d` 正交群），不同的 `R` 精度能差很多。

它的做法是把 `R` 参数化为**可学习**的，但必须始终留在正交流形上——直接用梯度下降会跑出正交约束。解法是 **Cayley 变换**：

```text
R = (I − A)(I + A)⁻¹ ,  A 反对称（Aᵀ = −A）
```

对任意反对称 `A`，这个 `R` **恒为正交**，于是梯度下降可以在无约束的 `A` 空间里做，`R` 自动合法。

```python
class CayleyRotation(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(d, d))   # 只学上三角

    def forward(self):
        A = self.A.triu(1) - self.A.triu(1).T            # 强行反对称
        I = torch.eye(A.shape[0], device=A.device)
        return torch.linalg.solve(I + A, I - A)          # (I+A)⁻¹(I−A)
```

训练目标不是"最小化量化误差"（不可微），而是**最小化量化前后模型输出的差异**——在 WikiText 校准集上跑几百步，只学 `R`，权重冻结。

SpinQuant 的结论很有价值：**学出来的 `R` 与 Hadamard 结构相似但不同**，在 LLaMA-2-7B W4A4 上把 PPL 从 6.10 降到 5.86（QuaRot 是 6.10），LLaMA-3-8B 上差距更明显。代价是需要一次训练（单卡 A100 数小时），且每个模型要单独训一套。

### 3.3 DuQuant：全局旋转会造出新的 outlier

DuQuant 指出了一个被前两代忽略的问题：**Hadamard 是全局的（跨整个 `d` 维），它会引入新的 outlier。**

原因在于激活分布的**通道间异质性**。假设某段连续通道的幅值普遍很小（"死区"），而另一段很大。全局 Hadamard 会把大通道的能量均摊到这些小通道上——这对小通道来说是**凭空出现的巨大值**。原本安静的区域被污染了。

DuQuant 的两个对策：

```text
① 块对角旋转：只在 128×128（或 64×64）的块内旋转，噪声不跨块传播
   代价：块内的 outlier 只摊到 128 个通道，压缩比只有 √128，不如全局 √d
② zigzag 置换：按激活幅值排序后「蛇形」重排通道
   让每个块内都包含「大-小-大-小」的混合，而不是「一块全大、一块全小」
   这样块内动态范围被拉平，块对角旋转才有效
```

`zigzag` 这个名字来自重排的路径：把按幅值排序的通道按 `[0, 1, 2, 3]` → 分到块 `[0,3]`、`[1,2]` 的蛇形分配，使每块都拿到"一强一弱"的搭配。这是**把全局信息压进局部块**的技巧，和稀疏矩阵的带宽优化是同一类思路。

DuQuant 的收益在**小模型和低比特**上最明显：LLaMA-2-7B W4A4 做到 5.98，而 LLaMA-3-8B、Qwen2 这类 attention 里 outlier 更顽固的模型，DuQuant 比 QuaRot 稳定得多（QuaRot 在某些层会直接崩）。

<div class="keybox">
<strong>三代方案的一句话总结：</strong><br>
QuaRot 回答「<strong>能不能</strong>用旋转做 W4A4」——能，随机 Hadamard 就够；<br>
SpinQuant 回答「<strong>哪个</strong>旋转最好」——学出来比手挑好，代价是一次训练；<br>
DuQuant 回答「全局旋转<strong>有什么副作用</strong>」——会造新 outlier，得按块 + 置换。
</div>

## 4. 为什么 W4A4 值得这么折腾

回到工程现实：W4A16 已经把权重压到 4 bit，为什么还要冒风险去量化激活？

因为 **W4A16 的收益只落在 decode 阶段**。qt1 里讲过：decode 是带宽瓶颈（每生成一个 token 要读完整个权重），所以权重位宽直接换算成加速比。但 prefill 是**算力瓶颈**——它要处理几千个 token，GEMM 是大矩阵乘大矩阵，此时激活不是瓶颈，权重读一次就摊薄了。

```text
                       权重带宽      激活带宽      GEMM 类型
decode（batch 1）      ★★★ 瓶颈     很小           GEMV
prefill（长 prompt）   摊薄           ★★ 相关        GEMM
```

量化激活（A4）的价值在于：

1. **prefill 的 Tensor Core 吞吐**：INT4×INT4 的 MMA 在同等算力下能跑出 2× 于 INT8、4× 于 FP16 的理论峰值（取决于硬件是否原生支持 int4 mma）。
2. **长上下文场景**：prefill 占比可达总延迟的 70% 以上，此时 A4 的收益才显现。
3. **显存带宽的二次收益**：虽然激活不常驻，但大 batch 的 KV Cache 和中间激活占用不可忽视。

但现实是残酷的：**目前主流 GPU（H100/H200/B200）的 Tensor Core 没有原生的 INT4×INT4 MMA**。INT4 只能走"解包 + 转 INT8/FP16 再算"，这时 W4A4 相对 W4A16 几乎没加速，反而多付了量化误差。所以：

<div class="warnbox">
<strong>W4A4 今天的真实处境：</strong>在通用 GPU 上，W4A4 是「<strong>省显存</strong>」而不是「<strong>提速度</strong>」——它能让你在单卡上跑更大的 batch 或更长的上下文，但 kernel 层面的吞吐优势要等硬件支持原生低比特 MMA。真正吃到 W4A4 红利的是<strong>定制加速器</strong>（部分国产 NPU、Groq/LPUs 类架构）和<strong>端侧芯片</strong>（高通 Hexagon、苹果 ANE 对 int4 有原生通路）。这也是为什么工业界目前的主流仍是 W4A16（AWQ/GPTQ）和 W8A8（FP8）。
</div>

## 5. 落地清单

如果你要真的用旋转量化，这是实操顺序：

```text
① 选基线：先跑 FP16 PPL，再跑 RTN W4A4 —— 如果 RTN 没崩，说明这个模型不需要旋转
② 挑方案：
   - 只想快速验证 → QuaRot（pip install quarot，一条命令改权重）
   - 追求最好精度 + 有训练预算 → SpinQuant
   - 模型小 / outlier 顽固（LLaMA-3、Qwen2）→ DuQuant
③ 确认融合是否做全：R1 是否折进所有 in/out 投影？R2 是否按 head 分块？
   R3 的 padding 是否处理了非 2 幂维？漏一个，精度就掉
④ 测三件事：PPL（平均）、最坏层误差（逐步 dump 每层输出对比）、
   长上下文（32k 以上最先崩，必须单独测）
⑤ 测速度：如果 kernel 不支持原生 int4 mma，W4A4 可能比 W4A16 还慢
   —— 务必在目标硬件上实测，不要看论文数字
```

一个实操经验：**旋转路线的调试成本远高于 AWQ/GPTQ**。因为出错时模型不会崩成乱码，而是"PPL 从 5.5 涨到 6.3"——你很难判断是旋转没折对、还是校准集不匹配、还是某一层融合漏了。建议的排查方式是**逐层 dump 输出**，和 FP16 对比每一层的相对误差，定位到第一个异常层。

## 6. 系列导航

- 上一篇 **qt3 · SmoothQuant 与 AWQ**：B 路线，改分布（缩放），上限 W4A16。
- 下一篇 **qt5 · KV Cache 量化与 FP4**：另一条正交的压缩维度，以及 E4M3 / NVFP4 这些新格式怎么改变游戏规则。
- **qt2 · RTN 与 GPTQ**：A 路线，只改舍入，是下面所有方法的基线。
- **qt1 · 量化的基本语言**：粒度、对称性、校准、四条路线地图。

站内相关：**op3（昇腾 EagleQBSA）** 里"Q/K 用 INT8 per-head、P 跟随 V 用 FP8"的选择，正是 R4 这一处旋转收益的实证场景；**fw4** 从引擎视角讲了量化 kernel 的融合层次；**op2** 是系列导读与谱系总表。
