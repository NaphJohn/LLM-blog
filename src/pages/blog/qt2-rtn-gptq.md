---
title: '量化手记（二）：RTN 与 GPTQ——从「直接取整」到「用 Hessian 补偿误差」'
description: 'RTN 是量化最朴素的形式：逐元素 round，零成本但会被大激活通道摧毁。GPTQ 的全部创新只有一句话——量化某一列时，用 Hessian 的逆把误差按重要性摊到还没量化的列上，让最终输出误差最小。本篇完整推导 OBS 的误差公式、逐列分块算法、Cholesky 与 Laplace 两种求解路径，附可跑伪代码、group_size 与 damp 的调参逻辑，以及为什么 GPTQ 在 decode 上收益最大。'
pubDate: 2026-10-27
series: 量化手记
lang: zh
altLang: en
altHref: /en/blog/qt2-rtn-gptq
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

RTN 和 GPTQ 用的是**同一个量化器、同一个 4-bit 网格**，唯一区别是**舍入时看不看上下文**。RTN 逐元素独立取整，谁也不知道自己给邻居带来了多少误差；GPTQ 一次量化一列，然后立刻用 **Hessian 的逆**把这一列产生的误差，按"层输出对哪个输入更敏感"重新摊到剩下还没量化的权重上。就这一个改动，让 LLaMA 这类模型在 4-bit 下从"明显掉点"变成"接近无损"。

<div class="keybox">
<strong>本篇位置：</strong>对应 qt1 图 2 的 A 路线——只改舍入，不动数值分布。之所以把 RTN 和 GPTQ 放在同一篇，是因为理解了 RTN 的失败模式，才看得懂 GPTQ 每一步在补什么。下一篇 qt3 会跳出这条路线，改从"分布"入手。
</div>

## 1. RTN：量化的起点，也是理解一切量化方法的参照系

### 1.1 算法本身只有三行

```python
import torch

def rtn_quantize(W: torch.Tensor, bits: int = 4, group_size: int = 128):
    """RTN（Round-To-Nearest）：逐元素独立取整，最朴素的 PTQ。"""
    W = W.clone()
    out_f = W.shape[0]
    # 按输出通道分组：每 group_size 个连续输入维度共享一个 scale
    Wg = W.reshape(out_f, -1)                    # [out, in]
    if group_size > 0:
        Wg = Wg.reshape(out_f, -1, group_size)   # [out, in/group, group]

    # 对称量化：scale 由该组内的绝对值最大值决定
    scale = Wg.abs().amax(dim=-1, keepdim=True) / (2 ** (bits - 1) - 1)

    # 核心一行：取整 + 饱和
    Wq = torch.clamp(torch.round(Wg / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)

    # 反量化回浮点（实际部署时这一步融进 GEMM，见第 5 节）
    W_deq = (Wq * scale).reshape(out_f, -1)
    return W_deq, Wq.to(torch.int8), scale
```

没有校准数据、没有梯度、没有 Hessian。读一遍权重、算一次 absmax、round、完事。**速度是它的唯一优势**：7B 模型的 4-bit RTN 在单卡上几十秒就能跑完。

### 1.2 为什么 RTN 会失败：误差没有"责任分配"

关键在于：RTN 假设**每个权重的量化误差是独立且等价的**。但真实情况是，层输出是输入的加权和：

```text
Y = X W     # X: [T, in],  W: [in, out]
```

假设 `W` 的第 `i` 列被量化成 `Ŵ_i`，产生的误差是 `ΔW_i = W_i - Ŵ_i`。它对输出的贡献是：

```text
ΔY = X ΔW        →   ΔY[:, j] = Σ_i X[:, i] · ΔW[i, j]
```

统计学上，第 `i` 个输入维度对第 `j` 个输出误差的贡献强度，正比于 `X[:, i]` 的**能量**（二阶矩）。如果某些输入通道的激活特别大——LLM 里这是常态，见下图——那么这些通道上的权重误差会被放大几十倍，而 RTN 对此一无所知，照样给它一个和其他通道同样"努力"的舍入。

<div class="fig">
<svg viewBox="0 0 680 290" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="RTN 与 GPTQ 的误差分配差别">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">同一个舍入决策，两种后果：RTN 不知道谁重要，GPTQ 知道</text>

  <!-- 左：RTN -->
  <text x="20" y="54" font-size="13" font-weight="700" fill="#b91c1c">RTN：每一列独立取整</text>

  <!-- 输入激活能量条 -->
  <text x="20" y="80" font-size="11" fill="#6b7280">输入激活能量 |X[:,i]|²</text>
  <rect x="20" y="86" width="26" height="14" fill="#dbeafe" stroke="#93c5fd"/>
  <rect x="46" y="86" width="26" height="14" fill="#dbeafe" stroke="#93c5fd"/>
  <rect x="72" y="86" width="26" height="14" fill="#dbeafe" stroke="#93c5fd"/>
  <rect x="98" y="86" width="26" height="14" fill="#1e40af" stroke="#1e40af"/>
  <text x="111" y="97" font-size="9" fill="#ffffff" text-anchor="middle">↑</text>
  <rect x="124" y="86" width="26" height="14" fill="#dbeafe" stroke="#93c5fd"/>
  <rect x="150" y="86" width="26" height="14" fill="#dbeafe" stroke="#93c5fd"/>
  <text x="98" y="118" font-size="10" fill="#b91c1c">outlier 通道（能量大 50×）</text>
  <text x="195" y="97" font-size="11" fill="#374151">← 缩放极小、数量极少，但主导误差</text>

  <!-- 权重行 + 误差 -->
  <g>
    <text x="20" y="146" font-size="11" fill="#6b7280">权重 Ŵ[i, :]</text>
    <rect x="20" y="152" width="26" height="20" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="46" y="152" width="26" height="20" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="72" y="152" width="26" height="20" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="98" y="152" width="26" height="20" fill="#fecaca" stroke="#ef4444" stroke-width="2"/>
    <rect x="124" y="152" width="26" height="20" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="150" y="152" width="26" height="20" fill="#f3f4f6" stroke="#d1d5db"/>
  </g>
  <text x="195" y="167" font-size="11" fill="#374151">← 照常 round，误差被 50× 放大</text>

  <!-- 输出误差柱 -->
  <text x="20" y="196" font-size="11" fill="#6b7280">输出误差 ΔY</text>
  <rect x="20" y="203" width="156" height="18" fill="#fecaca" stroke="#ef4444"/>
  <text x="98" y="216" font-size="10" fill="#b91c1c" text-anchor="middle">大</text>

  <!-- 分隔 -->
  <line x1="348" y1="44" x2="348" y2="270" stroke="#e5e7eb" stroke-width="1.5"/>

  <!-- 右：GPTQ -->
  <text x="368" y="54" font-size="13" font-weight="700" fill="#047857">GPTQ：量化一列，补偿剩余</text>

  <text x="368" y="80" font-size="11" fill="#6b7280">输入激活能量 |X[:,i]|²</text>
  <rect x="368" y="86" width="26" height="14" fill="#dbeafe" stroke="#93c5fd"/>
  <rect x="394" y="86" width="26" height="14" fill="#dbeafe" stroke="#93c5fd"/>
  <rect x="420" y="86" width="26" height="14" fill="#dbeafe" stroke="#93c5fd"/>
  <rect x="446" y="86" width="26" height="14" fill="#1e40af" stroke="#1e40af"/>
  <text x="459" y="97" font-size="9" fill="#ffffff" text-anchor="middle">↑</text>
  <rect x="472" y="86" width="26" height="14" fill="#dbeafe" stroke="#93c5fd"/>
  <rect x="498" y="86" width="26" height="14" fill="#dbeafe" stroke="#93c5fd"/>
  <text x="368" y="118" font-size="10" fill="#047857">同样存在，但被 H⁻¹ 计入权重</text>

  <text x="368" y="146" font-size="11" fill="#6b7280">权重 Ŵ[i, :]</text>
  <rect x="368" y="152" width="26" height="20" fill="#f3f4f6" stroke="#d1d5db"/>
  <rect x="394" y="152" width="26" height="20" fill="#d1fae5" stroke="#10b981"/>
  <rect x="420" y="152" width="26" height="20" fill="#d1fae5" stroke="#10b981"/>
  <rect x="446" y="152" width="26" height="20" fill="#fecaca" stroke="#ef4444" stroke-width="2"/>
  <rect x="472" y="152" width="26" height="20" fill="#d1fae5" stroke="#10b981"/>
  <rect x="498" y="152" width="26" height="20" fill="#d1fae5" stroke="#10b981"/>
  <text x="540" y="167" font-size="11" fill="#047857">← 后续列被微调，抵消误差</text>

  <text x="368" y="196" font-size="11" fill="#6b7280">输出误差 ΔY</text>
  <rect x="368" y="203" width="30" height="18" fill="#d1fae5" stroke="#10b981"/>
  <text x="383" y="216" font-size="10" fill="#047857" text-anchor="middle">小</text>

  <text x="20" y="258" font-size="12" fill="#6b7280">两者权重位宽相同、压缩率相同。差别只在：GPTQ 把「谁被量化误差影响更大」写进了更新公式。</text>
</svg>
<p class="cap">图 1：RTN 把每一列的舍入当作独立事件；GPTQ 把整层看成一个整体优化问题——先量化重要的维度，再用剩余维度的自由度去补偿。绿色列是 GPTQ 主动"改写"的补偿项，它们本身也还要被量化，但已经带着修正量了。</p>
</div>

一段能看出差距的最小实验：

```python
import torch

torch.manual_seed(0)
T, IN, OUT = 512, 4096, 4096

# 模拟 LLM 的激活：一个通道的幅值远大于其他（outlier channel）
X = torch.randn(T, IN) * 0.1
X[:, 7] *= 80.0                      # 制造一个 outlier 通道

W = torch.randn(IN, OUT) * 0.02
Y_ref = X @ W

Wq_rtn, _, _ = rtn_quantize(W, bits=4, group_size=128)
err_rtn = ((X @ Wq_rtn - Y_ref).norm() / Y_ref.norm()).item()

print(f"RTN  相对输出误差: {err_rtn:.4f}")   # 典型 0.05 ~ 0.15，视 outlier 强度
```

结论很稳定：**outlier 越强，RTN 的相对误差越接近线性增长**。而 GPTQ 在同样条件下能把这个数字压下一个量级。

## 2. GPTQ 的理论起点：OBS 框架

GPTQ 不是凭空发明的，它站在 **OBS（Optimal Brain Surgeon，1993）** 这条三十年的修剪（pruning）理论线上。理解 OBS，GPTQ 的公式就是自然推出来的。

### 2.1 问题设定

我们想量化一层权重 `W`，目标是最小化**层输出的重建误差**：

```text
min_Ŵ  || X W − X Ŵ ||²₂
```

注意这里最小化的不是 `||W − Ŵ||²`（那是 RTN 隐含在做的，逐元素最近），而是**经过输入加权后的**误差。这个区别是 GPTQ 全部威力的来源。

把 `W` 按列拆开（每列对应一个输出维度，它们互相独立），问题变成对每一列 `w` 独立求解：

```text
min_ŵ  || X w − X ŵ ||²  =  min_ŵ  (w − ŵ)ᵀ H (w − ŵ)
其中  H = Xᵀ X        # [in, in] 的 Hessian（二阶导），对称半正定
```

所以 **`H` 就是"输入通道重要性矩阵"**：对角线 `H_ii = ||X[:, i]||²` 是那个通道的能量，非对角项记录了通道之间的相关性。

### 2.2 OBS 的核心一步：误差如何在剩余维度上重新分配

现在考虑：我们决定把第 `q` 个权重固定为量化值 `ŵ_q`（不是最近值，而是某个确定的值）。误差是多少，剩余维度怎么补偿？

把误差 `δ = w − ŵ` 限制为只在第 `q` 维非零（其他维度暂时保持原值）：

```text
由最优性条件：  H δ = 0  仅对未被固定的维度成立
```

用拉格朗日或直接构造：假设只有 `δ_q ≠ 0`，其余维度需要调整 `Δ` 来消除这个扰动带来的梯度：

```text
H (δ + Δ) = 0
→ Δ = −H⁻¹ · (H δ) 的剩余部分
```

推导到具体形式，经典结论是：

**若把第 `q` 个权重钉在 `ŵ_q`，则误差为：**

```text
err_q = (w_q − ŵ_q)² / [H⁻¹]_qq
```

**且其余所有未固定的权重需要这样更新（这就是"补偿"）：**

```text
Δw_{:, q} 之外的维度：

w ← w − (w_q − ŵ_q) / [H⁻¹]_qq · H⁻¹[:, q]
```

这两个式子值得逐字读一遍：

- 第一式说：**误差大小被 `H⁻¹` 的对角元缩放**。`[H⁻¹]_qq` 大（这一维"冗余度高"、量化它代价小）→ 分母大 → 误差小。
- 第二式说：**补偿的方向由 `H⁻¹` 的第 `q` 列给出**。这个列向量告诉你"如果第 q 维偏了 1，应该让其他维度各自偏多少来抵消"。

<div class="keybox">
<strong>这里有个反直觉的地方值得停一下。</strong> 直觉上我们想"保护重要的通道"——`H_qq` 大的不要动。但正确逻辑恰好相反：<strong>那些 <code>H_qq</code> 大、但 <code>[H⁻¹]_qq</code> 小（即与其他通道高度相关、信息冗余）的维度，才是最适合被"钉住量化"的</strong>；而真正独特的维度必须保持高精度。这就是为什么所有 GPTQ 实现都按"`[H⁻¹]_qq` 从大到小"的顺序处理列——先钉住最容易补偿的那些。
</div>

### 2.3 为什么"顺序"是个真实的设计决策

OBS 原始算法对每一列都要求一次 `H⁻¹` 的第 `q` 列，代价 `O(in²)`，一层下来 `O(in³)`——对 4096 维的层是 680 亿次运算，太慢，这是 OBS 长期无法用于大模型的直接原因。

GPTQ 的两个工程创新：

**(1) 一次性的 Cholesky 分解代替逐列的矩阵求逆。** 因为 `H` 对称正定，可以预先做 `H⁻¹ = L Lᵀ`（Cholesky），或者更常用的是**对 `H` 做 Cholesky 后再前向/后向替换**拿到需要的列。这一步是 `O(in³/3)`，但只做一次，摊到所有列上后总复杂度降到 `O(in³/3 + in²·out)`。

**(2) 惰性批量更新（lazy batch update）。** 关键洞察：如果把列按固定大小分块（block，如 128 列），块内的补偿可以**先不做**，只记录下需要修正的量，等整块量化完再一次性应用 `H⁻¹` 的块。这样能把 `H⁻¹` 的访存和 GEMM 都组织成对齐的大矩阵乘，显存和速度都大幅改善。

## 3. GPTQ 的完整算法

下面是 GPTQ 论文 Algorithm 1 的可跑版本，附逐步注释。**读这段代码时请对照 2.2 的两个公式**——能对上，GPTQ 就真正懂了。

```python
import torch

def gptq_quantize_layer(W, X, bits=4, group_size=128, damp=0.01, blocksize=128):
    """
    GPTQ：逐层量化，用 Hessian 的逆把误差补偿到剩余列。
    W: [in, out]  一层的权重（不含 bias）
    X: [T, in]    校准数据在该层的输入激活
    """
    dev = W.device
    in_f, out_f = W.shape
    W = W.clone().float()

    # ---------- Step 1: 求 Hessian H = 2 Xᵀ X，并加阻尼 ----------
    H = 2 * (X.T @ X)                                  # [in, in]
    # 阻尼（damping）：对角加一个小量，保证 H 可逆且数值稳定
    dead = torch.diag(H) == 0                          # 零激活的死通道
    H[dead, dead] = 1.0
    H += damp * torch.eye(in_f, device=dev) * torch.diag(H).mean()

    # ---------- Step 2: Cholesky 分解求 H⁻¹ ----------
    # H⁻¹ 的上三角 Cholesky 因子：后面用它做逐列的补偿
    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    H_inv = torch.linalg.cholesky(H_inv, upper=True)   # 上三角

    # ---------- Step 3: 量化参数（per-group scale）----------
    gs = group_size if group_size > 0 else in_f
    Wg = W.reshape(in_f, -1, gs) if group_size > 0 else W.reshape(in_f, -1, in_f)
    maxq = 2 ** (bits - 1) - 1
    scale = Wg.abs().amax(dim=0, keepdim=True) / maxq   # 注意：按输入维分组
    scale = scale.clamp(min=1e-9).reshape(-1, 1) if group_size > 0 else scale.reshape(1, -1)

    Wq = W.clone()                                     # 逐列写入量化结果
    W_deq = torch.zeros_like(W)                        # 反量化后的权重（部署用）

    # ---------- Step 4: 逐块、逐列量化 ----------
    for blk_start in range(0, out_f, blocksize):
        blk_end = min(blk_start + blocksize, out_f)

        Err = torch.zeros(in_f, blk_end - blk_start, device=dev)  # 块内误差累积器

        for j in range(blk_start, blk_end):
            w = Wq[:, j]                               # 当前列（已被前面步骤修正过）
            d = H_inv[j, j]                            # [H⁻¹]_jj —— 该列的"量化代价"

            # ① 量化这一列
            s = scale.reshape(-1, 1)[j] if group_size > 0 else scale[0, j]
            q = torch.clamp(torch.round(w / s), -maxq - 1, maxq)
            wq = q * s

            # ② 记录这一列的反量化误差（用于部署）
            W_deq[:, j] = wq

            # ③ 误差 = (原值 − 量化值) / [H⁻¹]_jj，这就是 OBS 的补偿量
            err = (w - wq) / d

            # ④ 把补偿量写入块内累积器（懒更新，先不碰 Wq）
            Err[:, j - blk_start] = err

            # ⑤ 块内剩余列立即补偿（这一步是精度关键）
            Wq[:, j + 1 : blk_end] -= err.unsqueeze(1) * H_inv[j, j + 1 : blk_end].unsqueeze(0)

        # ---------- Step 5: 整块一次性应用剩余补偿 ----------
        if blk_end - blk_start >= 4:
            Wq[:, blk_end:] -= Err @ H_inv[blk_start:blk_end, blk_end:]

    return W_deq
```

### 3.1 三个参数怎么调

| 参数 | 默认 | 作用 | 调大的后果 | 调小的后果 |
|---|---|---|---|---|
| `group_size` | 128 | scale 共享粒度 | 精度差（同见 qt1） | 精度好但 scale 开销涨 |
| `damp` | 0.01 | Hessian 对角阻尼 | 补偿更保守、更稳，但精度损失变多 | 数值可能崩（`H⁻¹` 病态） |
| `blocksize` | 128 | 惰性更新的块大小 | 速度略降、精度略好 | 速度更快、精度略降 |

`damp` 是最需要经验的一个。它乘以 `diag(H).mean()` 再加到对角上，本质是 **Tikhonov 正则化**：当 `H` 接近奇异（层里有近共线的输入通道，Transformer 里很常见）时，`H⁻¹` 会出现巨大数值导致补偿量爆炸、权重被改得面目全非，最终反量化结果一团糟。判据：**量化完检查 `W_deq − W` 的最大绝对值**，如果出现 >1 的异常值，`damp` 加一个量级。

### 3.2 `[H⁻¹]_qq` 的作用：为什么它决定了"谁挨刀"

这个量在 GPTQ 里有名字叫 **Inverse Hessian Diagonal (ID)**，是判断"某列能不能便宜地量化"的核心指标。它和 `H_qq`（激活能量）的关系很有意思：

- 若某通道**独立且激活小**：`H_qq` 小，`[H⁻¹]_qq ≈ 1/H_qq` **大** → 量化它"便宜"（误差分母大）。
- 若某通道**是 outlier 但和其他通道冗余**：`H_qq` 巨大，但因为可以用其他通道重建，`[H⁻¹]_qq` 反而不大 → 也适合量化。
- 若某通道**独特且激活大**：`H_qq` 大且 `[H⁻¹]_qq` 小 → **绝不能量化**。

这就是 GPTQ 能在 W4A16 下逼近无损的物理原因：它是**outlier-aware 的，但方式不是"保护 outlier"，而是"利用冗余"**。

<div class="figure-warn">
<div class="warnbox">
<strong>GPTQ 的一个真实局限，选型时必须知道：</strong>它最小化的是<strong>单层输出的 L2 误差</strong>，不是端到端任务损失。所以：<br>
① 层与层之间存在误差累积，深度模型上"每层误差最小"不等于"整网误差最小"——这就是为什么 GPTQ 在高压缩比（3-bit 以下）会明显劣化，而 AWQ / 旋转类方法相对更稳。<br>
② 校准集分布必须匹配目标任务。用 wiki 校准的 GPTQ 模型去跑代码，掉点会比 PPL 上看到的大得多。<br>
③ 它假设 `<code>X</code>` 在校准集上能代表真实推理分布。长上下文场景下这个假设会破——校准通常只在 2K 长度上做，而部署可能是 128K。
</div>
</div>

## 4. 从 GPTQ 长出来的变种

GPTQ 之后有一整片变种。挑三个值得知道的：

**GPTQ + act-order（desc_act）。** 不按权重矩阵的原始列序处理，而按 `[H⁻¹]_qq` 从大到小排序后再量化。这样"容易量化的先钉住"，剩下的补偿空间更大。代价是**量化后的权重列序被打乱，部署时必须额外存一个 permutation index，并在 kernel 里做 gather**——这就是很多引擎里 `desc_act=True` 会明显变慢、甚至不支持的原因。

**GPTQ 的 group 内混合精度。** 对敏感层（通常是 `down_proj` 和第一层）保留 8-bit，其余 4-bit。实测能在几乎不增加平均位宽（4.1~4.3 bit）的前提下挽回大部分损失。这也是 AutoGPTQ / GPTQModel 里 `true_sequential` 与逐层 bit 配置的用途。

**GPTQ 与旋转的组合（GPTQ + QuaRot）。** 先用 Hadamard 旋转把 outlier 摊平（qt4），再对旋转后的权重跑 GPTQ。因为分布已经友好，`H` 的条件数变好、`damp` 可以调小、精度更高。这是目前做 W4A4 的常见配方。

## 5. 落地：GPTQ 权重在推理引擎里怎么跑

GPTQ 的量化产物不是"INT4 权重 + 反量化"，而是**打包格式 + scale + zero**。理解格式，才知道为什么不同引擎的推理速度差很多。

**典型打包（auto-gptq / marlin 格式）：**

```text
qweight:  int32[in/8, out]    # 8 个 4-bit 值打包进一个 int32
scales:   fp16[in/group, out] # per-group scale
qzeros:   int32[in/group, out] (非对称时才需要)
g_idx:    int32[in]           # desc_act 时的列重排索引
```

加速的核心是**反量化融进 GEMM 主循环（online dequantization）**：kernel 在从显存读 `qweight` 的同时，在寄存器里做 `w = q * s`，然后立即喂给 Tensor Core 的 FP16 乘法。这样显存带宽只走 4-bit，算力走 FP16——**正好匹配 decode 阶段的带宽瓶颈**。

<div class="keybox">
<strong>为什么 GPTQ 在 decode 上收益最大，在 prefill 上几乎无感？</strong><br>
<strong>decode</strong>：batch 小（甚至 =1），每生成一个 token 要把全部权重从 HBM 读一遍 → <strong>带宽瓶颈</strong>。权重 4-bit 就是访存量 ÷4，端到端接近 3~4× 提升。<br>
<strong>prefill</strong>：一次处理上千 token，矩阵乘的算术强度高 → <strong>算力瓶颈</strong>。权重虽小，但反量化的额外指令（每 8 个值一次 unpack + scale）反而增加开销，FP16 权重可能更快。这就是为什么引擎里常看到 <code>W4A16 在 prefill 阶段切换到 dequant-then-GEMM（先把整层反量化成 FP16 再乘）</code> 的实现，用显存换吞吐。
</div>

实测参考（LLaMA-2-7B，单卡 A100，batch=1 decode）：

| 格式 | 权重显存 | 单步延迟 | 相对 FP16 |
|---|---|---|---|
| FP16 | 13.5 GB | 基准 | 1.0× |
| GPTQ W4A16（group=128，ExLlama kernel） | ~3.9 GB | 明显更低 | **~2.5–3.2×** |
| GPTQ W4A16（朴素 kernel，无反量化融合） | ~3.9 GB | 略低或持平 | ~1.1× |

**结论：GPTQ 的加速不是"4-bit 权重"自动带来的，而是"4-bit 权重 + 反量化融合 kernel"共同带来的。** 选引擎时优先看它有没有 Marlin / ExLlama / fused-moe 这类专用 kernel，而不是只问"支不支持 GPTQ"。

## 6. RTN vs GPTQ：什么时候 RTN 就够了

不是所有场景都值得跑 GPTQ。RTN 的适用面比想象中广：

| 场景 | RTN | GPTQ | 说明 |
|---|---|---|---|
| 大模型（>13B）4-bit | 掉点明显 | ✅ 首选 | 参数越多，误差累积越严重 |
| 小模型（<1B）8-bit | ✅ 够用 | 过杀 | 8-bit 下两者差距很小 |
| W8A8 | ✅ 够用 | 少量收益 | 8-bit 已经不是极端压缩 |
| 无校准数据 / 数据敏感 | ✅ 唯一选项 | ✗ | GPTQ 需要 X，医疗/金融数据可能不能外传 |
| 无 GPU 或时间紧 | ✅ 分钟级 | 需数十分钟 | GPTQ 要跑校准前向 + Cholesky |
| 极致压缩（3-bit / 2-bit） | 崩 | 勉强 | 这时该看 qt4 的旋转类方法 |
| 端侧、需极简 kernel | ✅ 简单 | 需专用 kernel | RTN 的反量化无需查表最简 |

一条实用法则：**位宽 ≥ 8 用 RTN，位宽 = 4 用 GPTQ 或 AWQ，位宽 < 4 换方法。** 

再补一个统计口径的提醒：GPTQ 论文报告的开源模型 4-bit 结果，PPL 涨幅通常在 **0.1~0.3**（LLaMA 系列）。但这不等于"无损"——下游任务的掉点会大于 PPL 暗示的幅度，尤其是长尾知识和多步推理。生产上线前的验收，建议至少跑：目标任务的一个 in-domain 集 + 一个通用的 MMLU 切片 + 长上下文回归。

## 7. 系列导航

- 上一篇 **qt1 · 量化的基本语言**：粒度、对称性、校准，以及四条路线的地图。
- 下一篇 **qt3 · AWQ 与 SmoothQuant**：换一条思路——不改舍入，改数值分布，把 outlier 从难量化的位置搬走。
- **qt4 · QuaRot / SpinQuant / DuQuant**：用旋转做同样的事，但能做到 W4A4。
- **qt5 · KV Cache 量化与 FP4**：另一种维度的压缩。

站内相关：**qt1** 的 per-group 讨论解释了 `group_size=128` 的来历；**op3（昇腾 EagleQBSA）** 里 QK 按块 INT8 的 scale 计算，本质是"per-block 而非 per-group"的同类权衡；**fw4** 从推理引擎视角讨论过低比特量化的工程侧。
