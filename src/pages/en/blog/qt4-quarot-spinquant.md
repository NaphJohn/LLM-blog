---
title: "Quantization Notes (4): QuaRot, SpinQuant, DuQuant — flattening outliers with rotation, the only route to W4A4"
description: "Every method in the first three notes fights outliers: RTN ignores them, GPTQ compensates them, SmoothQuant relocates them, AWQ amplifies around them. All of them cap out at W4A16, because none of them truly removes an outlier. Rotation does something different: insert an orthogonal matrix R so that X·W = (XR)(RᵀW), the model output is mathematically identical, but a spike of 118 becomes 128 values of 10. This post covers why the trick is free, why it can only be inserted at four places (R1–R4), why SwiGLU's elementwise product forces R3 to be online, why the Q/K rotation is genuinely free, then compares QuaRot (hand-picked Hadamard), SpinQuant (Cayley-parameterised learning), and DuQuant (block-diagonal plus zigzag permutation). It closes with the honest question: can W4A4 be deployed today?"
pubDate: 2026-10-27
series: Quantization Notes
lang: en
altLang: zh
altHref: /blog/qt4-quarot-spinquant
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

The first three notes all work *against* outliers. Their common ceiling is **W4A16** — activations untouched, so activation outliers never become a problem. But the moment you want to quantize activations (W4A8, W4A4), outliers are unavoidable.

Rotation takes a completely different angle:

```text
Do not move the outlier. Flatten it across every channel.

x    = [118, 2, 1, 0, ...]        ← one channel holds all the energy
x·R  = [10.4, 10.4, ..., 10.4]    ← 128 channels share it evenly
```

`R` is orthogonal: `R Rᵀ = I`. Insert it into `Y = XW`:

```text
Y = X W = (X R)(Rᵀ W)
          ↑ rotate activations   ↑ rotate weights back, cancelling out
```

The model output is **mathematically unchanged** — this is an identity, not an approximation — while the distributions of both tensors are rearranged. Once 118 becomes 10.4, per-token INT4 quantization becomes viable: the dynamic range compresses by √128 ≈ 11×, about 3.5 bits of effective precision.

<div class="keybox">
<strong>Where this sits:</strong> route C of the map in qt1 — change the distribution by rotation. No rounding change, no relocation, no magnification. It is the only route that supports <strong>W4A4</strong>. The cost is a few fused kernels per layer and a modified weight layout. qt5 covers its orthogonal partner, KV cache quantization.
</div>

## 1. Why Rotation Removes Outliers for Free

### 1.1 Step one: insert an orthogonal matrix

For any orthogonal `R` (`R Rᵀ = I`):

```text
Y = X W
  = X (R Rᵀ) W
  = (X R) (Rᵀ W)
  = X̃  W̃
```

`X̃ = X R` is the rotated activation, `W̃ = Rᵀ W` the counter-rotated weight. Since `R Rᵀ` is exactly the identity, **no error is introduced**. It is a pure reparameterisation.

The key asymmetry: `W̃` is computed **offline**. At inference the weights are already rotated, and the activation only needs one multiplication by `R` before entering the GEMM — and if `R` is chosen well (§1.3), that multiplication costs almost nothing.

### 1.2 Step two: one outlier becomes d pieces

Suppose `x` has a single huge value `M` in channel 5 and near-zeros elsewhere. With `R` the normalised Hadamard matrix:

```text
x̃[k] = Σ_i x[i] · H[i, k] ≈ M · H[5, k] = ± M/√d
```

**Every output channel receives `M/√d`.** For `d = 128`, 118 becomes 10.4. The quantization `absmax` drops from 118 to 10.4, the step `Δ` shrinks by 11×, and effective precision rises by about 3.5 bits.

That is the entire magic: rotation does not destroy energy, it distributes it from one channel across `d` channels.

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Orthogonal rotation flattening an outlier">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">Same vector, one rotation, dynamic range down 11×</text>

  <text x="20" y="54" font-size="12" font-weight="700" fill="#b91c1c">Before: x</text>
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
  <text x="288" y="238" font-size="10" fill="#6b7280">128 channels</text>
  <text x="20" y="264" font-size="11" fill="#b91c1c" font-weight="700">absmax = 118 → INT4 step 15.7 → everything but the outlier rounds to zero</text>

  <path d="M 336 150 L 372 150" stroke="#1d4ed8" stroke-width="2" marker-end="url(#a4)"/>
  <defs><marker id="a4" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L6,3 L0,6 z" fill="#1d4ed8"/></marker></defs>
  <text x="354" y="142" font-size="11" fill="#1d4ed8" text-anchor="middle" font-weight="700">× R</text>

  <text x="384" y="54" font-size="12" font-weight="700" fill="#15803d">After: x·R (normalised Hadamard)</text>
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

  <text x="384" y="264" font-size="11" fill="#15803d" font-weight="700">absmax = 10.4 → INT4 step 1.4 → every channel survives</text>
</svg>
<p class="cap">Figure 1: Rotation redistributes rather than destroys energy. The L2 norm of the left and right vectors is identical — that is the visual form of the mathematical invariance.</p>
</div>

### 1.3 Hadamard: the only choice that is both good and fast

Could `R` be any orthogonal matrix? In theory yes — a random orthogonal matrix also flattens outliers (a high-dimensional concentration phenomenon). In practice there is exactly one answer: the **normalised Hadamard matrix**.

Three reasons:

```text
① Orthogonal: H Hᵀ = I, so the "free" precondition holds
② Fast: for d = 128, the fast Walsh-Hadamard transform needs only
        log₂(128) = 7 butterfly rounds. Cost O(d log d) = 128×7 ≈ 900 ops,
        against d² = 16384 for a dense GEMM — negligible
③ Almost no floating-point multiplies: butterflies are adds and subtracts,
        with a single 1/√d at the end. Fixed-point friendly, tiny error
```

`H` is built from Kronecker powers: `H_128 = H_2^⊗7`, where `H_2 = (1/√2)[[1,1],[1,-1]]`.

```python
import math, torch

def fwht(x):
    """Normalised Hadamard transform. x: [..., d], d must be a power of 2."""
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

H = fwht(torch.eye(128))                 # each row transformed == H itself
assert torch.allclose(H @ H.T, torch.eye(128), atol=1e-5)
```

This access pattern is a standard butterfly network: it fits entirely in shared memory, and it can be fused into the epilogue of the preceding operator. **That is why rotation overhead can be held to 1–3%.**

### 1.4 The caveat: rotation is only free across linear maps

This is the easiest trap in the whole route. The identity `Y = (XR)(RᵀW)` requires that both sides of the insertion point **are linear**.

The moment a nonlinearity intervenes, `R` cannot pass through:

```text
✓ linear:      (X R)(Rᵀ W) = X W

✗ RMSNorm γ:   (x ⊙ γ) · W  ≠  x · (γ ⊙ ...)   ← γ is elementwise, blocks rotation
✗ SwiGLU:      (R g) ⊙ (R u) ≠ R (g ⊙ u)      ← the elementwise product does not commute
✗ RoPE:        RoPE(x R) ≠ RoPE(x) R          ← the 2D block structure does not line up
```

So rotation cannot be inserted anywhere you like. There are exactly **four** valid positions, and each has a different cost.

## 2. The Four Insertion Points: R1–R4

<div class="fig">
<svg viewBox="0 0 680 520" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The four rotation insertion points in a transformer block">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">Rotation can only go at these four places (LLaMA-style decoder block)</text>

  <rect x="20" y="36" width="14" height="10" fill="#dbeafe" stroke="#1d4ed8"/>
  <text x="40" y="45" font-size="10" fill="#6b7280">folded offline (weights change once, zero runtime cost)</text>
  <rect x="300" y="36" width="14" height="10" fill="#ffedd5" stroke="#ea580c"/>
  <text x="320" y="45" font-size="10" fill="#6b7280">online kernel (runs every forward pass)</text>

  <g font-size="11" text-anchor="middle">
    <rect x="90" y="62" width="230" height="26" rx="4" fill="#f3f4f6" stroke="#6b7280"/>
    <text x="205" y="79" fill="#374151">residual stream x (already in the rotated basis)</text>

    <rect x="90" y="106" width="230" height="26" rx="4" fill="#fff" stroke="#6b7280"/>
    <text x="205" y="123" fill="#374151">RMSNorm(x) · γ</text>

    <rect x="20" y="146" width="44" height="22" rx="11" fill="#dbeafe" stroke="#1d4ed8"/>
    <text x="42" y="161" font-size="11" font-weight="700" fill="#1d4ed8">R1</text>
    <rect x="90" y="144" width="230" height="26" rx="4" fill="#fff" stroke="#1d4ed8" stroke-width="1.5"/>
    <text x="205" y="161" fill="#1d4ed8">q/k/v/gate/up_proj</text>
    <text x="336" y="152" font-size="10" fill="#1d4ed8">γ sits in between, so the fold is:</text>
    <text x="336" y="167" font-size="10" fill="#1d4ed8">W' = W · diag(γ) · R · diag(1/γ)</text>

    <rect x="90" y="184" width="230" height="26" rx="4" fill="#fff" stroke="#6b7280"/>
    <text x="205" y="201" fill="#374151">RoPE (2D pairs along head_dim)</text>

    <rect x="20" y="224" width="44" height="22" rx="11" fill="#ffedd5" stroke="#ea580c"/>
    <text x="42" y="239" font-size="11" font-weight="700" fill="#ea580c">R4</text>
    <rect x="90" y="222" width="230" height="26" rx="4" fill="#fff" stroke="#ea580c" stroke-width="1.5"/>
    <text x="205" y="239" fill="#ea580c">Q ← Q·R, K ← K·R (the same R)</text>
    <text x="336" y="231" font-size="10" fill="#ea580c">QKᵀ = Q R Rᵀ Kᵀ = Q Kᵀ → scores unchanged,</text>
    <text x="336" y="246" font-size="10" fill="#ea580c">cancels inside softmax, no compensation needed</text>

    <rect x="90" y="262" width="230" height="26" rx="4" fill="#fff" stroke="#6b7280"/>
    <text x="205" y="279" fill="#374151">softmax(QKᵀ/√d) · V</text>

    <rect x="20" y="302" width="44" height="22" rx="11" fill="#dbeafe" stroke="#1d4ed8"/>
    <text x="42" y="317" font-size="11" font-weight="700" fill="#1d4ed8">R2</text>
    <rect x="90" y="300" width="230" height="26" rx="4" fill="#fff" stroke="#1d4ed8" stroke-width="1.5"/>
    <text x="205" y="317" fill="#1d4ed8">o_proj (per-head rotation)</text>
    <text x="336" y="309" font-size="10" fill="#1d4ed8">P·(V R) = (P V) R → absorb into W_o:</text>
    <text x="336" y="324" font-size="10" fill="#1d4ed8">W_o' = W_o · R (block-diagonal per head)</text>

    <rect x="90" y="340" width="230" height="26" rx="4" fill="#f3f4f6" stroke="#6b7280"/>
    <text x="205" y="357" fill="#374151">⊕ residual add</text>

    <rect x="90" y="384" width="230" height="26" rx="4" fill="#fff" stroke="#6b7280"/>
    <text x="205" y="401" fill="#374151">RMSNorm₂(x) · γ₂</text>

    <rect x="90" y="422" width="230" height="26" rx="4" fill="#fff" stroke="#6b7280"/>
    <text x="205" y="439" fill="#374151">SiLU(gate(x)) ⊙ up(x)</text>
    <text x="336" y="430" font-size="10" fill="#b91c1c">↑ elementwise product: rotation stops here</text>
    <text x="336" y="445" font-size="10" fill="#b91c1c">  a Hadamard must be computed right here</text>

    <rect x="20" y="462" width="44" height="22" rx="11" fill="#ffedd5" stroke="#ea580c"/>
    <text x="42" y="477" font-size="11" font-weight="700" fill="#ea580c">R3</text>
    <rect x="90" y="460" width="230" height="26" rx="4" fill="#fff" stroke="#ea580c" stroke-width="1.5"/>
    <text x="205" y="477" fill="#ea580c">fwht(h) → down_proj</text>
    <text x="336" y="469" font-size="10" fill="#ea580c">fusable into the SiLU⊙up epilogue, once per layer;</text>
    <text x="336" y="484" font-size="10" fill="#ea580c">d=11008 is not a power of two, so pad to 16384</text>
  </g>

  <g stroke="#9ca3af" stroke-width="1.2">
    <path d="M205,88 L205,106"/><path d="M205,132 L205,144"/>
    <path d="M205,170 L205,184"/><path d="M205,210 L205,222"/>
    <path d="M205,248 L205,262"/><path d="M205,288 L205,300"/>
    <path d="M205,326 L205,340"/><path d="M205,366 L205,384"/>
    <path d="M205,410 L205,422"/><path d="M205,448 L205,460"/>
  </g>
</svg>
<p class="cap">Figure 2: The four positions. Blue folds offline into the weights at zero runtime cost; orange must be computed online. R3 and R4 are the latter, and R3 is where the real cost lives.</p>
</div>

### 2.1 R1: the residual stream — fold into every weight

This is the most valuable rotation, because it handles the input outliers of all five projection layers at once.

**Once the residual stream is rotated, it must stay rotated** — from the embedding to the final `lm_head`, the entire residual path lives in the rotated basis. So it is not one rotation but a rewrite of all the relevant weights:

```python
# inbound layers (consume the residual stream): y = x @ W.T, x already rotated
# γ sits between the norm and the linear, so it must be folded too
Dg, Dgi = torch.diag(gamma), torch.diag(1.0 / gamma)
W_in_new = W_in @ Dg @ R @ Dgi        # [out, in], note the multiplication order

# outbound layers (write the residual): require y' = y·R  ⇒  W_out_new = Rᵀ W_out
W_out_new = R.T @ W_out

# boundaries: embedding input and lm_head output
E_new    = E    @ R                   # [vocab, d]
W_lm_new = R.T  @ W_lm                # [vocab, d]; usually unquantized but still rotated
```

<div class="warnbox">
<strong>The step most write-ups get wrong:</strong> it is tempting to conclude that RMSNorm is rotation-invariant, therefore γ is unaffected. The first half is true (<code>||Rx||₂ = ||x||₂</code>, RMS only looks at the norm); the second half is false, because <code>γ</code> is elementwise and <code>(Rx) ⊙ γ ≠ R(x ⊙ γ)</code>. The correct fold is the <code>W · diag(γ) · R · diag(1/γ)</code> above: pull the inverse of γ back to the input side, and fold γ together with R into the weight. Dropping one diag breaks the model outright, and the symptom is an unexplained accuracy cliff that is very hard to trace.
</div>

### 2.2 R2: attention output — fold into o_proj

The attention output `PV` also carries outliers (certain channels of certain heads). Rotate along each head's `head_dim` (= 128, conveniently a power of two):

```text
P·(V R) = (P V) R     ← P multiplies from the left, R from the right, different axes
```

`R` passes through `P` and is then absorbed by `o_proj`:

```text
(PV R) W_o = (PV) (R W_o)   ⇒   W_o' = R · W_o   ← block-diagonal per head
```

Note `R` here is **block-diagonal**: `blockdiag(R_hd, R_hd, ..., R_hd)` with `n_heads` blocks, because `o_proj`'s input dimension is `n_heads × head_dim` and each head rotates independently.

### 2.3 R3: after SwiGLU — the one unavoidable online rotation

The input to `down_proj`, `h = SiLU(gate(x)) ⊙ up(x)`, has the most severe outliers of any intermediate activation, because SiLU clips the negative half and leaves the positive half unbounded — a naturally skewed distribution.

But `⊙` is elementwise:

```text
(R g) ⊙ (R u)  ≠  R (g ⊙ u)
```

**Rotation cannot pass through it.** So the Hadamard must be computed directly on `h`:

```python
h = F.silu(gate_out) * up_out     # [B, T, 11008]
h = fwht(h)                        # online rotation — the only non-foldable spot
y = h @ down_proj.T
```

> **A common misconception**: could you pre-rotate the `gate` and `up` outputs separately and skip the FWHT on `h`? No. `(R g) ⊙ (R u) ≠ R(g ⊙ u)` — the elementwise product always blocks the commutation. Even if `gate`/`up` outputs were already in a rotated basis, the rotation on `h` is unavoidable. This is also why nearly all of QuaRot's runtime overhead sits in R3.

Two engineering details:

1. **`d = 11008` is not a power of two** (LLaMA-2's intermediate size). QuaRot pads to **16384** (2¹⁴), fills the extra channels with zeros, rotates, then truncates. Padding inflates this step's compute by 49%, so some implementations instead construct an approximate 11008×11008 Hadamard, or use 128×128 block-diagonal rotation with cross-block permutation — that is exactly DuQuant's approach (§3.3).
2. **The FWHT can be fused into `up_proj`'s epilogue**: compute `h` and run the butterfly immediately in shared memory, without spilling back to HBM. Whether this fusion is done decides whether the overhead is 1% or 10%.

### 2.4 R4: Q/K rotation — genuinely free

This one is elegant. After RoPE, apply the **same** `R` to both `Q` and `K`, acting along `head_dim`:

```text
scores = Q R · (K R)ᵀ = Q R Rᵀ Kᵀ = Q Kᵀ
```

The scores are bit-identical (in exact arithmetic). No compensation term, no weight modification, and `V` is unaffected.

The benefit: the magnitudes of `Q` and `K` are spread out, so the **KV cache can be quantized more aggressively** — which qt5 covers in detail. R4 is the only genuine free lunch; QuaRot's KV4 result rests on it.

## 3. Three Generations: QuaRot → SpinQuant → DuQuant

Given the same `R`, how you choose it is the whole question. The three generations progressively open up the search space for `R`:

| Gen | Method | Where `R` comes from | Training needed | Key insight |
|---|---|---|---|---|
| 1 | **QuaRot** (2024.04) | hand-specified: Hadamard (+ random permutation) | none (pure PTQ) | a random rotation suffices; makes W4A4 move from impossible to feasible |
| 2 | **SpinQuant** (2024.05) | Cayley-parameterised optimisation on the manifold | yes (hours on a calibration set) | the **choice** of rotation matters a lot; a learned one beats a hand-picked one |
| 3 | **DuQuant** (2024.10) | block-diagonal rotation + zigzag channel permutation | none (pure PTQ) | global rotation creates **new** outliers; handle it block-locally |

### 3.1 QuaRot: proof that a random rotation is enough

QuaRot's contribution is not inventing rotation (that idea was already in the quantization community) but **finding all four insertion points and getting every fold right**, which is what first brought LLaMA-2-7B W4A4 (including KV4) close to FP16 under pure PTQ.

Its `R` choices are deliberately plain:

```text
R1 = random permutation P₁ · Hadamard H     (residual, d = 4096)
R2 = one H per head                         (head_dim = 128)
R3 = Hadamard H                             (intermediate, padded to 16384)
R4 = one H per head                         (Q/K, head_dim = 128)
```

The random permutation `P` has a specific purpose: the Hadamard matrix has fixed row structure, so certain input patterns could align to produce a *new* spike after rotation. A random permutation breaks that regularity by randomising where the outlier lands. **It is a small, very practical trick**: `x → permute(x) → H → ...`.

QuaRot's reported numbers (LLaMA-2-7B, WikiText PPL):

```text
FP16 baseline                 5.47
RTN W4A4 (no rotation)        ~2000   ← total collapse
QuaRot W4A4                   6.10    ← +0.63 only
QuaRot W4A4 + KV4             6.19
```

Compare with AWQ's W4A16 result of ~5.6 from qt3: **QuaRot spends half the bit width and pays 0.5 more PPL.** That is the case for the rotation route existing at all.

### 3.2 SpinQuant: the choice of rotation matters more than its existence

SpinQuant starts from the observation that QuaRot uses a **fixed** Hadamard, while `R` has enormous freedom (the full `d×d` orthogonal group) and different choices differ substantially in accuracy.

Its approach is to make `R` learnable while keeping it on the orthogonal manifold — plain gradient descent would leave the constraint. The remedy is the **Cayley transform**:

```text
R = (I − A)(I + A)⁻¹ ,  A skew-symmetric (Aᵀ = −A)
```

For any skew-symmetric `A`, this `R` is **automatically orthogonal**, so gradient descent can operate in the unconstrained `A` space and `R` stays valid.

```python
class CayleyRotation(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(d, d))   # only the upper triangle is used

    def forward(self):
        A = self.A.triu(1) - self.A.triu(1).T            # enforce skew-symmetry
        I = torch.eye(A.shape[0], device=A.device)
        return torch.linalg.solve(I + A, I - A)          # (I+A)⁻¹ (I−A)
```

The objective is not "minimise quantization error" (not differentiable) but **minimise the difference between the model's output before and after quantization** — a few hundred steps on a WikiText calibration set, learning only `R` with the weights frozen.

SpinQuant's useful conclusion: **the learned `R` resembles Hadamard but is not identical to it**, and on LLaMA-2-7B W4A4 it moves PPL from 6.10 to 5.86; on LLaMA-3-8B the gap is larger. The cost is one training run (hours on a single A100) and one `R` set per model.

### 3.3 DuQuant: global rotation creates new outliers

DuQuant points at something the first two generations overlooked: **Hadamard is global (across the whole `d` dimension), and it introduces new outliers.**

The cause is channel-wise heterogeneity. Suppose one contiguous segment of channels has uniformly small magnitudes (a "dead zone") while another segment is large. A global Hadamard spreads the large segment's energy into the small segment — creating huge values where there were none. A previously quiet region is contaminated.

DuQuant's two countermeasures:

```text
① block-diagonal rotation: rotate only within 128×128 (or 64×64) blocks,
   so noise cannot propagate across blocks
   cost: an outlier inside a block is spread over only 128 channels,
   so compression is √128 rather than √d

② zigzag permutation: sort channels by activation magnitude, then
   reorder them in a zigzag so that every block receives a mix of
   "large-small-large-small" rather than "one all-large block and
   one all-small block". This flattens each block's internal dynamic
   range, which is what makes block-diagonal rotation effective.
```

The `zigzag` name describes the reordering path: with channels sorted by magnitude `[0,1,2,3,...]`, they are dealt into blocks in a snake pattern so each block gets a "strong–weak" pair. It is the same idea as bandwidth optimisation in sparse matrices — **compress global information into local structure.**

DuQuant's gains are largest on **small models and low bit widths**: LLaMA-2-7B W4A4 reaches 5.98, and on models where attention outliers are more stubborn (LLaMA-3-8B, Qwen2) DuQuant is markedly more stable than QuaRot, which outright collapses on some layers.

<div class="keybox">
<strong>The three generations in one line each:</strong><br>
QuaRot answers «<strong>can</strong> rotation do W4A4» — yes, a random Hadamard suffices.<br>
SpinQuant answers «<strong>which</strong> rotation is best» — a learned one beats a hand-picked one, at the cost of a training run.<br>
DuQuant answers «what <strong>side effects</strong> does global rotation have» — it creates new outliers; go block-local with permutation.
</div>

## 4. Why W4A4 Is Worth All This Trouble

Back to engineering reality: W4A16 already compresses weights to 4 bits. Why risk quantizing activations too?

Because **W4A16's benefit lands entirely in decode**. As qt1 established: decode is bandwidth-bound (every generated token must stream the whole weight tensor), so weight bit width converts directly into speedup. Prefill is **compute-bound** — it processes thousands of tokens, the GEMM is large-times-large, activations are not the bottleneck, and the weight read is amortised.

```text
                       weight bandwidth   activation bandwidth   GEMM shape
decode (batch 1)       ★★★ bottleneck      negligible             GEMV
prefill (long prompt)  amortised          ★★ relevant            GEMM
```

Quantizing activations (A4) buys:

1. **Tensor Core throughput in prefill**: INT4×INT4 MMA has a theoretical peak 2× INT8 and 4× FP16 at equal compute, if the hardware supports native int4 mma.
2. **Long-context scenarios**: prefill can be 70%+ of total latency, which is when A4 pays off.
3. **Secondary memory bandwidth savings**: activations are not resident, but large-batch intermediate activations and KV cache are not negligible.

Reality is harsher: **current mainstream GPUs (H100/H200/B200) have no native INT4×INT4 MMA.** INT4 must be unpacked and converted to INT8/FP16 before compute, at which point W4A4 gives almost no speedup over W4A16 while paying extra quantization error. Hence:

<div class="warnbox">
<strong>W4A4's real situation today:</strong> on general-purpose GPUs it is a <strong>memory-saving</strong> technique, not a <strong>speed</strong> technique — it lets you fit a larger batch or a longer context on one card, but the kernel-level throughput advantage requires native low-bit MMA. The organisations actually harvesting W4A4 are those with <strong>custom accelerators</strong> (some domestic NPUs, LPU-class architectures) and <strong>edge silicon</strong> (Qualcomm Hexagon and Apple ANE have native int4 paths). This is precisely why industry still defaults to W4A16 (AWQ/GPTQ) and W8A8 (FP8).
</div>

## 5. Deployment Checklist

If you are actually going to use rotation-based quantization, this is the order of operations:

```text
① Establish the baseline: run FP16 PPL, then RTN W4A4.
   If RTN does not collapse, this model does not need rotation.
② Pick a generation:
   - quick validation           → QuaRot (a pip install, one command to rewrite weights)
   - best accuracy, training ok → SpinQuant
   - small model / stubborn outliers (LLaMA-3, Qwen2) → DuQuant
③ Verify every fold is complete: is R1 folded into all in/out projections?
   is R2 block-diagonal per head? is R3's non-power-of-two padding handled?
   Miss one and accuracy drops with no error message.
④ Measure three things: PPL (mean), worst-layer error (dump each layer's
   output against FP16), and long context (32k+ degrades first and must be
   tested separately).
⑤ Measure speed: without native int4 mma, W4A4 may be SLOWER than W4A16.
   Always benchmark on the target hardware — never trust paper numbers.
```

One practical warning: **rotation is far more expensive to debug than AWQ or GPTQ.** Failures do not produce gibberish; they produce "PPL went from 5.5 to 6.3." You cannot easily tell whether a fold was missed, the calibration set mismatched, or one layer's fusion was skipped. The recommended diagnostic is to **dump per-layer outputs** and compare relative error against FP16, locating the first anomalous layer.

## 6. Series Navigation

- Previous **qt3 · SmoothQuant and AWQ**: route B, change the distribution by scaling, ceiling W4A16.
- Next **qt5 · KV Cache Quantization and FP4**: the orthogonal compression dimension, and how E4M3 and NVFP4 change the rules.
- **qt2 · RTN and GPTQ**: route A, rounding only — the baseline everything else is measured against.
- **qt1 · The Basic Language**: granularity, symmetry, calibration, the four-route map.

Related on this site: **op3 (Ascend EagleQBSA)** contains a real instance of R4's benefit — INT8 per-head for Q/K while P and V stay FP8; **fw4** covers the fusion layers of quantization kernels from the engine side; **op2** is the series index and lineage table.
