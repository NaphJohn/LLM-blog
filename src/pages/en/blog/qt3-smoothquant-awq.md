---
title: "Quantization Notes (3): SmoothQuant and AWQ — two opposite answers to activation outliers"
description: "LLM activations contain roughly 0.1 percent of channels whose magnitude is 50 to 100 times the rest, and this is the real reason W8A8 and W4A16 lose accuracy. SmoothQuant moves the outlier from activations into weights using a per-channel equivalence transform. AWQ does the opposite: it leaves the outlier in place and enlarges the salient channels weights so they occupy more quantization bins. Both rest on the same identity X·diag(s)⁻¹·diag(s)·W and differ only in how s is computed and which direction it pushes. Includes the full derivation, the hardware contract that forces SmoothQuant to exist, an honest account of when AWQ actually pays off (it depends on whether the salient channel is already the group absmax), grid-search pseudocode, and how the two stack with GPTQ."
pubDate: 2026-10-27
series: Quantization Notes
lang: en
altLang: zh
altHref: /blog/qt3-smoothquant-awq
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

SmoothQuant and AWQ solve the **same problem** with the **same mathematical tool** in **opposite directions**:

```text
Y = X W = (X · diag(s)⁻¹) · (diag(s) · W)
          ↑ shrink activations   ↑ enlarge weights
```

- **SmoothQuant**: set `s` from the ratio of activation max to weight max, and **push the difficulty from activations into weights** — because weight distributions are naturally uniform and easy to quantize, while activation outliers are the actual problem.
- **AWQ**: set `s` from activation salience, and push the **other** way — enlarge the weights of salient channels before quantizing, which gives those channels a higher effective resolution at the cost of making their activations look more extreme.

One flattens activations, the other amplifies weights. Both aim at the same thing: making the 4-bit grid cover the information more evenly. Once you see this symmetry you never have to memorize the two separately.

<div class="keybox">
<strong>Where this sits:</strong> route B of the map in qt1 — change the distribution (scaling), leave the rounding alone. Orthogonal to route A: you can run AWQ to fix the scales, then GPTQ to fix the rounding. qt4 does the same kind of thing with rotation, but reaches W4A4.
</div>

## 1. The Root Problem: Activation Outliers

### 1.1 The phenomenon

LLM activations have a peculiar structure: **most channels are near zero on almost every token, but a few channels blow up to tens or hundreds on rare tokens.** This is specific to LLMs and quite unlike vision models — the LLM.int8() paper was the first to document it systematically.

<div class="fig">
<svg viewBox="0 0 680 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Distribution of activation outlier channels">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">Why one channel can destroy a whole layer</text>

  <text x="20" y="54" font-size="12" font-weight="700" fill="#1d4ed8">Max activation magnitude per channel, |X[:, i]|</text>
  <line x1="30" y1="200" x2="330" y2="200" stroke="#9ca3af" stroke-width="1.5"/>
  <line x1="30" y1="70" x2="30" y2="200" stroke="#9ca3af" stroke-width="1.5"/>
  <text x="14" y="74" font-size="10" fill="#6b7280">120</text>
  <text x="18" y="200" font-size="10" fill="#6b7280">0</text>

  <g fill="#bfdbfe" stroke="#60a5fa">
    <rect x="36" y="190" width="7" height="10"/><rect x="46" y="188" width="7" height="12"/>
    <rect x="56" y="191" width="7" height="9"/><rect x="66" y="186" width="7" height="14"/>
    <rect x="76" y="192" width="7" height="8"/><rect x="86" y="189" width="7" height="11"/>
    <rect x="96" y="187" width="7" height="13"/><rect x="106" y="191" width="7" height="9"/>
    <rect x="116" y="185" width="7" height="15"/>
  </g>
  <rect x="126" y="76" width="9" height="124" fill="#dc2626" stroke="#991b1b"/>
  <text x="130" y="70" font-size="10" fill="#b91c1c" text-anchor="middle">118</text>
  <g fill="#bfdbfe" stroke="#60a5fa">
    <rect x="140" y="189" width="7" height="11"/><rect x="150" y="192" width="7" height="8"/>
    <rect x="160" y="186" width="7" height="14"/><rect x="170" y="190" width="7" height="10"/>
    <rect x="180" y="188" width="7" height="12"/><rect x="190" y="191" width="7" height="9"/>
    <rect x="200" y="185" width="7" height="15"/><rect x="210" y="189" width="7" height="11"/>
    <rect x="220" y="192" width="7" height="8"/><rect x="230" y="187" width="7" height="13"/>
    <rect x="240" y="190" width="7" height="10"/><rect x="250" y="188" width="7" height="12"/>
    <rect x="260" y="191" width="7" height="9"/><rect x="270" y="186" width="7" height="14"/>
    <rect x="280" y="189" width="7" height="11"/><rect x="290" y="192" width="7" height="8"/>
  </g>
  <text x="36" y="222" font-size="10" fill="#6b7280">channel index →</text>

  <line x1="20" y1="240" x2="660" y2="240" stroke="#e5e7eb" stroke-width="1"/>
  <text x="20" y="262" font-size="11" fill="#374151">With INT8 and a per-tensor absmax of 118, the step size is 118/127 ≈ 0.93.</text>
  <text x="20" y="280" font-size="11" fill="#374151">Every channel below 0.47 rounds to zero — that is 99% of the tensor.</text>
  <text x="20" y="302" font-size="11" fill="#b91c1c" font-weight="700">One channel consumes the entire dynamic range; SmoothQuant and AWQ are two ways of taking it back.</text>
</svg>
<p class="cap">Figure 1: A handful of channels carry almost all the magnitude. A global absmax makes the grid useless for everything else.</p>
</div>

Note that the outlier is **not** a bug or a numerical artifact. It is structural, it appears in every LLM, and it is consistent across inputs — which is exactly why an offline calibration can find it.

### 1.2 Why you cannot just use per-channel scales for activations

This is the question that unlocks the whole section. If per-channel works for weights, why not for activations?

The answer is a hardware contract, not a mathematical one:

```text
INT8 GEMM on Tensor Cores:
  C = A · B     where A and B are int8 matrices
  The only place a scale can be applied is the EPILOGUE — after the whole
  accumulation, once. That means ONE scalar per output tile.

per-channel weights:      s_W[out, 1]     → folds into the epilogue fine
per-token activations:    s_X[1, in]      → does NOT fold in; it varies along
                                            the reduction dimension
per-token × per-channel:  s_X[t, 1] × s_W[1, out] → a full 2D scale matrix,
                                            which Tensor Cores cannot absorb
```

So the hardware is not being stubborn for no reason: a scale that varies along the accumulation axis would have to be applied **inside** the MMA loop, which means one scaling operation per partial sum — that is, it stops being a matmul and becomes a loop.

SmoothQuant's contribution is to accept this constraint and work with it: instead of fixing the scale problem, **fix the distribution so one scalar suffices.**

## 2. SmoothQuant: Move the Outlier from Activations to Weights

### 2.1 The core identity

```text
Y = X W = (X · diag(s)⁻¹) · (diag(s) · W)
           ↑ call it X̂        ↑ call it Ŵ
```

`s` is a per-input-channel vector. `X̂` is the activation with channel `j` divided by `s_j`; `Ŵ` is the weight with row `j` multiplied by `s_j`. The product is mathematically identical.

The design goal: choose `s_j` so that **`X̂` and `Ŵ` are both easy to quantize**. Since weights are already easy, that means pushing difficulty into them until activations are tame.

```text
s_j = max(|X[:, j]|)^α / max(|W[j, :]|)^(1−α)

α = 0    →  s = 1/max|W|        →  all the difficulty stays on activations
α = 1    →  s = max|X|          →  all the difficulty moves to weights
α = 0.5  →  the balanced midpoint, and the paper's default
```

The formula is easier to read in log space. Define `a_j = log max|X[:,j]|` and `w_j = log max|W[j,:]|`. Then:

```text
log s_j = α·a_j − (1−α)·w_j

The post-transform magnitudes:
  log max|X̂[:,j]| = a_j − log s_j = (1−α)(a_j + w_j)
  log max|Ŵ[j,:]|  = w_j + log s_j = α(a_j + w_j)

Both are proportional to (a_j + w_j) — the total difficulty is SPLIT, not moved.
```

That last line is the real insight: SmoothQuant cannot create or destroy "quantization difficulty" (the product is invariant), it can only **redistribute** it. `α` sets the split ratio.

### 2.2 Why α = 0.5

Set `α = 0.5` and both post-transform magnitudes become `√(max|X_j| · max|W_j|)`. That is, **both tensors end up with the same maximum per channel** — geometric mean equalisation.

```text
Why equal maxima is the right target:
  · The error of uniform quantization scales with the step size,
    and the step size scales with the absmax.
  · Two tensors with maxima M and m: total error ∝ M + m
  · Subject to M·m = const (the equivalence constraint), M + m is
    minimised when M = m — AM-GM inequality.
```

So `α = 0.5` is not a magic constant; it is the AM-GM optimum under the constraint that the product of the two maxima is fixed. This is a genuinely satisfying derivation and it also tells you exactly when to deviate.

### 2.3 When to deviate from 0.5

Equal maxima is optimal **only if both tensors are equally sensitive to quantization error.** In practice they are not, and the paper's own guidance (grid-search 0.4–0.7) reflects this:

```text
α < 0.5 (shift more difficulty onto weights):
  · weights are more tolerant (they have a lot of redundancy)
  · activation outliers are extremely severe (some models, some layers)
  · typical for early layers where activations are wild but weights are smooth

α > 0.5 (keep more difficulty on activations):
  · weights are small models with little redundancy
  · activations are already well-behaved
  · typical for the last layers
```

The practical recipe: grid-search `α ∈ {0.4, 0.5, 0.6, 0.7}` per model, measure output MSE on a held-out batch, pick the best. It is a one-dimensional search with four candidates — cheap enough that there is no excuse not to do it.

### 2.4 Implementation: fold it into LayerNorm

The crucial engineering detail — and the reason SmoothQuant has near-zero runtime cost — is that `s` is absorbed into the preceding LayerNorm or RMSNorm:

```python
# Original:  x̂ = RMSNorm(x) * γ ,  then y = x̂ @ W.T
# After SQ:  x̂' = RMSNorm(x) * γ / s   ← just modify γ!
#            Ŵ  = W * s                ← just modify the weight!

gamma_new = gamma / s                  # one vector, one division
W_new     = W * s.unsqueeze(0)         # broadcast over input dimension

# At inference: absolutely nothing extra. The norm already multiplies by γ.
```

So the deployed model has:

```text
· modified norm weights (γ/s)
· modified linear weights (W·s)
· NO extra kernel, NO extra memory, NO extra latency
```

This is why SmoothQuant became the standard W8A8 method so quickly. But it also reveals its biggest limitation:

<div class="warnbox">
<strong>SmoothQuant only works with pre-norm architectures.</strong> The transform must be absorbed into the normalisation that <em>immediately precedes</em> the linear layer. In a pre-norm transformer (GPT, LLaMA, most modern LLMs) that is always true. In a post-norm architecture, or anywhere the activation comes from a residual add with no norm in between, there is no γ to absorb into — and you need an extra elementwise kernel, which kills the zero-overhead property.
</div>

The paper also notes that for very large models, the `s` vector can be further folded into the preceding linear's output scale, but this requires modifying that layer too — a chain of absorptions that terminates at the embedding. In practice most implementations stop at the norm.

### 2.5 Results and limits

```text
LLaMA-2-7B, WikiText PPL:
FP16                      5.47
INT8 W8A8 per-tensor      5.60     (+2.4%)   ← naive baseline
INT8 W8A8 + SmoothQuant   5.51     (+0.7%)   ← the method's win
```

An 8× reduction in the accuracy gap, at zero runtime cost. That is a good deal.

The limits are strict, though:

```text
✗ Cannot reach W4A8 or W4A4
    The transform manipulates the MAXIMA. At 4 bit the grid is so coarse
    that equalising maxima is nowhere near enough — you need the whole
    distribution to be Gaussian-ish, which scaling cannot achieve.
    → that is qt4's job.

✗ Does not apply to the attention P matrix
    softmax(QKᵀ) has no preceding norm to absorb into, and its rows sum to 1
    by construction, which pinches the distribution in a way scaling cannot fix.
    Most implementations keep P in FP16.

✗ Loses relevance in the FP8 era
    See below.
```

### 2.6 The FP8 era problem

SmoothQuant exists because INT8's uniform ±127 range cannot cover both an outlier at 118 and normal values at 0.5. FP8 E4M3 has a dynamic range of ±448 **and** a non-uniform grid — the 4-bit exponent means 0.5 and 118 each get their own 3-bit mantissa resolution.

```text
LLaMA-2-7B W8A8, three configs:
INT8  per-tensor                       5.60
INT8  per-token × per-channel          5.51    ← needs SmoothQuant's machinery
FP8   E4M3 per-tensor, dynamic         5.50    ← no transform at all
```

**FP8 per-tensor already matches INT8's finest per-token-per-channel scheme.** That is format doing the work that algorithm used to have to do. Consequently, on H100-class hardware, the correct answer for W8A8 is usually "just use FP8", and SmoothQuant's remaining value is confined to INT8-only hardware — older GPUs, and a meaningful set of domestic accelerators.

Conversely, **AWQ's relevance is rising**, because on-device and single-card deployment is the dominant demand right now, and W4A16 is exactly the right answer there.

## 3. AWQ: Instead of Moving the Outlier, Protect the Salient Channels

### 3.1 The inversion

AWQ starts from an observation that reframes the problem: not all weights matter equally. About 0.1–1% of channels are "salient" — they carry most of the information, and they are identifiable by their **activation magnitude** (a channel with large activations is one the model actually uses).

So instead of moving difficulty around, AWQ asks: **can I give those few channels more effective resolution?**

Its scale is:

```text
s_j = mean(|X[:, j]|)^α        α ≈ 0.5, grid-searched over 0.4–0.7
```

No weight term at all — it depends only on activation statistics. And crucially, it is applied **per group**, geometric-mean normalised:

```python
def awq_scale(X, W, alpha=0.5, group_size=128):
    """Compute per-input-channel scales for AWQ. Returns s: [in_f]."""
    act_mean = X.abs().mean(dim=0)                     # [in_f]
    act_mean = act_mean.clamp(min=1e-6)
    s = act_mean ** alpha

    # geometric-mean normalisation WITHIN each group, so that the scale
    # does not change the group's absmax more than necessary
    out_f, in_f = W.shape
    s = s.reshape(in_f // group_size, group_size)
    s = s / s.prod(dim=-1, keepdim=True) ** (1.0 / group_size)
    return s.reshape(in_f)

# apply
W_scaled = W * s.unsqueeze(0)      # enlarge salient input channels
# X_scaled = X / s                 # (never materialised; folded into the norm)
```

The geometric-mean normalisation is the subtle part, and it is what makes the method actually work — see §3.3.

### 3.2 Why enlargement helps: the group-absmax mechanism

This is the step that most explanations get wrong, so it is worth being precise.

The naive story is: "enlarge the weight by `s`, quantize, then divide by `s` on dequantization, so the relative error stays the same but the absolute error shrinks by `s`." **That story is wrong**, because it implicitly assumes the quantization step `Δ` is constant. It is not — `Δ` is set by the group absmax, and enlarging a weight changes the group absmax.

Correct analysis. Within a group, uniform quantization gives every element the same absolute error bound `Δ/2`, where `Δ = absmax / (2^(b−1) − 1)`. Relative error is therefore `Δ/2 / |w|` — **large for small weights**. Salient channels often sit in the long tail of the weight distribution (they are important because they are *precisely tuned*, not because they are large), so they get poor relative error.

Now multiply a salient channel's weight by `s > 1` and consider two cases:

```text
Case A: the enlarged channel becomes the new group absmax
  Δ increases by factor s (the group must now cover s·w)
  · the enlarged channel: absolute error ↑s, then ÷s on dequant → unchanged
  · every other channel: absolute error ↑s, no compensation → WORSE
  → net loss. Enlarging was a mistake.

Case B: a larger weight elsewhere already anchors the group absmax
  Δ is unchanged
  · the enlarged channel: its relative magnitude goes from w to s·w,
    so it occupies s× more bins at the same Δ
    absolute error stays Δ/2, dequant divides by s → error ÷s   ✓
  · every other channel: Δ unchanged, error unchanged
  → net gain, for free.
```

<div class="fig">
<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="AWQ redistributing resolution within a quantization group">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">AWQ works when the group absmax is already anchored elsewhere</text>
  <text x="20" y="42" font-size="10" fill="#6b7280">A 16-bin INT4 group, one salient channel (B) and one non-salient anchor (A)</text>

  <!-- before -->
  <text x="20" y="72" font-size="12" font-weight="700" fill="#374151">Before scaling: absmax = 2.5, anchored by channel A</text>
  <line x1="40" y1="120" x2="640" y2="120" stroke="#9ca3af" stroke-width="1.5"/>
  <g stroke="#9ca3af">
    <line x1="40" y1="114" x2="40" y2="126"/><line x1="80" y1="114" x2="80" y2="126"/>
    <line x1="120" y1="114" x2="120" y2="126"/><line x1="160" y1="114" x2="160" y2="126"/>
    <line x1="200" y1="114" x2="200" y2="126"/><line x1="240" y1="114" x2="240" y2="126"/>
    <line x1="280" y1="114" x2="280" y2="126"/><line x1="320" y1="114" x2="320" y2="126"/>
    <line x1="360" y1="114" x2="360" y2="126"/><line x1="400" y1="114" x2="400" y2="126"/>
    <line x1="440" y1="114" x2="440" y2="126"/><line x1="480" y1="114" x2="480" y2="126"/>
    <line x1="520" y1="114" x2="520" y2="126"/><line x1="560" y1="114" x2="560" y2="126"/>
    <line x1="600" y1="114" x2="600" y2="126"/><line x1="640" y1="114" x2="640" y2="126"/>
  </g>
  <text x="40" y="140" font-size="9" fill="#6b7280" text-anchor="middle">0</text>
  <text x="640" y="140" font-size="9" fill="#6b7280" text-anchor="middle">2.5</text>
  <circle cx="630" cy="120" r="4" fill="#6b7280"/>
  <text x="630" y="108" font-size="9" fill="#6b7280" text-anchor="middle">A = 2.48</text>
  <circle cx="196" cy="120" r="4" fill="#dc2626"/>
  <text x="196" y="108" font-size="9" fill="#b91c1c" text-anchor="middle">B = 0.60</text>
  <text x="40" y="162" font-size="10" fill="#374151">Δ = 2.5/15 = 0.167.  B sits at bin 4 and spans 3.6 bins,</text>
  <text x="40" y="178" font-size="10" fill="#374151">so its relative error is ≈ 0.167/2/0.60 = 14%.</text>

  <!-- after -->
  <text x="20" y="212" font-size="12" font-weight="700" fill="#15803d">After scaling B by s = 4: absmax still 2.5, still anchored by A</text>
  <line x1="40" y1="260" x2="640" y2="260" stroke="#9ca3af" stroke-width="1.5"/>
  <g stroke="#9ca3af">
    <line x1="40" y1="254" x2="40" y2="266"/><line x1="80" y1="254" x2="80" y2="266"/>
    <line x1="120" y1="254" x2="120" y2="266"/><line x1="160" y1="254" x2="160" y2="266"/>
    <line x1="200" y1="254" x2="200" y2="266"/><line x1="240" y1="254" x2="240" y2="266"/>
    <line x1="280" y1="254" x2="280" y2="266"/><line x1="320" y1="254" x2="320" y2="266"/>
    <line x1="360" y1="254" x2="360" y2="266"/><line x1="400" y1="254" x2="400" y2="266"/>
    <line x1="440" y1="254" x2="440" y2="266"/><line x1="480" y1="254" x2="480" y2="266"/>
    <line x1="520" y1="254" x2="520" y2="266"/><line x1="560" y1="254" x2="560" y2="266"/>
    <line x1="600" y1="254" x2="600" y2="266"/><line x1="640" y1="254" x2="640" y2="266"/>
  </g>
  <circle cx="630" cy="260" r="4" fill="#6b7280"/>
  <text x="630" y="248" font-size="9" fill="#6b7280" text-anchor="middle">A = 2.48</text>
  <circle cx="616" cy="260" r="4" fill="#dc2626"/>
  <text x="616" y="284" font-size="9" fill="#b91c1c" text-anchor="middle">B·4 = 2.40</text>
  <text x="40" y="302" font-size="10" fill="#374151">Δ is STILL 0.167 — unchanged, because A still defines the max.</text>
  <text x="40" y="318" font-size="10" fill="#15803d" font-weight="700">B now spans 14.4 bins instead of 3.6, so error ÷4 → 3.5%. Free accuracy.</text>
</svg>
<p class="cap">Figure 2: The gain is entirely about the group absmax not moving. If the scaled channel did become the new max, the step size would grow and everyone else would lose.</p>
</div>

So the net benefit of AWQ depends on one empirical fact: **are salient channels' weights already not the group maximum?** In practice this holds reasonably well, which is why AWQ works. But it is not guaranteed, which is exactly why `α` must be grid-searched:

```text
α too small → s ≈ 1, no protection at all
α too large → salient channels overshoot and become new absmax → case A, net loss
α ≈ 0.5    → typically lands in case B
```

The grid search is not tuning a hyperparameter for its own sake — it is searching for the point where the salient channels get pulled out of the long tail **without** being pushed past the anchor.

### 3.3 How salience is measured

AWQ tries two proxies and reports the second:

```text
Option 1: weight magnitude |w|
  → wrong. Large weights are not necessarily important; this is the textbook
    mistake that magnitude-based pruning also makes.

Option 2: activation magnitude mean(|X[:, j]|)
  → correct. A channel the model routes information through will have
    large activations. This is the actual definition of salience.
```

The paper shows option 2 decisively beats option 1 in ablation, which is a nice confirmation that the mechanism is activation-driven, not weight-driven.

### 3.4 The grid search

```python
def awq_search(X, W, grid=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), group_size=128,
               bits=4):
    """Pick alpha by measuring output error, not by heuristics."""
    best_alpha, best_err = None, float("inf")
    Y_ref = X @ W.T
    for alpha in grid:
        s = awq_scale(X, W, alpha, group_size)
        Wq = quantize_groupwise(W * s.unsqueeze(0), bits, group_size)
        # dequantize and undo the scale before comparing
        err = ((X @ (Wq / s.unsqueeze(0)).T) - Y_ref).pow(2).mean().item()
        if err < best_err:
            best_alpha, best_err = alpha, err
    return best_alpha
```

Because `α` is a single scalar, 20 evaluations suffice and a single GPU takes minutes. This makes **AWQ's calibration cost an order of magnitude below GPTQ's**, and it has no `damp`-style failure mode where a bad parameter destroys the model.

One more detail worth remembering: **AWQ only needs to protect about 1% of channels to recover most of the accuracy.** The paper's ablation shows protection of 0.1%–1% of channels is close to saturating the gain. This matches the fact from Figure 1 that outliers are a tiny fraction of channels — the problem was never large in scope, only in magnitude. Locate it precisely and the fix is cheap.

## 4. Side-by-Side Comparison

| Dimension | RTN | GPTQ | SmoothQuant | AWQ |
|---|---|---|---|---|
| **What it changes** | nothing (baseline) | rounding only | distribution (flatten activations) | distribution (enlarge weights) |
| **Needs calibration** | no | yes (Hessian) | yes (act max) | yes (mean\|X\|) |
| **Needs backprop** | no | no | no | no |
| **Target format** | any | W4A16 / W4A8 | **W8A8** | W4A16 |
| **Compute cost** | seconds | minutes–hours | seconds | minutes |
| **Core formula** | `round(W/s)` | `+ H⁻¹ compensation` | `s = amax_X^α / amax_W^(1−α)` | `s = mean\|X\|^α` |
| **Stacks with** | — | AWQ, rotation | GPTQ | GPTQ, rotation |
| **Typical PPL loss (LLaMA-7B)** | W4: 0.5+ | W4: 0.1–0.3 | W8A8: 0.05–0.2 | W4: 0.1–0.25 |

**How to combine them.** The most common production recipe is **AWQ to set the scales, then GPTQ to set the rounding**:

```text
① compute s = mean(|X|)^α
② W ← W·s, X ← X/s   (equivalence transform, absorbed into the norm)
③ run GPTQ on the scaled weights (their distribution is now more uniform,
   so H = 2XᵀX is better conditioned and damp is less critical)
④ fold s into the preceding layer at deployment
```

You get AWQ's salient-channel protection plus GPTQ's error compensation. The cost is that the inference kernel must support both mechanisms, and only some engines (parts of vLLM) have fused them.

## 5. Selection Guidance

<div class="keybox">
<strong>One decision chain:</strong><br>
① Quantize weights only (W4A16, decode bottleneck) → <strong>AWQ</strong> (fast, stable, no tuning); add GPTQ if you need the last 0.1 PPL<br>
② Quantize weights and activations (W8A8, prefill bottleneck) → <strong>SmoothQuant</strong>, or FP8 if the hardware has it<br>
③ Have FP8 hardware → just use FP8; E4M3 is far more outlier-tolerant than INT8<br>
④ Need W4A8 or W4A4 → neither of these is enough, go to qt4
</div>

Two closing observations that are easy to miss.

**SmoothQuant's role in the FP8 era is diminishing.** E4M3 has a 4-bit exponent and a ±448 range versus INT8's ±127 uniform grid; the format itself absorbs outliers that used to require a transform. Many teams have measured that plain per-tensor FP8 is sufficient. SmoothQuant's remaining niche is INT8-only silicon.

**AWQ's role is growing.** On-device, single-card, memory-constrained deployment is the dominant current demand, and W4A16 is the optimal answer for it. Combined with the fact that AWQ is the most forgiving method to run (no `damp` to blow up, one scalar to search), it is the right default for anyone who does not have a specific reason to deviate.

## 6. Series Navigation

- Previous **qt2 · RTN and GPTQ**: route A, change only the rounding.
- Next **qt4 · QuaRot / SpinQuant / DuQuant**: route C, use rotation to genuinely flatten the outlier — the only route to W4A4.
- **qt5 · KV Cache Quantization and FP4**: the orthogonal compression dimension.
- **qt1 · The Basic Language**: α, granularity, symmetry, and the four-route map.

Related on this site: **op3 (Ascend EagleQBSA)** contains a real instance of this conclusion — quantizing P and V in FP8 per channel because their distributions exceed what INT8 can accommodate; **op2** is the series index; **fw4** covers the engineering differences between W8A8 and W4A16 from the serving-engine perspective.
