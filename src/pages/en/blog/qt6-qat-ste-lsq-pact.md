---
title: "Quantization Notes (6): QAT and Quantization-Aware Training — why STE is unsound, and the two repairs (PACT and LSQ)"
description: "In the four-route map from qt1 the D route, change the training, was drawn as a single cell and never expanded. But PTQ has a ceiling: the weights are frozen, so the model never gets a chance to adapt to quantization noise, and W4A4 necessarily loses accuracy. QAT lets the model see the error during training, and the price is that the derivative of round is zero almost everywhere — hence STE, a biased estimator that pretends the derivative is one. This post explains STE's three known failure modes, how PACT (learn the clipping bound alpha) and LSQ (learn the step size s, with a 1/sqrt(N times q_max) gradient scaling) each patch some of them, then gives a complete seven-step pipeline with runnable fake-quant code, and closes with the practical question: QAT is roughly 100x more expensive, so when is it actually worth it."
pubDate: 2026-10-27
series: Quantization Notes
lang: en
altLang: zh
altHref: /blog/qt6-qat-ste-lsq-pact
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

On the four-route map from qt1, routes A (change the rounding), B (change the distribution via scaling) and C (change the distribution via rotation) each got a full post in qt2 through qt4. Only **D, change the training**, has been a single empty box ever since. This post fills it in.

The first five posts share a premise that was never stated outright: **the weights are frozen.** RTN rounds element by element, GPTQ uses the Hessian to spread error onto other columns, AWQ shifts the distribution, QuaRot flattens the outliers. Every one of them performs error minimization **on the already-trained model**. The model itself never gets a say — it cannot say "let me express the same function a different way, just do not cut the channels I am most sensitive about."

QAT (Quantization-Aware Training) inverts this. It inserts the quantization operator into the training graph, lets gradients flow through it, and the model spends tens of thousands of steps learning **how to live inside the grid**. The price is a mathematically awkward obstacle:

```text
The derivative of round(x) is zero almost everywhere.
Take it literally and the gradient is all zeros: the network does not move.
```

So all of QAT rests on a lie: **pretend the derivative of round is 1.** The trick is called STE (Straight-Through Estimator), it dates from 1993, and it is still the foundation of all low-bit training. It is **biased**, **unsound**, and **known to cause three specific failures** — and PACT and LSQ are, in essence, repairs for subsets of those three.

<div class="keybox">
<strong>Where this post sits:</strong> the expansion of route D from the qt1 map. qt2 through qt5 are all PTQ; this is the only post where <strong>the weights move</strong>. <strong>The verdict up front:</strong> W8 does not need QAT (PTQ is already near-lossless); on W4A16, AWQ and GPTQ are good enough that QAT has low marginal value; <strong>only when you need W4A4 and cannot accept the PTQ gap is QAT the only option</strong> — but it is also how most shipping int4 models for edge devices are actually produced (Gemma 3 QAT, Llama 3.2 1B and 3B).
</div>

## 1. First, See Where the PTQ Ceiling Is

No mysticism — start with a curve. The same LLaMA-2-7B, and how perplexity behaves as bit width falls, for all four routes:

<div class="fig">
<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Perplexity delta of four routes across bit widths">
  <text x="20" y="22" font-size="14" font-weight="700" fill="#1a1a1a">The lower the bit width, the wider the PTQ/QAT gap — 4 bit is the watershed, 2 bit is the cliff</text>
  <text x="20" y="40" font-size="11" fill="#6b7280">Y axis = perplexity delta (lower is better). Magnitudes are indicative, not exact reproductions.</text>

  <!-- axes -->
  <line x1="76" y1="70" x2="76" y2="272" stroke="#9ca3af" stroke-width="1.2"/>
  <line x1="76" y1="272" x2="630" y2="272" stroke="#9ca3af" stroke-width="1.2"/>

  <!-- y ticks -->
  <g font-size="10" fill="#6b7280" text-anchor="end">
    <text x="68" y="76">+2.0</text>
    <text x="68" y="126">+1.5</text>
    <text x="68" y="176">+1.0</text>
    <text x="68" y="226">+0.5</text>
    <text x="68" y="275">0</text>
  </g>
  <g stroke="#e5e7eb" stroke-width="1">
    <line x1="76" y1="70" x2="630" y2="70"/>
    <line x1="76" y1="120" x2="630" y2="120"/>
    <line x1="76" y1="170" x2="630" y2="170"/>
    <line x1="76" y1="220" x2="630" y2="220"/>
  </g>

  <!-- x ticks -->
  <g font-size="11" fill="#374151" text-anchor="middle">
    <text x="140" y="292">W8</text>
    <text x="300" y="292">W6</text>
    <text x="460" y="292">W4</text>
    <text x="600" y="292">W2</text>
  </g>
  <text x="360" y="312" font-size="11" fill="#6b7280" text-anchor="middle">Weight bit width (activations dropped in step)</text>

  <!-- RTN: worst -->
  <polyline points="140,262 300,232 460,120 600,32" fill="none" stroke="#dc2626" stroke-width="2.4"/>
  <circle cx="140" cy="262" r="3.6" fill="#dc2626"/><circle cx="300" cy="232" r="3.6" fill="#dc2626"/>
  <circle cx="460" cy="120" r="3.6" fill="#dc2626"/><circle cx="600" cy="32" r="3.6" fill="#dc2626"/>

  <!-- GPTQ / AWQ -->
  <polyline points="140,266 300,248 460,168 600,74" fill="none" stroke="#ea580c" stroke-width="2.4" stroke-dasharray="6 4"/>
  <circle cx="140" cy="266" r="3.6" fill="#ea580c"/><circle cx="300" cy="248" r="3.6" fill="#ea580c"/>
  <circle cx="460" cy="168" r="3.6" fill="#ea580c"/><circle cx="600" cy="74" r="3.6" fill="#ea580c"/>

  <!-- rotation QuaRot -->
  <polyline points="140,267 300,256 460,196 600,110" fill="none" stroke="#2563eb" stroke-width="2.4" stroke-dasharray="2 3"/>
  <circle cx="140" cy="267" r="3.6" fill="#2563eb"/><circle cx="300" cy="256" r="3.6" fill="#2563eb"/>
  <circle cx="460" cy="196" r="3.6" fill="#2563eb"/><circle cx="600" cy="110" r="3.6" fill="#2563eb"/>

  <!-- QAT: best -->
  <polyline points="140,269 300,264 460,244 600,190" fill="none" stroke="#15803d" stroke-width="3"/>
  <circle cx="140" cy="269" r="4.2" fill="#15803d"/><circle cx="300" cy="264" r="4.2" fill="#15803d"/>
  <circle cx="460" cy="244" r="4.2" fill="#15803d"/><circle cx="600" cy="190" r="4.2" fill="#15803d"/>

  <!-- highlight at 4 bit -->
  <line x1="460" y1="70" x2="460" y2="272" stroke="#9ca3af" stroke-width="1" stroke-dasharray="3 3"/>
  <rect x="380" y="52" width="160" height="18" rx="3" fill="#fef3c7"/>
  <text x="390" y="65" font-size="10" fill="#92400e" font-weight="700">Widest split at 4 bit: 1.0 vs 0.3</text>

  <!-- legend -->
  <g font-size="11">
    <line x1="76" y1="322" x2="102" y2="322" stroke="#15803d" stroke-width="3"/>
    <text x="108" y="325" fill="#15803d" font-weight="700">QAT (model moves)</text>
    <line x1="248" y1="322" x2="274" y2="322" stroke="#2563eb" stroke-width="2.4" stroke-dasharray="2 3"/>
    <text x="280" y="325" fill="#2563eb">QuaRot (rotation)</text>
    <line x1="410" y1="322" x2="436" y2="322" stroke="#ea580c" stroke-width="2.4" stroke-dasharray="6 4"/>
    <text x="442" y="325" fill="#ea580c">GPTQ / AWQ</text>
    <line x1="558" y1="322" x2="584" y2="322" stroke="#dc2626" stroke-width="2.4"/>
    <text x="590" y="325" fill="#dc2626">RTN</text>
  </g>
</svg>
<p class="cap">Fig 1. The four routes as bit width falls. At W8 the lines basically coincide (PTQ already suffices); the watershed is W4 and the cliff is W2. The QAT curve is the flattest, because it is not approximating the original model better — it is training a new model that suits the grid.</p>
</div>

Three readings matter here:

**① W8 does not justify QAT at all.** At W8, PTQ adds 0.01 to 0.05, which sits inside evaluation noise. Spending 100x the compute for an unmeasurable gain is pure waste.

**② On W4A16, QAT has low marginal value.** AWQ and GPTQ already push the delta down to 0.1 to 0.3, and QAT might halve that again — but the price is going from a few minutes of calibration to hours or days of training. It only pays back when the **accuracy budget is extremely tight** (an edge model, an AB comparison against the FP16 release, users with zero tolerance for quality loss).

**③ W4A4 is QAT's home ground.** As covered in qt4, on W4A4 QuaRot moves LLaMA-2-7B from 5.47 to 6.10 (a delta of 0.63), SpinQuant to 5.86, DuQuant to 5.98. QAT on W4A4 can push the delta into the 0.1 to 0.2 range — **an interval that no PTQ route reaches, even with A, B and C combined**.

Why? Because all three PTQ routes do essentially the same thing: **find a better (s, z) and better rounding decisions without touching the model function**. But 4 bit has only 15 levels (symmetric int4). That is the entire expressive budget; no matter how you allocate resolution you hit a wall. QAT attacks a different wall — it lets the model **change the distribution of its own weight values**, so that "the things that matter" get expressed with less information. That is a difference in degrees of freedom, not a difference in tuning.

## 2. STE: The Foundation of QAT, and Its Least Sound Step

### 2.1 The problem: the derivative of round does not exist

The simplest possible quantization function, forward direction:

```python
def quantize_fwd(x, s, qmax):
    """Symmetric quantization; s is the step size, qmax = 2**(b-1) - 1."""
    q = torch.clamp(torch.round(x / s), -qmax - 1, qmax)   # (1) scale (2) round (3) clip
    return q * s                                            # (4) dequantize
```

`round` looks like this staircase:

```text
round(x)
  3 |        +-------
  2 |    +---+
  1 |+---+
  0 +
    +----+----+----+---> x
       0.5  1.5  2.5
```

Apart from the points 0.5, 1.5, 2.5 where the derivative does not exist (or is an impulse), **the derivative is zero everywhere**. In PyTorch the backward pass of `round` simply returns 0.

What does that mean? It means that when backpropagation reaches `round`, every gradient is zeroed out — **no upstream parameter (including s, including the weights further up) receives any signal at all**. QAT dies at the first step.

### 2.2 What STE does: pretend the derivative is 1

Since the true derivative is zero, we do not use it. **Treat round as the identity function during backpropagation:**

```python
class STERound(torch.autograd.Function):
    """Forward = round. Backward = pass the gradient straight through."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # The true gradient is zero almost everywhere; return grad_output directly,
        # which is exactly equivalent to asserting d(round)/dx := 1
        return grad_output
```

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="STE forward and backward">
  <text x="20" y="22" font-size="14" font-weight="700" fill="#1a1a1a">STE: the forward pass walks the real staircase, the backward pass takes a straight fake road</text>

  <!-- forward -->
  <text x="40" y="52" font-size="12" font-weight="700" fill="#1d4ed8">Forward: honest computation, round really happens</text>
  <g>
    <rect x="34" y="66" width="88" height="34" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="78" y="88" font-size="12" fill="#1e40af" text-anchor="middle">x / s</text>
    <line x1="122" y1="83" x2="152" y2="83" stroke="#6b7280" stroke-width="1.6" marker-end="url(#ar1)"/>

    <rect x="154" y="66" width="88" height="34" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="198" y="88" font-size="12" fill="#1e40af" text-anchor="middle">round(.)</text>
    <line x1="242" y1="83" x2="272" y2="83" stroke="#6b7280" stroke-width="1.6" marker-end="url(#ar1)"/>

    <rect x="274" y="66" width="88" height="34" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="318" y="88" font-size="12" fill="#1e40af" text-anchor="middle">clip(.)</text>
    <line x1="362" y1="83" x2="392" y2="83" stroke="#6b7280" stroke-width="1.6" marker-end="url(#ar1)"/>

    <rect x="394" y="66" width="88" height="34" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="438" y="88" font-size="12" fill="#1e40af" text-anchor="middle">x s</text>
    <line x1="482" y1="83" x2="512" y2="83" stroke="#6b7280" stroke-width="1.6" marker-end="url(#ar1)"/>
    <text x="540" y="88" font-size="12" fill="#1e40af" font-weight="700">y-hat</text>
  </g>

  <line x1="40" y1="122" x2="640" y2="122" stroke="#e5e7eb" stroke-width="1.5"/>

  <!-- backward -->
  <text x="40" y="148" font-size="12" font-weight="700" fill="#b91c1c">Backward: bypass round entirely, gradient goes straight through</text>
  <g>
    <text x="40" y="188" font-size="11" fill="#b91c1c" font-weight="700">dL/dy</text>
    <line x1="104" y1="184" x2="176" y2="184" stroke="#b91c1c" stroke-width="1.6" marker-end="url(#ar2)"/>
    <text x="140" y="176" font-size="10" fill="#6b7280" text-anchor="middle">x s</text>

    <line x1="176" y1="184" x2="248" y2="184" stroke="#b91c1c" stroke-width="1.6" marker-end="url(#ar2)"/>
    <text x="212" y="176" font-size="10" fill="#6b7280" text-anchor="middle">clip'</text>

    <!-- bypassed round -->
    <rect x="250" y="166" width="96" height="34" rx="5" fill="none" stroke="#d1d5db" stroke-width="1.4" stroke-dasharray="4 3"/>
    <text x="298" y="188" font-size="12" fill="#9ca3af" text-anchor="middle" text-decoration="line-through">round</text>
    <path d="M250 190 Q298 226 346 190" fill="none" stroke="#b91c1c" stroke-width="2.2"/>
    <text x="298" y="246" font-size="10" fill="#b91c1c" text-anchor="middle" font-weight="700">gradient detours (derivative is 0, skip it)</text>

    <line x1="348" y1="184" x2="420" y2="184" stroke="#b91c1c" stroke-width="1.6" marker-end="url(#ar2)"/>
    <text x="384" y="176" font-size="10" fill="#6b7280" text-anchor="middle">/ s</text>
    <text x="444" y="188" font-size="11" fill="#b91c1c" font-weight="700">dL/dx</text>
  </g>

  <rect x="498" y="150" width="142" height="72" rx="6" fill="#fef2f2" stroke="#fca5a5"/>
  <text x="510" y="170" font-size="10" fill="#991b1b" font-weight="700">In one line</text>
  <text x="510" y="188" font-size="10" fill="#991b1b">d round(x) / dx := 1</text>
  <text x="510" y="204" font-size="10" fill="#991b1b">biased, but good enough</text>

  <defs>
    <marker id="ar1" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 z" fill="#6b7280"/>
    </marker>
    <marker id="ar2" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 z" fill="#b91c1c"/>
    </marker>
  </defs>
</svg>
<p class="cap">Fig 2. STE forward and backward. The forward pass is honest: round really does quantize, and the quantization error really exists. The backward pass is fabricated: it pretends that staircase is a straight line with slope 1. The feasibility of all QAT rests on this fabrication.</p>
</div>

**Why does this lie work at all?** The intuition: `round(x/s)` differs from `x/s` only by a constant offset within a local region (one grid cell wide), and what we want is for the weights to move in a direction that lowers the loss. When the move is smaller than one cell, the rounding behavior does not change, but the continuous approximation of the output does. The direction STE supplies is right in a statistical sense — it amounts to optimizing a smoothed surrogate objective that treats quantization noise as a fixed offset. So training converges, and the solution it converges to is decent. This is an **extremely successful engineering heuristic**, not a mathematically correct derivation.

### 2.3 STE's three known problems

Being unsound has a price. STE's failure modes are very specific, and each one corresponds to a repair in a later section:

**Problem one: the saturated region should have no gradient, but it does.**

After `clip(x/s, -qmax, qmax)`, if `x/s > qmax` (the element is outside the representable range), the output is the constant `qmax`, and the true derivative with respect to x is **0** — changing it does nothing, it is clipped anyway. But STE's backward pass hands `dL/dy` straight back, which is equivalent to telling the optimizer "this element is still participating normally, adjust it freely".

The consequence is concrete: **outlier elements keep receiving gradients saying "go bigger" or "go smaller", but the loss does not change because of them.** So these elements are either pushed further out (amplifying the outlier) or drift under a wrong signal. And outliers are exactly what quantization is most sensitive to (all of qt3 is about them). This is the first problem PACT addresses.

**Problem two: the gradient magnitude does not scale with s, so the learning rate cannot be reused across bit widths.**

Look at the backward path: starting from `dL/dy`, multiply by s, skip round, divide by s — the two s factors cancel. That means **the gradient with respect to s gets no correct magnitude information under naive STE**. Worse, when bit width drops from 8 to 4, the same relative change in s matters far more, yet STE returns the same gradient magnitude — so at 4 bit you have to re-tune the learning rate. This is the core problem LSQ solves.

**Problem three: optimizer state is built on a wrong gradient.**

Adam maintains a first and second moment. If `m` and `v` accumulate STE's pseudo-gradient rather than the true gradient, then the adaptive scaling factor `1/sqrt(v)` loses its statistical meaning. In practice this makes the loss noticeably noisier in early QAT than in ordinary training, and makes training sensitive to `eps`.

<div class="warnbox">
<strong>A common misjudgment:</strong> the QAT loss curve looks normal and decreases smoothly, therefore quantization is fine. <strong>Wrong.</strong> STE's bias lets the model learn to "route around the error on the training distribution", and that routing typically does not generalize. qt2 already noted that GPTQ has to be validated on MMLU and GSM8K, not just PPL; QAT is worse — <strong>PPL barely moves while GSM8K drops 8 points</strong> is the classic QAT failure shape, because multi-step reasoning is most sensitive to the accumulation of small per-step errors. Validation has to include task-level metrics.
</div>

## 3. PACT: Make the Clipping Bound a Learnable Parameter

PACT (PArameterized Clipping acTivation, 2018) starts from a direct observation: since STE's gradient is wrong in the saturated region, **make "where the saturated region begins" itself learnable**.

### 3.1 The method

For activations (the non-negative distribution after ReLU), introduce a learnable parameter alpha as the clipping upper bound:

```python
class PACT(nn.Module):
    """Parameterized clipping. alpha is learnable and finds its own clipping point."""

    def __init__(self, init_alpha=8.0, qmax=7):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.qmax = qmax

    def forward(self, x):
        # Forward: clip to [0, alpha], then quantize using a scale derived from alpha
        alpha = self.alpha.clamp(min=1e-6)          # prevent alpha collapsing to 0
        x_clip = torch.clamp(x, 0.0, alpha)
        s = alpha / self.qmax                       # scale comes from alpha, not from absmax
        q = torch.clamp(torch.round(x_clip / s), 0, self.qmax)
        return q * s
```

The key line is `s = alpha / qmax`: **the scale is no longer a function of absmax, it is a function of alpha.** Move alpha and the whole quantization grid moves with it.

### 3.2 Why alpha's gradient is non-zero only on clipped samples

Differentiate with respect to alpha:

```text
y = clip(x, 0, alpha)

dy/dalpha = 1   if x > alpha   (clipped sample, pinned at the ceiling)
          = 0   if x <= alpha  (unclipped sample, output independent of alpha)
```

**Only clipped samples can pull on alpha.** This property is elegant, and it gives alpha's update the correct semantics automatically:

- If alpha is too small, many samples are clipped and they keep pushing alpha up, so alpha grows;
- If alpha is too large, almost nothing is clipped, alpha receives no gradient, and it stops growing;
- The equilibrium is computed by the loss itself: **the optimal trade-off between clipping more samples (giving up resolution for dynamic range) and keeping more resolution (giving up dynamic range).**

Compare this with absmax: absmax is **uniquely determined by a single outlier** (as qt2 showed, one value at 118 occupies the entire level at 127). PACT replaces it with "a statistical compromise point chosen by the loss" — either the outlier gets cut off (losing a little information but buying resolution for everyone), or alpha grows to keep it (at the cost of a coarser grid for everyone). **Delegating that decision to the optimizer is far more sensible than delegating it to max().**

### 3.3 What PACT actually buys, and where it stops

PACT is standard for W4A4 in vision models (MobileNet, the ResNet family) and clearly beats absmax there. Moving it to LLMs requires two adjustments:

**① Use it for activations only, not for weights.** Weight distributions are symmetric and heavy-bodied, without the one-sided long tail that ReLU produces, so PACT's one-sided ceiling semantics do not apply. Use LSQ (next section) for symmetric quantization.

**② SwiGLU's output is not non-negative.** LLaMA-family FFNs use SwiGLU and the intermediate activation has both signs. PACT's original `clip(x, 0, alpha)` has to become the symmetric `clip(x, -alpha, alpha)`, or you simply use LSQ's symmetric parameterization.

## 4. LSQ: Learn the Step Size, and That 1/sqrt(N * q_max)

LSQ (Learned Step Size Quantization, ICLR 2020) is the one more commonly used in LLM quantization. It does not learn alpha; it learns **s** directly. Its core contribution is a **gradient scaling factor** that keeps s's gradient at a sensible magnitude across bit widths and layer sizes.

### 4.1 Deriving the gradient with respect to s

Forward: `v-hat = s * round(clip(v/s, -Q_N, Q_P))`. Differentiate with respect to s, treating `round` per STE:

```text
dv-hat/ds = round(v/s) - v/s      (when -Q_N < v/s < Q_P)
          = -Q_N                  (when v/s <= -Q_N, lower saturation)
          = Q_P                   (when v/s >= Q_P, upper saturation)
```

Note the middle term `round(v/s) - v/s` — it is exactly **the negation of the normalized rounding error**, in the range `[-0.5, 0.5]`. The structure is very interpretable: **s's gradient is proportional to how far this element was rounded off.** A large offset means this s is not right and should move; a small offset means leave it alone.

### 4.2 Then multiply by an unremarkable but essential factor

This is where the LSQ paper does something specific: multiply the gradient above by a scaling factor

```text
g = 1 / sqrt(N * Q_P)
```

`N` is the element count of the tensor (for weights, the whole layer; for activations, per token) and `Q_P` is the positive bound (7 for symmetric int4). The two factors have entirely different intents, so take them separately:

**`1/sqrt(N)`: make s's gradient independent of layer size.**

s is a **scalar** but it affects the output of N elements. So `dL/ds = sum_i (dL/dv-hat_i)(dv-hat_i/ds)` — a sum of N terms. If those N terms have random signs (rounding errors do split roughly half positive and half negative), the sum grows like **sqrt(N)**, not like N. Layer widths differ by tens of times (hidden 4096 vs intermediate 11008 vs vocabulary 32000); without this divisor, one learning rate must explode in one layer and stall in another. `1/sqrt(N)` levels that scale out.

**`1/sqrt(Q_P)`: make s's gradient grow as bit width falls.**

Lower bit width means smaller `Q_P`. Dividing by `sqrt(Q_P)` means **s's gradient is larger at low bit width**. This matches intuition: at 4 bit, a slightly wrong s costs far more relative accuracy than at 8 bit, so s should move more aggressively. This factor is also the direct reason LSQ can **share one set of hyperparameters across 2, 3, 4 and 8 bit** — the paper's ablation shows explicitly that removing it forces per-bit-width learning-rate tuning.

### 4.3 Initialization: why not absmax

```text
s_0 = 2 * mean(|v|) / sqrt(Q_P)
```

Use the mean, not the max. The reason is the same as in the PACT section: absmax is held hostage by a single outlier, whereas `mean(|v|)` estimates where the bulk of the distribution sits and is insensitive to outliers. The factor 2 is an empirical compensation (`mean` is below a reasonable typical step), and `/sqrt(Q_P)` stays consistent with the gradient scaling factor.

The cost: if the distribution genuinely has a long tail, this initialization "cuts off" the tail at first and the loss jumps briefly. But since s is learnable, it finds its place within a few hundred steps — **starting from a point where the bulk is sensible beats starting from a point dragged off by outliers.**

<div class="keybox">
<strong>PACT vs LSQ in one line each:</strong><br>
<strong>PACT</strong> learns the <strong>clipping bound alpha</strong>, and the scale is derived from it (`s = alpha/qmax`). Suited to <strong>non-negative activations</strong> (post-ReLU); the semantics are "where to cut".<br>
<strong>LSQ</strong> learns the <strong>step size s</strong> directly, with `1/sqrt(N * q_max)` gradient scaling. Suited to <strong>symmetric distributions</strong> (weights, SwiGLU activations); the semantics are "how wide is a cell".<br>
In practice LLM QAT is mostly an LSQ variant (weights are symmetric, activations have no ReLU), while PACT shows up more in vision models and early work.
</div>

## 5. The Complete Seven-Step Pipeline

With the theory done, down to engineering. A workable LLM QAT pipeline looks like this:

```python
import torch, torch.nn as nn
from torch.optim import AdamW

# ---- Step 1a: define the STE rounding operator (the only "cheat" in this post) ----
class _STERound(torch.autograd.Function):
    """Forward: round plus saturation. Backward: straight through."""

    @staticmethod
    def forward(ctx, x, qmax):
        return torch.round(x).clamp(-qmax - 1, qmax)

    @staticmethod
    def backward(ctx, g):
        return g, None          # pass through to x; qmax is a constant, no gradient


# ---- Step 1b: build the fake-quant module that replaces nn.Linear ----
class QATLinear(nn.Module):
    """Weights use LSQ to learn s; activations use per-token dynamic symmetric quantization."""

    def __init__(self, base: nn.Linear, w_bits=4, a_bits=8, group_size=128):
        super().__init__()
        self.base = base
        self.w_bits, self.a_bits, self.g = w_bits, a_bits, group_size
        self.wqmax = 2 ** (w_bits - 1) - 1
        self.aqmax = 2 ** (a_bits - 1) - 1

        out_f, in_f = base.weight.shape
        n_groups = in_f // group_size
        # Learn one s per group, shape [out_f, n_groups], init at 2*mean(|w|)/sqrt(qmax)
        w = base.weight.data.view(out_f, n_groups, group_size)
        s0 = 2.0 * w.abs().mean(dim=2) / (self.wqmax ** 0.5)
        self.w_scale = nn.Parameter(s0)

    def forward(self, x):
        # -- weight side: per-group LSQ --
        w = self.base.weight
        out_f, in_f = w.shape
        s = self.w_scale.clamp(min=1e-8).unsqueeze(-1)          # [out_f, n_groups, 1]

        w_scaled = w.view(out_f, -1, self.g) / s                 # scale
        w_q = _STERound.apply(w_scaled, self.wqmax)              # STE round plus clip
        w_dq = (w_q * s).view(out_f, in_f)                       # dequantize back to shape

        # -- activation side: per-token dynamic, nothing to learn (absmax is available) --
        a_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.aqmax
        x_q = _STERound.apply(x / a_scale, self.aqmax)
        x_dq = x_q * a_scale

        return nn.functional.linear(x_dq, w_dq, self.base.bias)


# ---- Step 2: train only s (freeze the rest), or full fine-tuning ----
# The choice matters:
#   (a) learn scales only: a few hundred steps, one GPU, almost no accuracy loss -- enough for 95% of cases
#   (b) full fine-tuning: tens of thousands of steps, many GPUs, recovers another 0.1-0.3 PPL -- only for shipping edge releases
for name, p in model.named_parameters():
    p.requires_grad = ('w_scale' in name)     # strategy (a)

# ---- Step 3: optimizer and learning rate ----
# s's gradient already carries the 1/sqrt(N * qmax) scaling, so one LR works across bit widths
opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, betas=(0.9, 0.99))

# ---- Step 4: training loop (a small calibration-plus-training set, 1e8 to 1e9 tokens) ----
for step, batch in enumerate(loader):
    out = model(**batch)
    # The QAT loss is just the ordinary LM loss; adding a KD term makes it more stable
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # STE gradients are biased: clip
    opt.step(); opt.zero_grad()

# ---- Step 5: periodic real-quant evaluation (never trust fake-quant loss alone) ----
# In fake-quant s is a float; at deployment s is stored as fp16, adding another rounding
# So every N steps, run the real int4 kernel and check PPL plus task metrics

# ---- Step 6: export ----
# Store s as fp16; pack w_q into int32 (eight int4 values per int32, see qt2)
# Note: QAT-produced s is distributed differently from PTQ; do not reuse AWQ/GPTQ export scripts

# ---- Step 7: validation (the most important section) ----
# Always test: PPL (wiki-text) + MMLU + GSM8K + HumanEval + 32k needle
# Red line: if GSM8K drops more than 2 points, roll back -- the most common hidden QAT failure
```

Two details here are deliberate:

- **`s` uses `.unsqueeze(-1)` to broadcast over the last dim of the group**, because LSQ's s is a per-group scalar, not a per-element one.
- **`backward` must return two values**: `g` for `x` (pass-through) and `None` for `qmax` (a Python int that never enters the graph). Return one and PyTorch raises a gradient-count mismatch.

### 5.1 Two common traps

**Trap one: when to fold BN or norm.** Vision models must fold BN into the convolution before QAT (otherwise the quantized BN parameters break the scale). LLMs have no BN but do have RMSNorm — and **RMSNorm does not fold into the linear layer** (it acts on activations, it is not an affine transform), so this step simply does not exist for LLMs. What does matter is **not absorbing LayerNorm's gamma into the quantization scale**: gamma is per-channel and will fight the per-token activation scale. Keep gamma in FP16 and quantize only the input of the linear layer that follows it.

**Trap two: do not learn the per-token activation scale.** The weight's s is static (stored at deployment), so it must be learned. The activation's s is computed on the fly per token (`absmax(x)/qmax`, one line), and **making it learnable is actively harmful** — during training it drifts to a fixed value that is optimal for the training distribution but generalizes worse, and at deployment you do not even have "the value you learned" available (it differs per token). This is the same reasoning as in qt5 on KV quantization.

### 5.2 How QAT and PTQ fundamentally differ in validation

PTQ validation is relatively simple: given a calibration set, run PPL plus a few tasks, and the verdict is immediate — and **reproducible** (same checkpoint, same calibration set, same result).

QAT validation is much messier:

| Dimension | PTQ | QAT |
|---|---|---|
| Reproducibility | Fully reproducible | Depends on seed and data order |
| Validation cost | Minutes | Hours (full evaluation suite) |
| Failure shape | PPL rises visibly | **PPL normal, task metrics drop** (silent) |
| Overfitting risk | Overfits the calibration set | Overfits the training set and the calibration set |
| Rollback cost | Rerun with different parameters | Retrain (hours to days) |

**The dangerous row is "failure shape".** During QAT the model sees quantized forwards, so its loss on the training distribution looks great; but STE's biased gradient means what it learned is a **trick for routing around quantization**, not an adaptation to it. Those tricks fall apart on GSM8K, which needs several steps of exact reasoning. Hence the red line: **if GSM8K drops more than 2 points, roll back.**

## 6. The Trade-off: QAT Costs About 100x — When Is It Worth It

One decision table, by scenario:

| Scenario | Recommendation | Reasoning |
|---|---|---|
| **W8A8 deployment** | **Do not use QAT** | PTQ adds 0.01 to 0.05, inside the noise; QAT is waste |
| **W4A16, general serving** | **AWQ or GPTQ** | A delta of 0.1 to 0.3 is acceptable and PTQ takes minutes |
| **W4A16, extremely tight accuracy budget** | QAT (scales only) | Learning only scales takes a few hundred steps on one GPU; reasonable value |
| **W4A4, PTQ gap unacceptable** | **QAT is the only option** | Routes A, B and C combined cannot reach this interval |
| **W2 / ternary / binary** | **QAT is mandatory** | PTQ collapses outright at this width; not a discussion |
| **Shipping int4 edge models** | **QAT** | Gemma 3 QAT and Llama 3.2 1B/3B are produced this way; small models, cheap training, zero user tolerance for quality loss |
| **Frequent model swaps** | Do not use QAT | Retraining on every swap is not sustainable engineering |

<div class="keybox">
<strong>An often-ignored cost item:</strong> QAT's real cost is not GPU hours, it is that <strong>it turns quantization from a step in deployment into a part of the training pipeline</strong>. PTQ can be done after release, independently, by whoever deploys the model (which is exactly why the AWQ/GPTQ ecosystem is so healthy). QAT has to be inserted by the model author during training, and the quantization configuration (bit width, group size, which layers to skip) must be fixed before training starts — <strong>change one bit width and you retrain</strong>. That is an organizational cost, not a compute cost.
</div>

So my verdict on QAT is this: **it is the heavy weapon in the quantization arsenal, not a better default.** The reason the qt1 map puts D last is not that it is weak but that its barrier to use is highest — once you adopt it, the whole chain (training, export, validation, iteration) has to change with it. The three PTQ routes can be plugged in and rolled back at will, and that is the real reason they dominate nearly all production scenarios.

**But conversely:** if your target is W4A4, or you are shipping to edge devices, then all the PTQ techniques in the previous five posts only **minimize the loss**, whereas QAT **retrains a model that suits 4 bit**. Those two are not on the same scale.

## 7. Series Navigation

- Previous: **qt5, KV Cache Quantization and the FP4 Frontier** — an orthogonal compression axis, and how formats rewrite algorithms.
- **qt4, QuaRot / SpinQuant / DuQuant** — route C, rotation for W4A4, the baseline this post keeps comparing against.
- **qt3, SmoothQuant and AWQ** — route B, two answers to activation outliers. PACT answers the same question from a later generation.
- **qt2, RTN and GPTQ** — route A, the baseline for everything; the packing format and validation criteria are reused here.
- **qt1, The Basic Language of Quantization** — the four-route map; this post is the expansion of route D.
- **op2, Model Quantization Overview** — the series guide and lineage table.

Related on this site: **qt7 (The Three Boundaries of Quantization)** takes up the question this post leaves open — does a QAT-tuned model still hold up when stacked with speculative decoding, moved onto an MLA architecture, or run on an edge NPU? **fa4 (DeepSeek V4)** covers quantization-aware training recipes that combine STE with KD; **fw5** covers kernel selection inside inference engines, which determines whether QAT-exported weights can actually be executed efficiently.
