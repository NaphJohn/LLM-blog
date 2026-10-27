---
title: "Quantization Notes (7): The Three Boundaries of Quantization — speculative decoding, MLA, and edge hardware"
description: "qt1 through qt6 all discuss how to minimize error inside quantization. This post looks outward: quantization is not a variable you can optimize in isolation. Boundary one is speculative decoding — 4 bit speeds up single-token forward passes but slows tree verification, so the optimal draft tree shrinks from 60 to about 16 rather than growing. Boundary two is the MLA/GQA structure — KIVI's per-channel K plus per-token V breaks down under MLA, because what the cache stores is not K and V at all. Boundary three is edge hardware — what decides whether a quantized format can run on an NPU is the format, not the bit width. All three lead to the same rule: hardware picks the format, the format picks the algorithm, and the algorithm only has room to optimize once the first two are fixed."
pubDate: 2026-10-27
series: Quantization Notes
lang: en
altLang: zh
altHref: /blog/qt7-quantization-boundaries
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

The previous six posts share an implicit frame: **given a model and a bit width, how do you minimize quantization error**. That is a clean optimization problem — an objective (PPL, or per-layer reconstruction error `||WX - W-hat X-hat||`), constraints (4 bit, group size 128), and methods (RTN, GPTQ, AWQ, rotation, QAT).

But real deployment does not work like that. The moment you finish quantizing a model, it runs into three external constraints:

```text
(1) It has to stack with other acceleration   -> speculative decoding
(2) It has to run on some attention structure -> GQA / MLA
(3) It has to land on some hardware           -> NPU / GPU / datacenter card
```

Each of these three can **turn the "internally optimal" choice into a wrong one** — not slightly worse numbers, but conclusions that flip: the optimal draft tree halves, per-token scales must be abandoned, and whether 4 bit is usable at all depends on whether you picked `Q4_0` or `Q4_K_M`.

<div class="keybox">
<strong>Where this post sits:</strong> qt1 through qt6 discuss <strong>how to optimize inside quantization</strong>; this post discusses <strong>what quantization is constrained by</strong>. <strong>The verdict up front:</strong> quantization is never a variable you can tune independently — it is <strong>a function of three external conditions</strong>. When a paper claims "our 4-bit scheme is lossless", the right question is not "how much PPL did it lose" but "lossless at what tree size, on what attention structure, through what hardware path". Every conclusion in the previous six posts holds <strong>conditionally</strong>, and this post writes those conditions down.
</div>

## 1. Boundary One: Stacked With Speculative Decoding, the Optimal Configuration Flips

### 1.1 Write the speedup formula out first

Speculative decoding's speedup is not "higher acceptance rate means faster". It is governed by a fuller expression:

```text
speedup ~= (alpha * k + 1) / (k * c_target + c_draft)
```

where `alpha` is the acceptance rate, `k` is the draft token count (or tree width), `c_target` is the cost of one target forward pass, and `c_draft` is the cost of one draft forward pass.

Both denominator terms are being added, but they **scale completely differently**: `k * c_target` grows linearly with draft length (verifying k tokens means k target forward passes, though they can be batched), while `c_draft` is a fixed overhead. So the formula hides two independently adjustable knobs:

- **Raise alpha**: the numerator grows, direct gain
- **Push c_draft down**: the denominator shrinks, direct gain

The intuition for quantizing the draft model comes from the second knob — the draft is small, it runs often, so quantizing it should pay. But there are two traps here, one that has been stepped on many times and one that only got quantified clearly in 2025.

### 1.2 Trap one: draft quantization barely hurts acceptance, if you do not mismatch precision families

Start with the good news: **quantizing the draft model to 4 bit barely moves acceptance.**

A measurement in SGLang (PR #39254, a Qwen-family 27B-class model with a DFlash speculation head) gives clean numbers:

| Draft precision | Mean accept length | Acceptance rate |
|---|---|---|
| BF16 | 3.25 | 0.31 |
| W4A16 | 3.23 | 0.32 |

Dropping from 16 to 4 bit costs 0.02 in accept length; the acceptance rate even ticks up (within noise). The reason is not hard: **speculative decoding's acceptance criterion is `u < p_target(x)/p_draft(x)`, a ratio**. When draft and target share the same quantization error, the numerator and denominator biases partly cancel; and even without cancellation, the top-k ordering perturbation that 4 bit introduces is far smaller than sampling noise.

Now the bad news, which is far more serious than it looks:

**Precision families must match the target model, or acceptance collapses to zero.**

The same draft model (the `-DFlash-NVFP4` build), paired with targets of different precision:

| Target precision | Acceptance rate |
|---|---|
| FP8 | **0%** |
| NVFP4 | **~62%** |

Not halved — **completely zero**. This looks absurd: an NVFP4 draft against an FP8 target, where the target is clearly more precise, should be safer. Why does it collapse?

The reason is the **implicit assumption** inside `u < p_target/p_draft`: it assumes the difference between the draft's and the target's probability for a token comes only from "who understands language better", not from "who rounded differently". NVFP4 and FP8 differ by more than precision — their **error structures are entirely different**: NVFP4 is 16-element blocks with an FP8 scale, FP8 is per-tensor or per-channel scaling. The two models may round the same logits vector in opposite directions, so `p_draft` is systematically too large on some tokens and the test `u < p_target/p_draft` rejects en masse.

<div class="warnbox">
<strong>The practical form of this conclusion:</strong> the draft and target must <strong>either both use the same quantization family, or the draft stays in FP16/BF16</strong>. Do not "quantize the draft more aggressively and keep the target at high precision" — that combination looks like a win on paper (smaller denominator, same numerator) and measures zero or negative speedup in practice. <strong>And the metric most likely to fool you is PPL</strong>: an NVFP4 draft has perfectly normal PPL, possibly closer to BF16 than the FP8 build, because it genuinely is the more precise format. Normal PPL and zero acceptance can be true at the same time.
</div>

### 1.3 Trap two: at 4 bit the draft tree should shrink, not grow

This was the most counter-intuitive finding of the research for this post, and the one I most wanted to get into writing.

Background first. The intuition behind tree speculation is "the bigger the tree, the more tokens one verification recovers". Under BF16 that intuition basically holds: as tree width goes from 8 to 60, accept length rises steadily, because the marginal cost of verification (one batched target forward pass) is relatively cheap.

**But quantize the target model to 4 bit and the relationship inverts.**

HierSpec (arXiv 2505.22179) does a nice measurement: it defines a metric called **V/D** (Verification over Decoding time ratio) — the time to verify one draft tree divided by the time to decode one token autoregressively. Ideal speculative decoding should have V/D near 1.0: verifying a tree costs about the same as decoding a token, which is what "free parallelism" means.

Measured (A100 with Llama-3-8B):

| Model precision | Tree size | V/D |
|---|---|---|
| BF16 | 60 | ~1.0 (acceptable) |
| **W4A16** | **60** | **1.8** (already losing) |
| **W4A16** | **~16** | ~1.0 |

Same tree size of 60: V/D is reasonable under BF16 and rises to 1.8 under 4 bit — **verification has become the bottleneck** and the speedup falls instead of rising.

The reasons are worth unpacking, because they expose a boundary condition on "quantization gains":

<div class="fig">
<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Optimal draft tree size shifts left after 4-bit quantization">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">Same 4 bit: single-token forward is 2x faster, tree verification only 1.15x</text>
  <text x="20" y="42" font-size="11" fill="#6b7280">Y axis = V/D (verify one tree / decode one token), ideal 1.0; X axis = draft tree size</text>

  <!-- axes -->
  <line x1="70" y1="66" x2="70" y2="250" stroke="#9ca3af" stroke-width="1.2"/>
  <line x1="70" y1="250" x2="640" y2="250" stroke="#9ca3af" stroke-width="1.2"/>

  <!-- y ticks -->
  <g font-size="10" fill="#6b7280" text-anchor="end">
    <text x="62" y="72">2.4</text>
    <text x="62" y="112">1.8</text>
    <text x="62" y="152">1.2</text>
    <text x="62" y="192">0.6</text>
  </g>
  <g stroke="#e5e7eb" stroke-width="1">
    <line x1="70" y1="68" x2="640" y2="68"/>
    <line x1="70" y1="108" x2="640" y2="108"/>
    <line x1="70" y1="148" x2="640" y2="148"/>
    <line x1="70" y1="188" x2="640" y2="188"/>
  </g>

  <!-- V/D = 1.0 line -->
  <line x1="70" y1="128" x2="640" y2="128" stroke="#15803d" stroke-width="1.4" stroke-dasharray="6 4"/>
  <text x="536" y="122" font-size="10.5" fill="#15803d" font-weight="700">V/D = 1.0 ideal</text>

  <!-- x ticks -->
  <g font-size="10.5" fill="#374151" text-anchor="middle">
    <text x="110" y="268">8</text>
    <text x="200" y="268">16</text>
    <text x="290" y="268">24</text>
    <text x="380" y="268">32</text>
    <text x="470" y="268">44</text>
    <text x="560" y="268">60</text>
  </g>
  <text x="355" y="288" font-size="11" fill="#6b7280" text-anchor="middle">Draft tree size (tokens verified in one pass)</text>

  <!-- BF16 curve: gentle, ~1.0 at 60 -->
  <polyline points="110,196 200,180 290,166 380,152 470,140 560,130"
            fill="none" stroke="#2563eb" stroke-width="2.6"/>
  <circle cx="110" cy="196" r="3.6" fill="#2563eb"/><circle cx="200" cy="180" r="3.6" fill="#2563eb"/>
  <circle cx="290" cy="166" r="3.6" fill="#2563eb"/><circle cx="380" cy="152" r="3.6" fill="#2563eb"/>
  <circle cx="470" cy="140" r="3.6" fill="#2563eb"/><circle cx="560" cy="130" r="4.4" fill="#2563eb"/>
  <text x="572" y="196" font-size="11" fill="#2563eb" font-weight="700">BF16</text>
  <text x="428" y="140" font-size="10" fill="#2563eb">bigger tree pays</text>

  <!-- W4A16 curve: steeper, 1.8 at 60 -->
  <polyline points="110,214 200,168 290,142 380,122 470,110 560,98"
            fill="none" stroke="#dc2626" stroke-width="3"/>
  <circle cx="110" cy="214" r="4" fill="#dc2626"/><circle cx="200" cy="168" r="4" fill="#dc2626"/>
  <circle cx="290" cy="142" r="4" fill="#dc2626"/><circle cx="380" cy="122" r="4" fill="#dc2626"/>
  <circle cx="470" cy="110" r="4" fill="#dc2626"/><circle cx="560" cy="98" r="5" fill="#dc2626"/>
  <text x="572" y="102" font-size="11" fill="#dc2626" font-weight="700">W4A16</text>

  <!-- warning at tree 60 -->
  <line x1="560" y1="98" x2="560" y2="250" stroke="#dc2626" stroke-width="1" stroke-dasharray="3 3"/>
  <rect x="386" y="78" width="166" height="20" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <text x="394" y="92" font-size="10" fill="#b91c1c" font-weight="700">At tree 60, V/D = 1.8: verification is the bottleneck</text>

  <!-- optimal point moves left -->
  <rect x="140" y="184" width="126" height="20" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="148" y="198" font-size="10" fill="#15803d" font-weight="700">4-bit optimum moves left to ~16</text>

  <!-- bottom note -->
  <line x1="20" y1="302" x2="660" y2="302" stroke="#e5e7eb" stroke-width="1"/>
  <text x="20" y="322" font-size="11" fill="#374151">Why: 4 bit only accelerates contiguous-access GEMM; tree verification is scattered KV access plus tree masking plus variable-length batching, so it gets no bandwidth dividend</text>
</svg>
<p class="cap">Fig 1. Quantization made single-token forward 2x faster but tree verification only 1.15x faster. When two speeds do not scale proportionally, the optimum must move — so a 4-bit draft tree should be smaller than a BF16 one, not larger.</p>
</div>

Three reasons, all pointing at the same thing:

1. **Tree verification's KV access is not contiguous.** The n candidates in a tree share a prefix, but attention has to gather non-contiguous KV pages according to a tree mask. Quantization delivers a **bandwidth dividend**, whereas the bottleneck in gather access is latency and indexing, not bandwidth.
2. **Tree masking plus variable-length batching degrades kernel efficiency.** Fast 4-bit kernels (Marlin, Machete) target **large fixed-shape matrix multiplies**. Tree verification feeds them a shape of `[num_nodes, hidden]` where `num_nodes` varies with tree structure and carries a custom attention mask — so it lands on the slow path.
3. **The dequantization overhead is amortized differently.** 4-bit weights must be dequantized before use, and that cost is fixed per element. During single-token decode the matrix multiply is large and amortization is fine; during tree verification the effective compute per node is small, so dequantization's relative share rises.

HierSpec's solution is elegant and worth remembering as an idea rather than only as numbers: since the bottleneck is that verifying a large tree with the big model does not pay, insert an **8B W4A16 intermediate model** to absorb tree verification and let the 70B target do only **sequential** verification (one token per step, fixed shape, fast path). Result: **98.19 tok/s, 2.78x over autoregressive and 1.31x over EAGLE-2**.

> **The general form of this boundary**: any combination of "quantization plus another form of parallel acceleration" requires measuring the two paths' speedups separately. If both accelerations act on the same stretch of computation, the gains overlap (1.5x times 1.5x is not 2.25x); if they act on different stretches, the bottleneck moves. Quantization accelerates **fixed, large, contiguous** computation, whereas tree-structured verification is precisely **dynamic, fragmented, gather-style** computation. That is the structural reason the two do not stack.

### 1.4 Two real incidents: acceptance rate is the only acceptance criterion

There are two classes of engineering incident where PPL and loss look completely normal and only acceptance collapses.

**Incident one: in vLLM a 4-bit draft is silently bit-cast to BF16.** Even if you supply 4-bit draft weights, vLLM by default still builds the model with an unquantized configuration and then **bit-casts** the 4-bit tensor to BF16 — producing not a numerically approximate model but a **numerically wrong** one. Symptom: acceptance rate 1.03 (accepting only the bonus token, equivalent to no speculation), zero speedup. Fix: set `"quantization": "modelopt_mixed"` explicitly.

**Incident two: in SGLang `fc.weight_packed` is silently skipped by `load_weights()`.** The quantized weight is named `fc.weight_packed` (packed int32), while the loader only recognizes `fc.weight`. Unmatched keys **do not raise, they are skipped**, so `fc` keeps its random initialization. Symptom: acceptance collapses to only the bonus token.

<div class="warnbox">
<strong>The shared lesson: errors in the quantization chain do not show up in the loss, only in the acceptance rate.</strong> A randomly initialized fc layer, if it sits on only a small branch of the draft model, barely affects overall loss; a bit-cast weight, if its values still land in plausible ranges (read int32 bit patterns as BF16 and many values fall in extremely small or large exponent bands), may only make PPL "a bit worse". <strong>So validation for speculative decoding plus quantization must treat mean accept length and acceptance rate as first-class metrics</strong>, on par with PPL, rather than an afterthought check.<br><br>
Concretely: run a fixed prompt batch and record both the mean and the distribution of `accept_length`. A healthy 4-bit draft should be at 95 percent or more of the BF16 value (for example 3.23 versus 3.25). If it drops to 1.x, check whether the weights were actually loaded — do not suspect the algorithm first.
</div>

## 2. Boundary Two: Under MLA / GQA, the KV Quantization Rules Change

The KV quantization rules from qt5 (per-channel for K, per-token for V) carry an implicit premise: **the cache stores K and V, and they are two separate tensors**. That premise barely holds under GQA and fails outright under MLA.

### 2.1 GQA: fewer statistical samples, wider error impact

GQA (Grouped-Query Attention) lets multiple query heads share one set of KV heads. Look at how the per-channel scale statistics change:

```text
MHA: per-channel scale samples = n_h * d_h        (each head independent)
GQA: per-channel scale samples = n_kv_heads * d_h (shared)
```

For Llama-3-8B: `n_h = 32`, `n_kv_heads = 8` (GQA 4:1). So:

- **Statistical samples drop from 32 groups to 8** — each scale must be estimated from more data, raising estimation variance (not that serious, since the token dimension still averages)
- **The real problem is that error impact widens 4x** — one KV head's quantization error now affects the output of 4 query heads

But GQA has a **compensating effect in the other direction**, which is rarely mentioned: K's outliers are the **union of what several query heads need**. Under MHA each head has its own set of outlier channels, and the per-channel scale must hug that head's sharp structure; under GQA, the attention patterns of 8 query heads are averaged into one KV head, so **the outliers are actually less sharp**.

The corollary: **per-channel is easier to get right under GQA, but per-token is more dangerous** — because per-token takes one scale for all channels of this token in this KV head, and after sharing that granularity is too coarse: it cannot distinguish "only 1 of these 4 query heads needs that big channel value". qt5's conclusion that "K should use per-channel" not only still holds under GQA, it holds more strongly than under MHA.

### 2.2 MLA: what the cache stores is not K and V at all

This is the core of this section. MLA (Multi-head Latent Attention, used by DeepSeek V2/V3) compresses KV into a low-rank latent, and the cache stores only:

| Content | Dimension | Note |
|---|---|---|
| `c_KV` | 512 | the compressed KV latent; both K and V are projected from it |
| `k_R` | 64 | the decoupled RoPE part (decoupled RoPE key) |
| **Total** | **576** | versus MHA-7B's `2 * 32 * 128 = 8192`, a **14x reduction** |

<div class="fig">
<svg viewBox="0 0 680 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="KV cache layout across MHA, GQA and MLA">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">What the cache actually stores, across three structures</text>
  <text x="20" y="42" font-size="11" fill="#6b7280">7B-class model as the example; box heights are not to scale, the labels give the real element counts</text>

  <!-- MHA -->
  <text x="20" y="76" font-size="12" font-weight="700" fill="#1a1a1a">MHA: K and V are two independent tensors</text>
  <rect x="20" y="86" width="150" height="30" rx="3" fill="#dbeafe" stroke="#93c5fd"/>
  <text x="95" y="105" font-size="11" fill="#1e40af" text-anchor="middle" font-weight="700">K  [32, 128]</text>
  <rect x="180" y="86" width="150" height="30" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="255" y="105" font-size="11" fill="#15803d" text-anchor="middle" font-weight="700">V  [32, 128]</text>
  <text x="348" y="105" font-size="11" fill="#6b7280">= 8192 per token</text>
  <text x="20" y="134" font-size="10.5" fill="#374151">-> Strategy: per-channel for K (structural outliers), per-token for V (transient outliers)</text>
  <text x="20" y="150" font-size="10.5" fill="#15803d">-> Independent tensors, so each can use a different strategy; KIVI's split holds</text>

  <line x1="20" y1="166" x2="660" y2="166" stroke="#e5e7eb" stroke-width="1"/>

  <!-- GQA -->
  <text x="20" y="190" font-size="12" font-weight="700" fill="#1a1a1a">GQA: fewer KV heads than query heads (Llama-3-8B is 4:1)</text>
  <rect x="20" y="200" width="62" height="30" rx="3" fill="#dbeafe" stroke="#93c5fd"/>
  <text x="51" y="219" font-size="10" fill="#1e40af" text-anchor="middle">K[8,128]</text>
  <rect x="90" y="200" width="62" height="30" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="121" y="219" font-size="10" fill="#15803d" text-anchor="middle">V[8,128]</text>
  <text x="162" y="219" font-size="11" fill="#6b7280">= 2048 per token (4x reduction)</text>
  <text x="20" y="248" font-size="10.5" fill="#374151">-> Samples drop from 32 groups to 8; one KV head's error affects 4 query heads</text>
  <text x="20" y="264" font-size="10.5" fill="#15803d">-> But outliers are the union over 4 query heads, so less sharp -> per-channel is easier</text>

  <line x1="20" y1="280" x2="660" y2="280" stroke="#e5e7eb" stroke-width="1"/>

  <!-- MLA -->
  <text x="20" y="304" font-size="12" font-weight="700" fill="#1a1a1a">MLA: the cache stores a latent; both K and V are projections of it</text>
  <rect x="20" y="314" width="176" height="30" rx="3" fill="#fef3c7" stroke="#fcd34d"/>
  <text x="108" y="333" font-size="11" fill="#92400e" text-anchor="middle" font-weight="700">c_KV  [512]</text>
  <rect x="206" y="314" width="52" height="30" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <text x="232" y="333" font-size="10" fill="#b91c1c" text-anchor="middle" font-weight="700">k_R[64]</text>
  <text x="270" y="333" font-size="11" fill="#6b7280">= 576 per token (14x reduction)</text>
  <text x="20" y="362" font-size="10.5" fill="#b91c1c">-> K and V are two projections of one tensor: you cannot do per-channel for K and per-token for V</text>
</svg>
<p class="cap">Fig 2. Under MHA, K and V are independent, so KIVI's split holds. Under GQA they are shared but still separate tensors, so the conclusion holds even more strongly. Under MLA there is only one latent, and every strategy that treats K and V differently breaks down.</p>
</div>

### 2.3 Special case one: KIVI's split fails outright

qt5 spent a whole section on "per-channel for K, per-token for V", reasoning that K's outliers concentrate structurally along the channel dimension while V's appear transiently along the token dimension.

Under MLA that reasoning's **premise disappears**: `K = c_KV * W_UK` and `V = c_KV * W_UV` are projections of the same `c_KV`, and the cache holds only `c_KV`. You cannot say "the first half of this tensor takes per-channel and the second half takes per-token" — it is one whole.

The correct approach is **per-channel throughout**. The reason is `c_KV`'s structure: it comes from an SVD-style low-rank decomposition (down-projected by `W_DKV`, up-projected by `W_UK` and `W_UV`), and that structure **concentrates variance into a few leading latent dimensions**. In other words, outliers concentrate structurally along the channel dimension (the 512 latent dims) rather than transiently along tokens — exactly the sufficient condition for per-channel.

By the same token, qt5's per-token recommendation on the V side loses its basis: V's "transience" comes from the sparsity of the softmax output (one token's attention weights concentrated in a few positions), but under MLA `V` is a linear transform of `c_KV`, whose magnitude structure is set by the latent rather than by softmax.

### 2.4 Special case two: `k_R` must never be quantized

`k_R` (64 dims, 11 percent of the cache) is the decoupled RoPE key. Two of its properties make quantization extremely risky:

**Property one: it participates in RoPE, so quantization error becomes phase error.** RoPE rotates `k_R`: `k_R * [cos(m*theta), sin(m*theta)]`. If `k_R` carries a quantization error `delta_k_R`, that error is **moved into the orthogonal component** by the rotation, and then:

```text
phase error in scores ~= projection of delta_k_R onto the (cos, sin) basis
accumulates with position distance m (since theta_m = m * theta_0 grows linearly)
```

The result is **score error that accumulates linearly or worse with distance at long context** — invisible on short prompts, catastrophic past 32k. This class of error is as well hidden as the softmax exponential amplification discussed in qt5.

**Property two: a per-token scale destroys the linear decomposability of relative position.** RoPE's central property is that relative position encoding can be written as an inner product, which depends on `k` and `q` rotating in the same `(cos, sin)` basis. If each token uses its own scale, then `s_t * (k * rot(m*theta))` is not equal to `(s_t * k) * rot(m*theta)` — quantization and rotation do not commute (this is qt4's "rotation is not free" boundary reappearing on the KV side).

**Conclusion: keep `k_R` in FP16.** The payoff is 11 percent and the risk is breaking long context. That is a clearly asymmetric bet with no room for discussion.

### 2.5 Special case three: quantization error gets remixed by `W_UK`

This is the most technical point, and the one general-purpose tools most easily miss.

Under MHA, `delta_K`'s effect on scores is direct — `delta_scores = Q * delta_K^T / sqrt(d)`, with channels independent. Under MLA, what gets quantized in the cache is `c_KV`, and the error must pass through the up-projection first:

```text
delta_scores = Q * delta_K^T = Q * (W_UK * delta_c_KV)^T = Q * W_UK * delta_c_KV^T
```

So **the error is remixed by `W_UK` before it ever reaches scores**: the quantization error in dimension `j` of `c_KV` is distributed across all `d_h` attention dimensions according to the norm and direction of column `j` of `W_UK`.

This yields an optimization that general-purpose tools (GPTQ, AWQ) cannot find: **the per-channel scale should not be set by each channel's absmax of `c_KV`, but weighted by "row norm of `W_UK` times channel absmax"** — because a channel with a large absmax but a tiny corresponding column norm in `W_UK` has error that never reaches scores and does not deserve a fine scale; conversely, a channel with moderate absmax but a large `W_UK` column norm has its error amplified and deserves protected precision.

```text
Suggested importance weight:
  imp[j] = || W_UK[:, j] ||_2 * absmax(c_KV[:, j])

Versus the generic approach:
  imp[j] = absmax(c_KV[:, j])
```

### 2.6 Special case four: R4 rotation must be re-derived under MLA

qt4 covered the four rotation insertion points R1 through R4 in the QuaRot family. **R4 acts between V and O** (`V -> R_V`, `O <- R_O`) to handle outliers on the V side — because `V * W_OV` can be rewritten as `(V*R)(R^T * W_OV)`, which leaves the function unchanged when `R` is orthogonal while flattening V's distribution.

qt4's derivation has a premise: **`V` and `O` live in the same space with the same dimension**, so the rotation inserts directly.

<div class="warnbox">
<strong>The following is my own derivation, not a measured result from the literature — please read it as a hypothesis awaiting verification.</strong><br><br>
Under MLA, `V` is projected from `c_KV` (`V = c_KV * W_UV`), RoPE acts on `k_R`, and `W_UK`'s mixing has to be applied to `c_KV` first. To reuse the R4 idea under MLA, the rotation cannot act on `V` (`V` is an intermediate product and never enters the cache). It must act on <strong><code>c_KV</code></strong>, and the dimension is the <strong>post-absorption <code>d_c + d_R = 576</code></strong> rather than MHA's <code>d_h = 128</code>; and it must be applied to `c_KV` <strong>before RoPE</strong> (otherwise it breaks `k_R`'s rotation basis).<br><br>
I lean toward believing this transform is mathematically valid (since `W_UK` and `W_UV` are fixed linear maps, an orthogonal transform can be absorbed into them), but I have <strong>not found published measured results for rotation-based quantization of MLA at W4A4</strong> to corroborate it. Current MLA deployments (the DeepSeek V2/V3 family) run W4A16 or W8, which do not yet need anything at R4's level — so practice has not yet forced this problem to be solved.
</div>

## 3. Boundary Three: The Edge int4 Payoff Is Real, Just Not Where You Think

### 3.1 Fact one: the edge mainstream is W4A16, not W4A4

If your picture of quantization comes from datacenter discussions, you probably assume "4 bit" means W4A4 (both weights and activations at 4 bit). **On the edge it does not.**

In Qualcomm QNN model bundles, the overwhelming majority of LLM quantization configurations are **`w4a16`** — int4 weights, int16 activations. The reason is direct: **edge decoding is extremely memory-bound**.

```text
Edge decode arithmetic intensity = 2 * batch / bytes_per_weight
                                 ~= 2 * 1 / 0.5625 ~= 3.5 FLOP/byte   (4 bit, group 128)

Whereas the Hexagon NPU's compute-to-bandwidth ratio is tens of FLOP/byte
```

In other words, during edge decoding more than 90 percent of the NPU's compute units are waiting for data. In that regime **weight bit width is the only variable that matters**; activation bit width barely affects speed (activations are two orders of magnitude fewer bytes than weights). Squeezing activations to 4 bit only adds kernel complexity and loses accuracy without buying speed.

That is a useful calibration: **the datacenter narrative that "W4A4 is the real 4 bit" does not hold on the edge**. W4A16 on the edge is not a compromise; it is the correct choice under this hardware constraint.

### 3.2 Fact two: what decides NPU compatibility is the format, not the bit width

This is, I think, the most easily overlooked rule in all of edge quantization.

llama.cpp has a pile of 4-bit quantization types: `Q4_0`, `Q4_1`, `Q4_K_S`, `Q4_K_M`, `Q5_K_M` and more. Intuitively they are "all 4 bit and should be similar". But on the Hexagon NPU:

| Quantization type | Bit width | Lands on |
|---|---|---|
| `Q4_0` | 4 bit | **Hexagon NPU** ✓ |
| `Q4_K_M` | ~4.5 bit | GPU / CPU ✗ |
| `Q5_K_M` | ~5.5 bit | GPU / CPU ✗ |

`Q4_K_M` and `Q4_0` differ by 0.5 bit in average width. Why can one reach the NPU and the other not?

**Because `Q4_K_M` is a mixed-precision super-block format**: within a 256-element super-block it stores the scales of 8 sub-blocks (32 elements each) at **different precisions** — some at 6 bit, some at 4 bit — plus an extra super-block scale. Mathematically this is better (more accurate), but **the NPU's hardware path does not support this nested scale structure**.

So:

```text
The question is not "can 4 bit run on the NPU" but "does the NPU recognize this 4-bit scale layout".
```

<div class="warnbox">
<strong>The practical form of this rule:</strong> when choosing an edge quantization format, <strong>check the hardware support list first, then talk about accuracy</strong>. <code>Q4_0</code> is less accurate than <code>Q4_K_M</code>, but if only <code>Q4_0</code> runs on the NPU then <code>Q4_K_M</code>'s accuracy advantage is meaningless — it runs on the CPU and may be 5x slower. <strong>This is not the usual precision-for-speed trade-off; it is a discrete jump where format decides the path.</strong> A similar rule exists in the datacenter (NVFP4 needs Blackwell, and MXFP4's E8M0 scales need specific MMA support), but on the edge it is stricter and less transparent.
</div>

### 3.3 Fact three: the payoff is real on the edge and fake in the datacenter

Qualcomm's official numbers first (operator-level microbenchmarks):

| Metric | INT4 vs INT8 |
|---|---|
| Performance | **+90%** |
| Energy efficiency | **+60%** |
| Tensor throughput | **2x** |

These numbers are credible, but read the measurement conditions: **operator-level microbenchmark, memory-bound regime, `w4a16` configuration**. Under those conditions INT4 reduces weight bandwidth almost linearly relative to INT8, so roughly 2x throughput is expected.

Third-party reproducible end-to-end numbers are much more modest: **Llama 2-7B at 20 tok/s on the Snapdragon 8 Gen 3 NPU versus 13 tok/s on the GPU**. Note that this comparison is **NPU versus GPU**, not int4 versus int8 — it measures an architectural difference (the NPU has dedicated weight-decompression and matmul paths), and int4 accounts for only part of it.

Now look at the datacenter side: H100, H200 and B200 have **no native INT4-by-INT4 MMA** (as noted back in qt1). Doing W4A4 in the datacenter means the bandwidth saved on 4-bit weights gets spent on dequantizing to FP16 before compute, so the net gain is far smaller than on the edge. That is why "4 bit gives 2x" is true on the edge and false in the datacenter — **not because the quantization method differs, but because the hardware path does**.

### 3.4 The payoff structure of edge int4, in one summary

```text
Edge: memory-bound to an extreme -> weight bit width is the only variable -> W4A16 is the answer
      Format decides NPU compatibility -> accuracy is not first, the path is
      The payoff is real but comes from bandwidth plus dedicated paths, not from INT4 being fast

Datacenter: a mix of compute-bound prefill and bandwidth-bound decode
      No native INT4 MMA -> W4A4's dequantization overhead eats most of the gain
      -> W4A16 with FP8 KV is the current sweet spot
```

## 4. What the Three Boundaries Have in Common

The three boundaries look unrelated (one is a decoding algorithm, one an attention structure, one hardware), but their lesson is the same sentence repeated three times:

<div class="keybox">
<strong>Hardware picks the format, the format picks the algorithm, and the algorithm only has room to optimize once the first two are fixed.</strong><br><br>
<strong>Boundary one (speculative decoding):</strong> quantization changes the <strong>ratio</strong> between the cost of single-token forward and the cost of tree verification, so the optimal configuration (tree size) moves with it. This is constraint propagation at the algorithm layer.<br>
<strong>Boundary two (MLA):</strong> the attention structure decides what tensors the cache holds, which in turn decides which scale partitioning is meaningful. This is constraint propagation at the structure layer.<br>
<strong>Boundary three (edge):</strong> what scale layouts the hardware supports decides which quantization formats are usable, which in turn sets the accuracy ceiling. This is constraint propagation at the hardware layer.<br><br>
All three propagate in the same direction: <strong>bottom up</strong>. So "which quantization method is best" <strong>has no answer divorced from context</strong> — every method in qt1 through qt6 is an optimization <strong>given a format and a structure</strong>, not a scheme that can select the format and structure in reverse.
</div>

One table to close out the three boundaries:

| Boundary | Default assumption that fails | Correct choice at 4 bit | How the failure hides |
|---|---|---|---|
| **Speculative decoding** | bigger tree is always better | tree size from 60 down to ~16, or insert an intermediate model | PPL normal, acceptance zero |
| **Speculative decoding** | the draft can be quantized more aggressively | draft and target in the same family, else keep the draft BF16 | PPL normal, acceptance 0% |
| **GQA** | per-channel and per-token are freely interchangeable | per-channel for K (even more so than under MHA) | the drop only appears at long context |
| **MLA** | K and V can use different strategies | per-channel throughout (`c_KV` outliers are in the latent dims) | short prompts fine, 32k collapses |
| **MLA** | every cache component should be quantized | keep `k_R` in FP16 (11 percent, very high risk) | completely invisible at short context |
| **MLA** | per-channel scales follow absmax | weight by the `W_UK` column norm times absmax | small but persistent drop, hard to attribute |
| **Edge** | equal bit width means equivalent | `Q4_0` rather than `Q4_K_M` (the NPU only takes simple scales) | the model runs, 5x slower |
| **Edge** | W4A4 is more advanced than W4A16 | W4A16 (edge activation width does not affect speed) | W4A4 is actually slower |

Finally, one practical judgment. The previous six posts teach **how to quantize well**. What this post adds is that **you have to validate it again afterwards**. Because the acceptance metrics of the previous six (PPL, per-layer reconstruction error, downstream task scores) are all measured inside the closed world of the model itself — they cannot detect the V/D ratio of tree verification, cannot detect phase-error accumulation at 32k context, and cannot detect whether a format runs on an NPU.

**Quantization validation has to extend to the end of the deployment chain, not stop at the checkpoint.** That is the one thing this post is trying to say.

## 5. Series Navigation

- Previous: **qt6, QAT and Quantization-Aware Training** — route D, STE, PACT and LSQ. All three boundaries apply to it too: a QAT-tuned model also runs into speculative decoding and MLA.
- **qt5, KV Cache Quantization and the FP4 Frontier** — KIVI's split, and the accuracy-allocation difference between INT4 and FP4. Section 2 here supplements it with the structural boundary.
- **qt4, QuaRot / SpinQuant / DuQuant** — route C, rotation, the R1 through R4 insertion points, and the "rotation is not free" boundary. Section 2.6 here re-derives R4 under MLA.
- **qt3, SmoothQuant and AWQ** — route B. Section 2.5's `W_UK` row-norm weighting is essentially AWQ's "allocate precision by importance" idea rewritten for MLA.
- **qt2, RTN and GPTQ** — route A, packing formats and group size. Section 3.2's `Q4_K_M` super-block structure is an extension of packing formats.
- **qt1, The Basic Language of Quantization** — the four-route map and the naming convention.
- **op2, Model Quantization Overview** — the series guide and lineage table.

Related on this site: **sd1 through sd5 (the speculative decoding series)** cover the full mechanics of acceptance rates and tree structure design; Section 1 here is a patch to that from the quantization side. **fa4 (DeepSeek V4)** covers the MLA-to-NSA evolution directly relevant to Section 2's cache structure. **fw5** covers kernel selection in inference engines, and Section 3.2's "format decides the path" is the same logic expressed on the hardware side.
