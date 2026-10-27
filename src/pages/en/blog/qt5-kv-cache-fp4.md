---
title: "Quantization Notes (5): KV Cache Quantization and the FP4 Frontier — an orthogonal compression axis"
description: "At long context the KV cache can consume more memory than the weights — at 32k context and batch 32 it is over 80 percent of the footprint — yet it is completely orthogonal to weight quantization. Weights are static and can be calibrated at leisure; the KV cache is produced at runtime and must be quantized online. This post covers the three difficulties that make KV quantization different (per-token absmax is unknowable in advance, K is an order of magnitude more sensitive than V, block alignment with PagedAttention), the KIVI scheme of per-channel K plus per-token V plus a full-precision residual window, why K errors get amplified exponentially by softmax, and how FP8 E4M3/E5M2 and NVFP4/MXFP4 microscaling formats change the entire selection logic. Ends with a concrete 2026 selection table."
pubDate: 2026-10-27
series: Quantization Notes
lang: en
altLang: zh
altHref: /blog/qt5-kv-cache-fp4
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

The first four notes are all about **weights and activations**. But the memory bill at inference has a third line item, and it is often the largest:

```text
memory = weights + KV cache + intermediate activations
         ↑static   ↑grows with context     ↑grows with batch
```

Concretely (LLaMA-2-7B, FP16):

```text
weights                 13.5 GB
KV @ 4k,  bs=1           2.0 GB
KV @ 32k, bs=32          512 GB   ← does not fit on a single 80 GB card
```

**At long context and large batch, the KV cache takes over 80% of memory.** Optimising weights at that point is beside the point — compressing weights to 4 bit saves 10 GB while KV grows by hundreds.

KV quantization looks like "do the same thing to two more tensors", but it has three difficulties that weight quantization does not have. This post covers those three, plus the reason FP8 and FP4 fundamentally change the nature of the problem.

<div class="keybox">
<strong>Where this sits:</strong> fully orthogonal to qt1–qt4. There is exactly one coupling point: qt4's R4 (Q/K rotation) makes KV quantization strictly easier, and that connection is developed in §1.3.
</div>

## 1. Three Difficulties Weight Quantization Does Not Have

### 1.1 Difficulty one: the absmax is only known at runtime

Weight quantization is **offline**: read the whole matrix, compute an exact `absmax`, run GPTQ's Hessian, grid-search `α`. Slow is fine, it happens once.

KV cache quantization is **online** and must be per-token:

```text
decode step t:
  compute new k_t, v_t of shape [1, n_kv_heads, head_dim]
  they must be written to cache immediately, already quantized
  (otherwise you saved no memory)
  but at this moment you only have this one token — you cannot know
  the magnitude of future tokens
```

So KV quantization is forced into **per-token dynamic** (or a fixed per-tensor scale). Per-channel works for K — the `head_dim` axis is static and can be profiled in advance — but not for V, because V's per-token structure is exactly what varies.

**This is the central insight of KIVI**:

```text
K: outliers concentrate on a few fixed CHANNELS (structural, like weights)
   → quantize per-channel; the scale is profiled offline and reused across
     all tokens   ✓

V: outliers concentrate on a few TOKENS (transient)
   → quantize per-token; each token gets its own scale   ✓
```

This division is the opposite of the natural guess (that K and V should be treated identically), and it has clear empirical support: K's variance across channels is much larger than across tokens, and V's is the reverse.

<div class="fig">
<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="K has column outliers, V has row outliers">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">K's outliers are vertical stripes; V's are horizontal stripes</text>
  <text x="20" y="44" font-size="11" fill="#6b7280">Each cell is one cache element; dark = large magnitude</text>

  <text x="30" y="72" font-size="12" font-weight="700" fill="#1d4ed8">K cache　[token ↓ × channel →]</text>
  <g stroke="#bfdbfe" fill="#dbeafe">
    <rect x="30" y="82" width="16" height="8"/><rect x="46" y="82" width="16" height="8"/><rect x="62" y="82" width="16" height="8"/><rect x="78" y="82" width="16" height="8"/><rect x="94" y="82" width="16" height="8"/><rect x="110" y="82" width="16" height="8"/><rect x="126" y="82" width="16" height="8"/><rect x="142" y="82" width="16" height="8"/><rect x="158" y="82" width="16" height="8"/><rect x="174" y="82" width="16" height="8"/><rect x="190" y="82" width="16" height="8"/><rect x="206" y="82" width="16" height="8"/>
    <rect x="30" y="90" width="16" height="8"/><rect x="46" y="90" width="16" height="8"/><rect x="62" y="90" width="16" height="8"/><rect x="78" y="90" width="16" height="8"/><rect x="94" y="90" width="16" height="8"/><rect x="110" y="90" width="16" height="8"/><rect x="126" y="90" width="16" height="8"/><rect x="142" y="90" width="16" height="8"/><rect x="158" y="90" width="16" height="8"/><rect x="174" y="90" width="16" height="8"/><rect x="190" y="90" width="16" height="8"/><rect x="206" y="90" width="16" height="8"/>
    <rect x="30" y="98" width="16" height="8"/><rect x="46" y="98" width="16" height="8"/><rect x="62" y="98" width="16" height="8"/><rect x="78" y="98" width="16" height="8"/><rect x="94" y="98" width="16" height="8"/><rect x="110" y="98" width="16" height="8"/><rect x="126" y="98" width="16" height="8"/><rect x="142" y="98" width="16" height="8"/><rect x="158" y="98" width="16" height="8"/><rect x="174" y="98" width="16" height="8"/><rect x="190" y="98" width="16" height="8"/><rect x="206" y="98" width="16" height="8"/>
    <rect x="30" y="106" width="16" height="8"/><rect x="46" y="106" width="16" height="8"/><rect x="62" y="106" width="16" height="8"/><rect x="78" y="106" width="16" height="8"/><rect x="94" y="106" width="16" height="8"/><rect x="110" y="106" width="16" height="8"/><rect x="126" y="106" width="16" height="8"/><rect x="142" y="106" width="16" height="8"/><rect x="158" y="106" width="16" height="8"/><rect x="174" y="106" width="16" height="8"/><rect x="190" y="106" width="16" height="8"/><rect x="206" y="106" width="16" height="8"/>
    <rect x="30" y="114" width="16" height="8"/><rect x="46" y="114" width="16" height="8"/><rect x="62" y="114" width="16" height="8"/><rect x="78" y="114" width="16" height="8"/><rect x="94" y="114" width="16" height="8"/><rect x="110" y="114" width="16" height="8"/><rect x="126" y="114" width="16" height="8"/><rect x="142" y="114" width="16" height="8"/><rect x="158" y="114" width="16" height="8"/><rect x="174" y="114" width="16" height="8"/><rect x="190" y="114" width="16" height="8"/><rect x="206" y="114" width="16" height="8"/>
  </g>
  <g fill="#dc2626" stroke="#991b1b">
    <rect x="78" y="82" width="16" height="8"/><rect x="78" y="90" width="16" height="8"/><rect x="78" y="98" width="16" height="8"/><rect x="78" y="106" width="16" height="8"/><rect x="78" y="114" width="16" height="8"/>
    <rect x="158" y="82" width="16" height="8"/><rect x="158" y="90" width="16" height="8"/><rect x="158" y="98" width="16" height="8"/><rect x="158" y="106" width="16" height="8"/><rect x="158" y="114" width="16" height="8"/>
  </g>
  <text x="30" y="138" font-size="10" fill="#b91c1c">↑ dark cells form whole columns → per-channel scale matches them</text>
  <text x="30" y="156" font-size="10" fill="#15803d" font-weight="700">Conclusion: K uses per-channel (scale profiled offline, reused)</text>

  <text x="380" y="72" font-size="12" font-weight="700" fill="#7c3aed">V cache　[token ↓ × channel →]</text>
  <g stroke="#ddd6fe" fill="#ede9fe">
    <rect x="380" y="82" width="16" height="8"/><rect x="396" y="82" width="16" height="8"/><rect x="412" y="82" width="16" height="8"/><rect x="428" y="82" width="16" height="8"/><rect x="444" y="82" width="16" height="8"/><rect x="460" y="82" width="16" height="8"/><rect x="476" y="82" width="16" height="8"/><rect x="492" y="82" width="16" height="8"/><rect x="508" y="82" width="16" height="8"/><rect x="524" y="82" width="16" height="8"/><rect x="540" y="82" width="16" height="8"/><rect x="556" y="82" width="16" height="8"/>
    <rect x="380" y="90" width="16" height="8"/><rect x="396" y="90" width="16" height="8"/><rect x="412" y="90" width="16" height="8"/><rect x="428" y="90" width="16" height="8"/><rect x="444" y="90" width="16" height="8"/><rect x="460" y="90" width="16" height="8"/><rect x="476" y="90" width="16" height="8"/><rect x="492" y="90" width="16" height="8"/><rect x="508" y="90" width="16" height="8"/><rect x="524" y="90" width="16" height="8"/><rect x="540" y="90" width="16" height="8"/><rect x="556" y="90" width="16" height="8"/>
    <rect x="380" y="98" width="16" height="8"/><rect x="396" y="98" width="16" height="8"/><rect x="412" y="98" width="16" height="8"/><rect x="428" y="98" width="16" height="8"/><rect x="444" y="98" width="16" height="8"/><rect x="460" y="98" width="16" height="8"/><rect x="476" y="98" width="16" height="8"/><rect x="492" y="98" width="16" height="8"/><rect x="508" y="98" width="16" height="8"/><rect x="524" y="98" width="16" height="8"/><rect x="540" y="98" width="16" height="8"/><rect x="556" y="98" width="16" height="8"/>
    <rect x="380" y="106" width="16" height="8"/><rect x="396" y="106" width="16" height="8"/><rect x="412" y="106" width="16" height="8"/><rect x="428" y="106" width="16" height="8"/><rect x="444" y="106" width="16" height="8"/><rect x="460" y="106" width="16" height="8"/><rect x="476" y="106" width="16" height="8"/><rect x="492" y="106" width="16" height="8"/><rect x="508" y="106" width="16" height="8"/><rect x="524" y="106" width="16" height="8"/><rect x="540" y="106" width="16" height="8"/><rect x="556" y="106" width="16" height="8"/>
    <rect x="380" y="114" width="16" height="8"/><rect x="396" y="114" width="16" height="8"/><rect x="412" y="114" width="16" height="8"/><rect x="428" y="114" width="16" height="8"/><rect x="444" y="114" width="16" height="8"/><rect x="460" y="114" width="16" height="8"/><rect x="476" y="114" width="16" height="8"/><rect x="492" y="114" width="16" height="8"/><rect x="508" y="114" width="16" height="8"/><rect x="524" y="114" width="16" height="8"/><rect x="540" y="114" width="16" height="8"/><rect x="556" y="114" width="16" height="8"/>
  </g>
  <g fill="#dc2626" stroke="#991b1b">
    <rect x="380" y="90" width="16" height="8"/><rect x="396" y="90" width="16" height="8"/><rect x="412" y="90" width="16" height="8"/><rect x="428" y="90" width="16" height="8"/><rect x="444" y="90" width="16" height="8"/><rect x="460" y="90" width="16" height="8"/><rect x="476" y="90" width="16" height="8"/><rect x="492" y="90" width="16" height="8"/><rect x="508" y="90" width="16" height="8"/><rect x="524" y="90" width="16" height="8"/><rect x="540" y="90" width="16" height="8"/><rect x="556" y="90" width="16" height="8"/>
    <rect x="380" y="106" width="16" height="8"/><rect x="396" y="106" width="16" height="8"/><rect x="412" y="106" width="16" height="8"/><rect x="428" y="106" width="16" height="8"/><rect x="444" y="106" width="16" height="8"/><rect x="460" y="106" width="16" height="8"/><rect x="476" y="106" width="16" height="8"/><rect x="492" y="106" width="16" height="8"/><rect x="508" y="106" width="16" height="8"/><rect x="524" y="106" width="16" height="8"/><rect x="540" y="106" width="16" height="8"/><rect x="556" y="106" width="16" height="8"/>
  </g>
  <text x="380" y="138" font-size="10" fill="#b91c1c">↑ dark cells form whole rows → only per-token scale matches them</text>
  <text x="380" y="156" font-size="10" fill="#15803d" font-weight="700">Conclusion: V uses per-token (scale computed each step)</text>

  <line x1="20" y1="176" x2="660" y2="176" stroke="#e5e7eb" stroke-width="1"/>
  <text x="20" y="200" font-size="12" font-weight="700" fill="#1a1a1a">Why the division matters: the cost of getting it backwards</text>
  <text x="20" y="222" font-size="11" fill="#374151">K with per-token → each token gets its own scale, but every row has the same outlier channels → no gain, extra scale storage</text>
  <text x="20" y="242" font-size="11" fill="#374151">V with per-channel → one token's spike pollutes a whole column's scale → every other token in that column is crushed, error explodes</text>
  <text x="20" y="268" font-size="11" fill="#b91c1c" font-weight="700">Measured: KIVI's K-per-channel + V-per-token stays usable at 2 bit; the reverse combination collapses</text>

  <rect x="20" y="284" width="640" height="32" rx="4" fill="#f0fdf4" stroke="#86efac"/>
  <text x="32" y="304" font-size="11" fill="#15803d">Bonus: K's scale can be profiled once offline, so inference pays nothing; V's scale costs O(head_dim) per token</text>
</svg>
<p class="cap">Figure 1: K's outliers are structural (fixed channels), V's are transient (fixed tokens). That structural difference dictates the quantization axis — and getting it wrong fails in a way that is easy to miss.</p>
</div>

### 1.2 Difficulty two: K's error is amplified exponentially by softmax

This is the most counterintuitive and most frequently overlooked point in KV quantization. Recall attention:

```text
scores = Q Kᵀ / √d
P = softmax(scores)
out = P V
```

A quantization error `ΔK` propagates to `scores`:

```text
Δscores = Q ΔKᵀ / √d
```

That looks linear, exactly like weight error. **But `softmax` is not linear.**

```text
softmax(s + Δs)_i / softmax(s)_i = exp(Δs_i) / Σ_j P_j exp(Δs_j)
```

If `Δs_i ≈ 1`, then `exp(1) = 2.72` — **the attention weight is off by 172%.** And `scores` typically range over 20–60 (LLaMA-2 attention logits routinely exceed 40), with `√d = √128 ≈ 11.3`, so a 1% relative error in `ΔK` corresponds to `Δscores ≈ 0.5`.

**This is why K is roughly an order of magnitude more sensitive than V:**

| | Error path | Sensitivity |
|---|---|---|
| **K (key)** | `ΔK → Δscores → exp() → ΔP → Δout` | **very high** (exponential) |
| **V (value)** | `ΔV → Δ(PV) = P ΔV` | moderate (linear weighted average) |
| **weights** | `ΔW → X ΔW` | moderate (linear) |

<div class="warnbox">
<strong>Practical conclusion:</strong> if you can only quantize one, quantize <strong>V</strong>, and leave K alone. The standard industrial configuration is "K stays FP16/FP8, V compressed to 4 bit" — not the reverse. KIVI achieves 2-bit K by combining per-channel quantization (removing the structural outlier) with a full-precision residual window, not by spreading precision more evenly.
</div>

A related engineering detail: **the absolute magnitude of `scores` depends on the RoPE base frequency and on position.** At long context, `scores` grow larger (distant token dot products more readily hit large values), so **KV quantization degrades more at long context than at short context.** This is why so many reports say "lossless within 4k, noticeable degradation from 32k."

### 1.3 Difficulty three: it has to coexist with PagedAttention

In vLLM the KV cache is not a contiguous tensor — it is split into fixed-size blocks (16 tokens by default) stitched together by a block table. That imposes two constraints on quantization:

```text
① the block is the atomic unit of quantize/dequantize → block_size and
   group_size should align (vLLM's FP8 KV cache is block-wise, one scale
   per 16 tokens)
② blocks are shared across sequences (prefix caching) → the scale must be
   stored per block, never globally
③ preemption / recompute discards blocks → the scale must be reconstructible,
   not dependent on whole-sequence statistics
```

vLLM's implementation (`kv_cache_dtype="fp8"`) uses `fp8_e4m3` or `fp8_e5m2` with per-tensor or per-block static scales, computed once via `calculate_kv_scales`. This is a deliberate simplification: fully dynamic per-token scales would break block sharing and prefix caching.

There is a nice connection back to qt4 here: **applying R4 (rotating Q and K by the same orthogonal matrix) leaves attention scores unchanged but spreads K's magnitudes across channels.** That reduces the dynamic range the KV quantization has to cover, which is why QuaRot can claim KV4 — the rotation did the hard part before the quantizer ever runs.

## 2. The Complete KIVI Scheme

Putting the analysis together, KIVI (2024.02) prescribes:

```text
① K: per-channel quantization, 2 bit, channel scales profiled offline
② V: per-token quantization, 2 bit, scale computed per token
③ residual window: the most recent R tokens (default 128) stay full precision
   → the newest tokens matter most to the current output (attention recency bias)
   → and their magnitude distribution has not stabilised, so hard quantization
     is expensive there
④ group-wise: every G tokens (default 32) share one V scale
   → reduces scale storage while staying adaptive
```

The residual window deserves emphasis, because it exposes a fundamental property of KV quantization: **the error contribution is not uniform across tokens.**

```text
error contribution = Σ_t P_t · ΔV_t     ← P_t is larger for recent tokens
```

Keeping the last 128 tokens at full precision costs only `128/ctx` extra memory (0.4% at 32k) while recovering more than half the perplexity gap. **That is an extremely high-value engineering compromise**, and every subsequent KV quantization effort (KVQuant, ZipCache, QServe) inherits it.

## 3. FP8 and FP4: The Format Changes the Problem

Everything above concerns **INT** quantization (uniform grids). But since 2024, hardware and mainstream schemes have been migrating to **floating point**. This is not a cosmetic format swap — it changes the nature of the problem.

### 3.1 FP8: half the outlier problem dissolves into the format

Two FP8 variants:

```text
E4M3: 4 exponent bits + 3 mantissa, range ±448, smallest normal 2⁻⁶ ≈ 0.0156
E5M2: 5 exponent bits + 2 mantissa, range ±57344, but only 2 mantissa bits
```

The key is **dynamic range**. Compare with INT8:

```text
INT8: uniform grid, ±127, spacing = absmax/127
      → one 100× outlier squeezes everything else into 1–2 bins
FP8 E4M3: the exponent gives large and small values their own bin density
      → 100 and 0.1 are both representable, each with 3 mantissa bits
        (about 12% relative precision)
```

**This is why FP8 makes SmoothQuant less necessary** (as qt3 noted). SmoothQuant exists because INT8's uniform grid cannot cover an outlier and normal values simultaneously; FP8's non-uniform grid solves that natively — no relocation needed, just convert.

Measured (LLaMA-2-7B, per-tensor dynamic):

| Config | PPL | vs FP16 |
|---|---|---|
| FP16 | 5.47 | — |
| INT8 W8A8 per-tensor | 5.60 | +2.4% |
| INT8 W8A8 per-token × per-channel | 5.51 | +0.7% |
| **FP8 E4M3 W8A8 per-tensor** | **5.50** | **+0.5%** |
| FP8 E4M3 W8A8 + KV8 | 5.52 | +0.9% |

Note the last two rows: **FP8 per-tensor already matches INT8's finest per-token-per-channel scheme.** That is a format dividend — better accuracy from a simpler kernel.

The catch is that FP8 requires hardware support (H100 and later have native FP8 MMA; A100 does not). On A100 fleets, INT8 remains the pragmatic choice.

### 3.2 FP4: microscaling is a genuine paradigm change

FP4 is only 4 bits: 1 sign, 2 exponent, 1 mantissa (E2M1). Bare FP4 has far too narrow a dynamic range (max 6, smallest normal 0.5) and collapses if used alone.

The real solution is **microscaling**: share a high-precision scale within a very small block.

```text
MXFP4 (OCP standard, 2023):
  one E8M0 scale (exponent only, 8 bits) per 32 elements
  → 32 FP4 values + 1 8-bit scale = 4.25 bit/element
  → effective dynamic range = FP4's 12× times E8M0's 2^±127 ≈ very wide

NVFP4 (NVIDIA, 2025, native on Blackwell):
  one FP8 E4M3 scale per 16 elements
  → 16 FP4 values + 1 8-bit scale + 2-bit global = 4.5 bit/element
  → smaller blocks (16 vs 32) localise outliers more tightly
```

<div class="fig">
<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="How INT4 and NVFP4 spend their four bits differently">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">Same four bits, spent on different things</text>

  <text x="20" y="56" font-size="12" font-weight="700" fill="#b91c1c">INT4: all 4 bits count bins (16 uniform bins)</text>
  <g>
    <rect x="30" y="68" width="620" height="30" fill="#fef2f2" stroke="#f87171"/>
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
  <text x="20" y="140" font-size="11" fill="#374151">Problem: bin width = absmax/15. With a 10× outlier in the block, everything else lands in 1–2 bins and rounds to zero.</text>

  <text x="20" y="176" font-size="12" font-weight="700" fill="#15803d">NVFP4: FP4's 4 bits (1 sign + 2 exp + 1 mantissa) + one FP8 scale per 16 elements</text>

  <g>
    <rect x="30" y="188" width="620" height="34" fill="#f0fdf4" stroke="#4ade80"/>
    <g text-anchor="middle">
      <rect x="42" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <rect x="70" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <rect x="98" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <rect x="126" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <text x="97" y="209" font-size="9" fill="#166534">small values</text>
      <text x="97" y="238" font-size="9" fill="#166534">scale ≈ 0.05</text>

      <rect x="196" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <rect x="224" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <rect x="252" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <rect x="280" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <text x="251" y="209" font-size="9" fill="#166534">mid</text>
      <text x="251" y="238" font-size="9" fill="#166534">scale ≈ 0.8</text>

      <rect x="350" y="196" width="26" height="18" fill="#fecaca" stroke="#f87171"/>
      <rect x="378" y="196" width="26" height="18" fill="#fecaca" stroke="#f87171"/>
      <rect x="406" y="196" width="26" height="18" fill="#fecaca" stroke="#f87171"/>
      <rect x="434" y="196" width="26" height="18" fill="#fecaca" stroke="#f87171"/>
      <text x="405" y="209" font-size="9" fill="#991b1b">outlier</text>
      <text x="405" y="238" font-size="9" fill="#991b1b">scale ≈ 6.0</text>

      <rect x="504" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <rect x="532" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <rect x="560" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <rect x="588" y="196" width="26" height="18" fill="#bbf7d0" stroke="#4ade80"/>
      <text x="559" y="209" font-size="9" fill="#166534">mid</text>
      <text x="559" y="238" font-size="9" fill="#166534">scale ≈ 1.2</text>
    </g>
    <text x="20" y="238" font-size="10" fill="#6b7280">each 16-element block →</text>
  </g>

  <text x="20" y="264" font-size="11" fill="#374151">Key: the outlier is confined to its own block and contaminates only 15 neighbours; every other block gets its own optimal scale.</text>
  <text x="20" y="284" font-size="11" fill="#15803d" font-weight="700">That is microscaling: localise the outlier with tiny blocks, then recover dynamic range with a wide scale.</text>

  <rect x="20" y="298" width="640" height="30" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="32" y="317" font-size="10" fill="#1d4ed8">Cost: one 8-bit scale per 16 elements → 4.5 bit/element average, 12.5% more than bare INT4. The accuracy gain far exceeds that 12.5%.</text>
</svg>
<p class="cap">Figure 2: INT4 spends all four bits on mantissa (uniform grid, zero tolerance for outliers); FP4 spends two of them on exponent (non-uniform) and uses a per-block scale to localise outliers. You pay 12.5% more bits for a large accuracy return.</p>
</div>

### 3.3 Why FP4 may end the W4A16 versus W4A4 argument

Return to the question qt4 closed on: is W4A4 worth it? Without native INT4×INT4 hardware support, no.

FP4 changes that, because **Blackwell (B200/GB200) has native FP4 MMA**:

```text
H100: FP8 native, INT4 only emulated
B200: FP4 native (NVFP4), theoretical throughput 2× FP8

Meaning: W4A4 turns from "memory saving only" into "a real 2× speedup"
```

More importantly, FP4 makes **A4 viable**: everything qt4's rotation route does is, at bottom, help INT4's uniform grid claw back dynamic range. NVFP4's per-block scale supplies that dynamic range at the format level — so in principle, **NVFP4 does not need rotation to do W4A4.**

Measured (NVIDIA technical blog 2025 plus community reproductions, LLaMA-3-8B):

| Scheme | Bits | PPL | Note |
|---|---|---|---|
| BF16 | 16 | 6.14 | baseline |
| INT4 W4A16 (AWQ) | 4.5 | 6.28 | production default |
| INT4 W4A4 (QuaRot) | 4.5 | 6.55 | needs rotation kernels |
| **NVFP4 W4A4 (no rotation)** | **4.5** | **6.22** | native on Blackwell |
| **NVFP4 W4A4 + rotation** | **4.5** | **6.16** | best |

Note the third and fourth rows: **NVFP4 without rotation already beats INT4 with QuaRot.** That is the paradigm shift — rotation drops from prerequisite to refinement. It still buys 0.06 PPL, but it is no longer what makes W4A4 possible.

## 4. Realistic Selection in 2026

| Scenario | Weights | Activations | KV cache | Recommended |
|---|---|---|---|---|
| **Single card 7B/8B, chat** | W4 | A16 | FP16 | **AWQ** (qt3), least effort |
| **Single card 70B, long context** | W4 | A16 | **FP8** | AWQ + vLLM `kv_cache_dtype=fp8` |
| **H100 cluster, high throughput** | FP8 | FP8 | FP8 | all FP8, per-tensor (no SmoothQuant needed) |
| **B200/GB200** | **NVFP4** | **NVFP4** | FP8 | native FP4, no rotation |
| **Memory ceiling (70B @ 32k on one card)** | W4 | A4 | **INT4 + residual window** | QuaRot + KIVI |
| **Edge / NPU** | W4 | A8/A4 | INT8 | match the silicon's native format, do not force GPU recipes |

<div class="keybox">
<strong>Three directly usable judgements:</strong><br>
① <strong>KV cache is the first priority at long context</strong> — more so than weights. Start with FP8 KV; it is essentially free.<br>
② <strong>K is an order of magnitude more sensitive than V</strong> (softmax amplification). If you compress only one, compress V.<br>
③ <strong>With Blackwell, use NVFP4</strong> and stop investing in INT4 rotation; without it, stay at W4A16 + AWQ.
</div>

## 5. Closing the Series

Five posts in, the skeleton of quantization compresses to three sentences:

```text
① Every quantization method answers the same question:
   with a limited number of bins, how do you distribute resolution?

② The four routes are four distribution strategies:
   A change the rounding (RTN/GPTQ)   — minimise the error
   B change distribution · scale (SQ/AWQ) — move difficulty from hard to easy
   C change distribution · rotate (QuaRot) — flatten the spike
   D change the training (QAT)        — let the model adapt to the bins

③ And the choice of FORMAT (INT vs FP8 vs FP4) often matters more than
   the algorithm: the format defines what the bins look like, and the
   algorithm can only work within the bins it is given.
```

**The last sentence is the most practical.** In 2026: if you have an A100, use AWQ W4A16 and stop tinkering. If you have an H100, use FP8 and skip SmoothQuant. If you have a B200, use NVFP4 and skip rotation. **Hardware decides format, format decides algorithm, and the algorithm only has room to optimise once the first two are fixed.**

## 6. Series Navigation

- Previous **qt4 · QuaRot / SpinQuant / DuQuant**: route C, rotation for W4A4.
- **qt3 · SmoothQuant and AWQ**: route B, two answers to activation outliers.
- **qt2 · RTN and GPTQ**: route A, the baseline for everything.
- **qt1 · The Basic Language**: granularity, symmetry, calibration, the four-route map.
- **op2 · Model Quantization Overview**: series index and lineage table.

Related on this site: **op3 (Ascend EagleQBSA)** uses per-head INT8 for Q/K with FP8 for P/V — a direct engineering confirmation of this post's "K sensitive, V compressible" conclusion; **fw5** covers PagedAttention's block management, which is exactly the block-alignment constraint in §1.3; **fw4** covers dequantization fusion from the kernel perspective.
