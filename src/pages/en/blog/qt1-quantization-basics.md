---
title: "Quantization Notes (1): The Basic Language — granularity, symmetry, calibration, and how to pick among the four routes"
description: "Before memorizing RTN, GPTQ, AWQ, SmoothQuant, QuaRot, you need the three knobs they all share: granularity (per-tensor / per-channel / per-group / per-token), symmetry (symmetric vs asymmetric, and why weights and activations differ), and calibration (absmax vs percentile vs MSE). This post builds that vocabulary, explains where group_size=128 actually comes from, maps every method onto four orthogonal routes (rounding / scaling / rotation / training), and ends with a concrete selection cheat sheet and a warning about why perplexity alone is not enough. Series entry point."
pubDate: 2026-10-27
series: Quantization Notes
lang: en
altLang: zh
altHref: /blog/qt1-quantization-basics
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

There are a dozen quantization methods with a dozen names, but they all answer one question:

```text
Given a limited number of grid points, how do you distribute resolution?
```

Every knob, every algorithm, every paper is a different answer to that single question. Once you see that, the field shrinks from "twelve things to memorize" to "three knobs plus four strategies."

<div class="keybox">
<strong>Where this sits:</strong> entry point of the Quantization Notes series. qt2–qt5 each expand one route; op2 is the index and lineage table. If you already know quantization terminology, skip to §4 for the route map and §6 for the cheat sheet.
</div>

## 1. One Formula Unifies Every Quantization Method

### 1.1 Quantize and dequantize

Every uniform quantization scheme, from INT8 to INT4, is this:

```text
quantize:    q = clamp( round(x / s) + z , 0, 2^b − 1 )
dequantize:  ŷ = s · (q − z)
```

Three symbols carry everything:

| Symbol | Name | What it controls |
|---|---|---|
| `b` | bit width | how many grid points exist (`2^b`) |
| `s` | scale | the spacing between adjacent grid points (`Δ`) |
| `z` | zero-point | where the grid is anchored |

Substituting `z = 0` gives **symmetric** quantization; `z ≠ 0` gives **asymmetric**. That is the entire difference between the two, and it is a difference in how `Δ` is chosen, not in the compute structure.

<div class="fig">
<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Symmetric versus asymmetric quantization grids">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">Same range, two grid layouts</text>

  <!-- symmetric -->
  <text x="20" y="54" font-size="12" font-weight="700" fill="#1d4ed8">Symmetric ( z = 0 ): grid centered on zero</text>
  <line x1="60" y1="100" x2="620" y2="100" stroke="#9ca3af" stroke-width="1.5"/>
  <line x1="340" y1="72" x2="340" y2="128" stroke="#374151" stroke-width="1.2"/>
  <g stroke="#1d4ed8" stroke-width="2">
    <line x1="100" y1="90" x2="100" y2="110"/><line x1="160" y1="90" x2="160" y2="110"/>
    <line x1="220" y1="90" x2="220" y2="110"/><line x1="280" y1="90" x2="280" y2="110"/>
    <line x1="340" y1="90" x2="340" y2="110"/><line x1="400" y1="90" x2="400" y2="110"/>
    <line x1="460" y1="90" x2="460" y2="110"/><line x1="520" y1="90" x2="520" y2="110"/>
    <line x1="580" y1="90" x2="580" y2="110"/>
  </g>
  <text x="340" y="142" font-size="10" fill="#374151" text-anchor="middle">0</text>
  <text x="100" y="142" font-size="10" fill="#6b7280" text-anchor="middle">−7Δ</text>
  <text x="580" y="142" font-size="10" fill="#6b7280" text-anchor="middle">+7Δ</text>
  <text x="20" y="166" font-size="10" fill="#374151">Range: ±7Δ, uniform spacing. Anchor at 0 is free, kernel is simpler.</text>
  <text x="20" y="182" font-size="10" fill="#6b7280">Wastes one representable value ( −8 ) but matches Tensor Core int paths natively.</text>

  <!-- asymmetric -->
  <text x="20" y="212" font-size="12" font-weight="700" fill="#7c3aed">Asymmetric ( z ≠ 0 ): grid follows the actual range</text>
  <line x1="60" y1="238" x2="200" y2="238" stroke="#9ca3af" stroke-width="1.5"/>
  <g stroke="#7c3aed" stroke-width="2">
    <line x1="70" y1="232" x2="70" y2="244"/><line x1="130" y1="232" x2="130" y2="244"/><line x1="190" y1="232" x2="190" y2="244"/>
  </g>
  <text x="130" y="256" font-size="9" fill="#6b7280" text-anchor="middle">actual range, e.g. ReLU output [0, 60]</text>
  <text x="230" y="230" font-size="10" fill="#374151">→ no grid points wasted on negatives</text>
  <text x="230" y="246" font-size="10" fill="#374151">→ extra zero-point on the decode side</text>
  <text x="230" y="262" font-size="10" fill="#15803d" font-weight="700">Required for post-ReLU / post-SwiGLU activations</text>
</svg>
<p class="cap">Figure 1: Symmetric hides the anchor at zero and pays one wasted level; asymmetric spends a zero-point to skip empty range. Weights are near-zero-centered, activations often are not.</p>
</div>

### 1.2 The naming convention W&lt;bits&gt;A&lt;bits&gt;

You will see `W4A16`, `W8A8`, `W4A4KV4` everywhere. Decode them like this:

| Notation | Weights | Activations | KV Cache | Typical method |
|---|---|---|---|---|
| W8A8 | 8 | 8 | 16 | SmoothQuant, FP8 |
| W4A16 | 4 | not quantized | 16 | **AWQ, GPTQ — the production default** |
| W4A8 | 4 | 8 | 16 | GPTQ, mixed schemes |
| W4A4 | 4 | 4 | 16 | **QuaRot / SpinQuant / DuQuant, NVFP4** |
| W4A4KV4 | 4 | 4 | 4 | QuaRot (full stack) |

`A16` means activations stay in FP16 — they are never quantized. This is the single most important detail in the whole table, because it determines whether you have to fight activation outliers at all. W4A16 does not; every other configuration does. That single fact explains why W4A16 dominates production (see qt3).

### 1.3 Three common confusions

```text
✗ "INT4 is always smaller than FP4"
✓ Both are 4 bit. FP4 has a wider dynamic range but coarser mantissa;
  INT4 has uniform spacing. For near-Gaussian data INT4 wins;
  for heavy-tailed data FP4 wins. See qt5 §3.

✗ "per-channel is always better than per-tensor"
✓ Better accuracy, yes. But per-channel scales break the Tensor Core
  contract for INT8 GEMM (only one reshape allowed in the epilogue).
  This exact constraint is what motivated SmoothQuant — qt3 §1.2.

✗ "quantization is lossless if my PPL barely moved"
✓ PPL is a mean over a smooth distribution. Long context and
  reasoning-heavy tasks degrade first and hide inside that mean — §5.
```

## 2. Knob One: Granularity

Granularity asks: **how large is the group of values that share one scale?**

| Granularity | One `s` per | Overhead | Where used |
|---|---|---|---|
| per-tensor | entire tensor | 1 scalar | simplest, FP8 default |
| per-channel | one output channel | `out` scalars | weights (classic) |
| per-group | `group_size` weights along `in` | `in/group_size × out` | **INT4 weights, group=128** |
| per-token | one token (dynamic) | `batch × seq` | activations, KV |

The trend is obvious: finer granularity means better accuracy and more metadata. The interesting question is why the industry settled on exactly `group_size = 128`.

### 2.1 Where group_size = 128 comes from

Two independent constraints happen to agree:

```text
① Metadata cost:   one fp16 scale per group
   4-bit weight + 16-bit scale per 128 values
   = 4 + 16/128 = 4.125 bit/element   ← 0.125 bit, about 3% overhead

   At group=64:   4 + 16/64  = 4.25 bit   (6% overhead)
   At group=128:  4.125 bit               (3% overhead)  ← knee of the curve
   At group=256:  4.0625 bit              (1.5% overhead, but accuracy drops)

② Hardware width: 128 = 4 × 32, so one group is exactly 4 warps worth of
   lanes on modern GPUs, and one 128-byte cache line at 1 byte/element.
   Kernel can load a group in a single coalesced transaction.
```

Neither constraint alone pins it at 128 — but jointly they make 128 the only point where you are not obviously wasting something. This kind of "two unrelated constraints meeting" is why 128 keeps showing up in groups, tiles, block sizes, and page sizes across the whole inference stack.

## 3. Knob Two: Calibration

For weights, `absmax` is exact and free — you have the whole tensor offline. For activations you cannot see the future, so you must **estimate** the range from a sample.

| Method | How `absmax` estimated | Behavior |
|---|---|---|
| **absmax** | true max over calibration set | preserves outliers, compresses everything else |
| **percentile** | 99.9th percentile (clip the rest) | trades a few outlier values for resolution elsewhere |
| **MSE** | grid-search the clip point minimizing error | best average, slowest, needs a forward pass per candidate |

Practical defaults that actually work:

```text
Calibration set:  128 samples from wiki-text or c4, sequence length 2048
                  → this is what GPTQ, AWQ, SmoothQuant papers all use
Domain:           MUST match deployment. A code model calibrated on wiki-text
                  will clip code-specific patterns. Recalibrate on 128 real prompts.
Percentile:       99.9% is the safe default; 99.99% if you see degradation on
                  long-tail tasks; plain absmax if the model is small and the
                  outlier structure is stable.
```

<div class="warnbox">
<strong>The most expensive mistake in quantization:</strong> calibrating on the wrong distribution. The symptom is nasty — average PPL looks fine (the calibration set dominated it), but a specific workload collapses. Always hold out 20–50 real prompts from your actual traffic and check those separately. This has burned more production deployments than any algorithmic subtlety.
</div>

## 4. Four Routes: The Map of This Series

Every method in the literature does exactly one of four things:

<div class="fig">
<svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The four quantization routes">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">Four routes, one question: how to fit the data into the grid</text>

  <!-- A -->
  <rect x="20" y="42" width="640" height="76" rx="6" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>
  <circle cx="46" cy="66" r="13" fill="#1d4ed8"/>
  <text x="46" y="71" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">A</text>
  <text x="70" y="70" font-size="12" font-weight="700" fill="#1d4ed8">Change the rounding</text>
  <text x="70" y="88" font-size="10" fill="#374151">Keep the distribution, minimize the error on a fixed grid.</text>
  <text x="70" y="103" font-size="10" fill="#6b7280">RTN: round to nearest. GPTQ: use the Hessian to compensate the columns you break.</text>
  <text x="560" y="70" font-size="10" fill="#1d4ed8" font-weight="700">ceiling W4A16</text>
  <text x="560" y="86" font-size="9" fill="#6b7280">qt2</text>

  <!-- B -->
  <rect x="20" y="130" width="640" height="76" rx="6" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5"/>
  <circle cx="46" cy="154" r="13" fill="#7c3aed"/>
  <text x="46" y="159" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">B</text>
  <text x="70" y="158" font-size="12" font-weight="700" fill="#7c3aed">Change the distribution — scaling</text>
  <text x="70" y="176" font-size="10" fill="#374151">Move the difficulty from the hard-to-quantize side to the easy one.</text>
  <text x="70" y="191" font-size="10" fill="#6b7280">SmoothQuant: push activation outliers into weights. AWQ: enlarge salient channels instead.</text>
  <text x="560" y="158" font-size="10" fill="#7c3aed" font-weight="700">W8A8 / W4A16</text>
  <text x="560" y="174" font-size="9" fill="#6b7280">qt3</text>

  <!-- C -->
  <rect x="20" y="218" width="640" height="76" rx="6" fill="#f0fdf4" stroke="#15803d" stroke-width="1.5"/>
  <circle cx="46" cy="242" r="13" fill="#15803d"/>
  <text x="46" y="247" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">C</text>
  <text x="70" y="246" font-size="12" font-weight="700" fill="#15803d">Change the distribution — rotation</text>
  <text x="70" y="264" font-size="10" fill="#374151">Spread the spike across all channels with an orthogonal matrix.</text>
  <text x="70" y="279" font-size="10" fill="#6b7280">QuaRot: Hadamard at four insertion points. SpinQuant: learn the rotation. DuQuant: block-diagonal.</text>
  <text x="570" y="246" font-size="10" fill="#15803d" font-weight="700">W4A4</text>
  <text x="570" y="262" font-size="9" fill="#6b7280">qt4</text>

  <!-- D -->
  <rect x="20" y="306" width="640" height="76" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/>
  <circle cx="46" cy="330" r="13" fill="#ea580c"/>
  <text x="46" y="335" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">D</text>
  <text x="70" y="334" font-size="12" font-weight="700" fill="#ea580c">Change the training</text>
  <text x="70" y="352" font-size="10" fill="#374151">Let the model adapt to the grid instead of adapting the grid to the model.</text>
  <text x="70" y="367" font-size="10" fill="#6b7280">QAT: fake-quantize in the forward pass so gradients see the real error.</text>
  <text x="565" y="334" font-size="10" fill="#ea580c" font-weight="700">breaks the PTQ ceiling</text>
  <text x="565" y="350" font-size="9" fill="#6b7280">op2 §6</text>

  <text x="20" y="396" font-size="9" fill="#6b7280">Orthogonal to all four: KV Cache quantization (qt5) — it compresses a runtime artifact, not the model.</text>
</svg>
<p class="cap">Figure 2: The four routes. A does not touch the distribution, B and C do (differently), D does not touch the grid at all. They stack, and production recipes usually do.</p>
</div>

Route B and C share one mathematical foundation — the **equivalence transform**:

```text
Y = X W = (X S⁻¹)(S W)    ← B: S is diagonal, per-channel scaling
Y = X W = (X R)(Rᵀ W)     ← C: R is orthogonal, rotation
```

In both cases the product is unchanged (exactly, not approximately) and only the *numerical distribution* shifts. That is the entire trick behind both families, and it is why neither needs training.

| Route | Touches distribution | Needs calibration | Needs training | Ceiling |
|---|---|---|---|---|
| A | no | GPTQ: yes | no | W4A16 |
| B | yes (scaling) | yes | no | W8A8 / W4A16 |
| C | yes (rotation) | QuaRot: yes, SpinQuant: training | QuaRot: no | **W4A4** |
| D | n/a | n/a | **yes** | FP4 |

## 5. How to Measure Precision: Perplexity Is Not Enough

The standard metric is WikiText perplexity, and it is necessary but badly insufficient:

```text
Problem 1: PPL is a mean over a smooth next-token distribution.
           A model can keep PPL while losing the ability to do arithmetic.

Problem 2: Long context degrades first. 4k PPL unchanged, 32k PPL +15%.
           The KV cache is what breaks, not the weights — see qt5.

Problem 3: The average layer hides the worst layer. If layer 17 collapsed and
           layer 3 improved, the mean looks fine and your model is broken.
           Always dump per-layer relative error against FP16.
```

A minimum useful evaluation suite:

| Metric | What it catches | Threshold to worry |
|---|---|---|
| WikiText PPL | global distribution shift | +5% |
| MMLU / GSM8K | reasoning, arithmetic | −2 points |
| HumanEval | structured output, long generation | −3 points |
| 32k needle-in-haystack | long context, KV degradation | any drop |
| per-layer relative error | which layer broke | one layer > 3× median |

## 6. Selection Cheat Sheet

<div class="keybox">
<strong>Four questions, in this order:</strong><br>
① <strong>What is the bottleneck?</strong> Memory wall → KV quantization (qt5). Bandwidth wall in decode → W4A16 (qt2/qt3). Compute wall in prefill → FP8 or W8A8 (qt3).<br>
② <strong>What does the hardware support natively?</strong> A100 → INT8 / AWQ W4A16. H100 → FP8. B200 → NVFP4. <em>Hardware decides format; format decides algorithm.</em><br>
③ <strong>Do you need to quantize activations?</strong> If no, stop at AWQ. If yes, you need B or C, and the cost jumps.<br>
④ <strong>How much debugging budget do you have?</strong> AWQ: an afternoon. GPTQ: a day. QuaRot: a week, and failures are silent.
</div>

A quick reference for what each method actually costs you:

```text
RTN          seconds     no calibration      accuracy: bad below 8 bit
GPTQ         minutes     Hessian per layer   accuracy: good at 4 bit
AWQ          minutes     alpha grid search   accuracy: best W4A16 per effort
SmoothQuant  seconds     act max stats       accuracy: only for W8A8
QuaRot       minutes     calibration + fuse  accuracy: W4A4 possible
SpinQuant    hours       train the rotation  accuracy: best W4A4
KIVI         online      K offline, V dynamic accuracy: KV4 usable
```

## 7. Series Navigation

- Next **qt2 · RTN and GPTQ**: route A. Why naive rounding fails, the OBS framework from 1993, and the full GPTQ derivation with runnable pseudocode.
- **qt3 · SmoothQuant and AWQ**: route B. Two opposite answers to the same activation-outlier problem.
- **qt4 · QuaRot / SpinQuant / DuQuant**: route C. Why rotation is free, where it can be inserted (R1–R4), and why W4A4 is the only route that gets there.
- **qt5 · KV Cache Quantization and FP4**: the orthogonal dimension, plus how FP8 and NVFP4 change the whole selection logic.

Related on this site: **op2** is the series index and lineage table; **op3 (Ascend EagleQBSA)** is a real deployment where these choices had to be made per-tensor-role; **fw4** covers how quantization kernels are fused in actual engines.
