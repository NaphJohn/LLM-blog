---
title: "Operator Explainers (2): Model Quantization — from the precision spectrum to the four routes"
description: "The index page for the Quantization Notes series (qt1–qt5). Keeps the original precision spectrum, QAT discussion, KV cache notes, and inference-engine checklist, and adds a complete method-lineage table grouping RTN, GPTQ, AWQ, SmoothQuant, QuaRot, SpinQuant, DuQuant, and KIVI into four orthogonal routes with their bit-width ceilings, compute costs, calibration requirements, and which pairs stack. Jump to the note that covers the method you need."
pubDate: 2026-09-08
series: Operator Explainers
lang: en
altLang: zh
altHref: /blog/op2-model-quantization
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

Model quantization is the art of trading precision for memory/bandwidth/compute — it is **orthogonal** to *distillation* (which trades capacity for size); the two stack (distill a small model, then quantize it for deployment).

This post is the **index**: it gives the full spectrum and the selection checklist. Each method is expanded in the Quantization Notes series.

<div class="keybox">
<strong>Series index:</strong><br>
<strong>qt1 · The Basic Language</strong> — quantization formula, granularity (per-tensor/channel/group/token), symmetry, calibration, the four-route map<br>
<strong>qt2 · RTN and GPTQ</strong> — route A "change the rounding", including the full OBS derivation and working GPTQ pseudocode<br>
<strong>qt3 · SmoothQuant and AWQ</strong> — route B "change the distribution by scaling", two opposite answers to activation outliers<br>
<strong>qt4 · QuaRot / SpinQuant / DuQuant</strong> — route C "change the distribution by rotation", the only route to W4A4, four insertion points R1–R4<br>
<strong>qt5 · KV Cache Quantization and FP4</strong> — the orthogonal axis, and how FP8/FP4 formats rewrite the selection logic
</div>

## 1. Precision Spectrum and Compression

| Precision | Bits | vs FP16 | Typical loss | Arena |
|-----------|------|---------|--------------|-------|
| FP16 / BF16 | 16 | 1× | baseline | training, high-fidelity inference |
| FP8 (E4M3 / E5M2) | 8 | **2×** | ~0.3% | weights + activations, KV cache |
| INT8 | 8 | 2× | slightly above FP8 | edge, older hardware |
| INT4 (W4A16) | 4 | **4×** | ~0.5% | on-device / single-card, current default |
| INT4 (W4A4) | 4 | 4× | needs rotation compensation | memory-ceiling scenarios |
| FP4 (NVFP4/MXFP4) | 4.5 | 3.6× | ~0.1% (native hardware) | Blackwell and later |

- **BF16 vs FP16**: BF16 has more exponent bits and a wider dynamic range, which trains more stably; FP16 has finer mantissa but overflows more easily.
- **Two FP8 formats**: `E4M3` (4 exponent, 3 mantissa, range ±448 — good for weights/activations and numerically friendly) vs `E5M2` (5 exponent, 2 mantissa, range ±57344 — good for long text and multilingual). The choice directly determines the accuracy drop. Details in **qt5 §3.1**.
- **Why FP4 is 4.5 bits**: the microscaling format shares one 8-bit scale per 16 or 32 elements, about 0.5 bit amortised. That 0.5 bit buys dynamic range, and it is a very good trade — see **qt5 §3.2**.

## 2. Complete Method Lineage

This is the core table of the post. Every method is classified by **what it actually changes**:

| Route | What changes | Methods | Bit ceiling | Compute | Calibration | Stacks with |
|---|---|---|---|---|---|---|
| **A** | **rounding** only | RTN | W4A16 | seconds | no | anything |
| **A** | **rounding** only | GPTQ | W4A16 / W4A8 | minutes–hours | yes (Hessian) | AWQ, rotation |
| **B** | distribution · **scaling** | SmoothQuant | **W8A8** | seconds | yes (act max) | GPTQ |
| **B** | distribution · **scaling** | AWQ | W4A16 | minutes | yes (mean\|X\|) | GPTQ |
| **C** | distribution · **rotation** | QuaRot / SpinQuant / DuQuant | **W4A4** | minutes (+training) | yes / needs training | can precede A/B |
| **D** | **training** | FP4 QAT | W4A4 / FP4 | hours–days | needs a training stack | orthogonal to all |
| **KV** | orthogonal axis | KIVI / FP8 KV / KV eviction | KV4–KV8 | online (tiny) | K profiled offline | stacks with everything |

Three sentences read the table:

```text
① Route A (RTN/GPTQ) does not change the data distribution; it minimises error
   on a fixed grid → the ceiling is set by the grid, usually W4A16
② Routes B/C change the distribution so the grid covers it better → B can only
   rescue activations, C can flatten spikes → C reaches W4A4
③ Route D makes the model adapt to the grid → breaks the PTQ ceiling, but costs
   a training run
```

**One misconception worth correcting**: many sources describe AWQ as a replacement for GPTQ. That is wrong. They are strongly orthogonal — AWQ decides `s` (the distribution), GPTQ decides `round` (the rounding). The production recipe of "AWQ for scales, then GPTQ for rounding" gets both benefits (see **qt3 §4**).

## 3. Intuition for the Four Routes

<div class="fig">
<svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="How the four routes handle the same data">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">The same data (with one outlier), handled four ways</text>

  <text x="20" y="52" font-size="11" font-weight="700" fill="#1a1a1a">Original activations X (red bar is the outlier)</text>
  <g>
    <rect x="30" y="62" width="8" height="20" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="42" y="62" width="8" height="24" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="54" y="62" width="8" height="18" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="66" y="62" width="8" height="22" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="78" y="72" width="8" height="10" fill="#dc2626" stroke="#991b1b"/>
    <rect x="90" y="62" width="8" height="21" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="102" y="62" width="8" height="19" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="114" y="62" width="8" height="23" fill="#bfdbfe" stroke="#60a5fa"/>
  </g>
  <text x="132" y="80" font-size="10" fill="#b91c1c">absmax = 118 → INT4 rounds everything else to zero</text>

  <line x1="20" y1="98" x2="660" y2="98" stroke="#e5e7eb"/>

  <text x="20" y="120" font-size="11" font-weight="700" fill="#1d4ed8">A · RTN / GPTQ: grid unchanged, rounding changed</text>
  <g>
    <rect x="30" y="130" width="8" height="20" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="42" y="130" width="8" height="24" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="54" y="130" width="8" height="18" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="66" y="130" width="8" height="22" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="78" y="140" width="8" height="10" fill="#dc2626" stroke="#991b1b"/>
    <rect x="90" y="130" width="8" height="21" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="102" y="130" width="8" height="19" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="114" y="130" width="8" height="23" fill="#dbeafe" stroke="#60a5fa"/>
  </g>
  <text x="132" y="140" font-size="10" fill="#374151">the outlier remains → ceiling W4A16 (leave activations unquantized)</text>
  <text x="132" y="154" font-size="10" fill="#6b7280">GPTQ compensates sensitive channels with the Hessian, but "who is large" is unchanged</text>

  <text x="20" y="180" font-size="11" font-weight="700" fill="#7c3aed">B · SmoothQuant / AWQ: rescale elsewhere</text>
  <g>
    <rect x="30" y="190" width="8" height="22" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="42" y="190" width="8" height="26" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="54" y="190" width="8" height="20" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="66" y="190" width="8" height="24" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="78" y="190" width="8" height="16" fill="#f59e0b" stroke="#b45309"/>
    <rect x="90" y="190" width="8" height="23" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="102" y="190" width="8" height="21" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="114" y="190" width="8" height="25" fill="#ede9fe" stroke="#a78bfa"/>
  </g>
  <text x="132" y="200" font-size="10" fill="#374151">SQ moves the outlier into the weights (which are already uniform)</text>
  <text x="132" y="214" font-size="10" fill="#6b7280">AWQ enlarges the salient channel's weight instead of moving it → ceiling W4A16 / W8A8</text>

  <text x="20" y="240" font-size="11" font-weight="700" fill="#15803d">C · QuaRot / SpinQuant / DuQuant: rotate and flatten</text>
  <g>
    <rect x="30" y="250" width="8" height="20" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="42" y="250" width="8" height="21" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="54" y="250" width="8" height="19" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="66" y="250" width="8" height="22" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="78" y="250" width="8" height="20" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="90" y="250" width="8" height="21" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="102" y="250" width="8" height="19" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="114" y="250" width="8" height="22" fill="#bbf7d0" stroke="#4ade80"/>
  </g>
  <text x="132" y="260" font-size="10" fill="#374151">every bar is the same height (energy spread over d channels)</text>
  <text x="132" y="274" font-size="10" fill="#15803d" font-weight="700">the only route that reaches W4A4</text>

  <text x="20" y="300" font-size="11" font-weight="700" fill="#ea580c">D · QAT: the model learns to live with the grid</text>
  <g>
    <rect x="30" y="310" width="8" height="21" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="42" y="310" width="8" height="22" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="54" y="310" width="8" height="20" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="66" y="310" width="8" height="23" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="78" y="310" width="8" height="18" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="90" y="310" width="8" height="21" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="102" y="310" width="8" height="22" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="114" y="310" width="8" height="20" fill="#ffedd5" stroke="#fb923c"/>
  </g>
  <text x="132" y="320" font-size="10" fill="#374151">fake-quantize in the forward pass so gradients see the real error</text>
  <text x="132" y="334" font-size="10" fill="#6b7280">the outlier is absorbed during training → breaks the PTQ ceiling</text>

  <rect x="20" y="352" width="640" height="38" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="32" y="369" font-size="10" fill="#1d4ed8" font-weight="700">A leaves the distribution alone · B relocates the difficulty · C flattens the spike · D trains for it</text>
  <text x="32" y="383" font-size="10" fill="#1d4ed8">All four are orthogonal and can be stacked. Production typically does B+C+A: rotate, then AWQ for scales, then GPTQ for rounding.</text>
</svg>
<p class="cap">Figure 1: The same data under four routes. Only route C actually makes all bars the same height — which is exactly why it can reach W4A4.</p>
</div>

## 4. Post-Training Quantization (PTQ)

PTQ quantizes an already-trained model; the central question is how to allocate scale factors. Each method belongs to a route:

- **RTN (round to nearest)**: elementwise rounding, fastest, but large activation channels get crushed. → **qt2 §1**
- **GPTQ**: layer-wise OBS-style second-order quantization — when quantizing a column, use the Hessian to compensate the remaining columns, minimizing `‖WX − ŴX‖²`. Suitable for offline compression on GPU. → **qt2 §2–3**
- **SmoothQuant**: `s = amax_X^α / amax_W^(1−α)`, moving activation outliers into weights. **Exists specifically for W8A8**. → **qt3 §2**
- **AWQ (activation-aware)**: computes salience `s = mean(|X|)^α`, enlarges the weights of important channels before quantizing, with almost no overhead. The current mainstream for LLM on-device deployment. → **qt3 §3**
- **per-group quantization**: `group_size=128` is the industry default granularity, balancing fine-grained fidelity against lookup overhead. → **qt1 §2**

## 5. Rotation-Based Quantization: The W4A4 Branch

It changes the **distribution**, not the **rounding**. For any orthogonal matrix `R` (`R Rᵀ = I`):

```text
XW = (X R)(Rᵀ W)
```

The product is unchanged, but the numerical distribution is stirred — activation outliers concentrated in a few channels get flattened across all channels, so 4-bit quantization is no longer blown out by the outlier.

Three representatives:

- **QuaRot (NeurIPS 2024)**: randomised Hadamard plus random permutation, applied at four positions (R1 residual stream / R2 attention output / R3 after SwiGLU / R4 Q·K), first to achieve **W4A4KV4** on LLaMA-2/3 with no training.
- **SpinQuant (arXiv:2405.16406)**: upgrades the rotation from "random and fixed" to "learnable" — Cayley-parameterised to remain orthogonal, optimised on the Stiefel manifold, recovering substantially more accuracy than QuaRot on LLaMA-3.
- **DuQuant**: shows that a global Hadamard **creates new outliers**, so it uses block-diagonal rotation plus zigzag channel permutation, more stable than both predecessors on small models and stubborn outliers (LLaMA-3, Qwen2).

<div class="warnbox">
<strong>An important practical correction:</strong> on general-purpose GPUs the rotation route buys <strong>memory, not speed</strong> — H100/H200 Tensor Cores have no native INT4×INT4 MMA, so W4A4 must unpack and recompute and is not faster than W4A16. Only Blackwell's native NVFP4 genuinely unlocks W4A4 throughput. Full analysis in <strong>qt4 §4</strong> and <strong>qt5 §3.3</strong>.
</div>

> VLA in practice: OpenVLA quantized to INT4 comes out at about 4 GB with near-lossless success rate, runnable on Jetson Orin, leaving headroom for the perception/planning/control stack — the necessary path for putting large models into the physical world.

## 6. Quantization-Aware Training (QAT)

PTQ degrades noticeably at INT4/FP4, which requires QAT:

- **FP4 QAT (the DeepSeek V4 / V4-Flash route)**: fake-quantize the MoE routed expert weights and the CSA indexer's QK path to FP4 — **quantize to FP4 in the forward pass, dequantize back to high precision for compute**, with sampling/rollout using native FP4. Gradients then see the true post-quantization distribution, so deployment matches training.
- Prerequisites: mixed-precision training (BF16 AMP) plus FP8 KV cache. QAT typically also pairs with the Muon optimizer and tricks like SwiGLU clamping to bring 4-bit degradation into a usable range.

## 7. KV Cache Quantization and Eviction

At long context the bottleneck is not compute but **KV cache memory**. Three orthogonal paths:

- **Quantization (compress precision)**: FP16→FP8 gives 2× for ~0.3% loss; FP16→INT4 requires noting that **K is an order of magnitude more sensitive than V** (softmax amplification), and KIVI's answer is K per-channel + V per-token + a full-precision residual window of 128 tokens. → **qt5 §1–2**
- **Eviction (reduce token count)**: H2O / ThinKV and similar prune the cache by importance, cutting a further 3–244×.
- **Combined frontier**: RDKV (2026) unifies quantization and eviction under a rate-distortion objective, 2.48% cache retained for 97.81% accuracy; ThinKV (ICLR 2026 Oral) does thought-adaptive KV compression, near-lossless below 5% retention at 5.8× throughput.
- **Pitfalls**: E4M3 vs E5M2 selection, the practical INT2 floor, quantization+eviction needing joint tuning, and H2O's extra 5–10% compute. → the PagedAttention block-alignment constraint in **qt5 §1.3**

## 8. Landing in Inference Engines and Speculative Decoding

- **vLLM / SGLang**: FP8 KV cache plus AWQ is where quantization actually lands in serving engines; HiSparse (see sys6 on this site) goes further, treating HBM as a cache with host DRAM holding the authoritative full KV — the "quantize + tier" idea extended to the system level.
- **Speculative decoding**: quantization frees memory headroom for speculation — a compressed KV cache allows a larger batch and a heavier draft model, so end-to-end speedup is amplified.
- **Training side**: Megatron-LM `--fp8-format hybrid` with BF16 AMP is the current normal configuration for large-model training.

## 9. Deployment Checklist

1. **Locate the bottleneck first**: memory wall → KV cache quantization/eviction; bandwidth wall → INT4/AWQ weights; compute wall → FP8 forward.
2. **Let the hardware pick the format**: A100 → INT8 / AWQ W4A16; H100 → FP8 (no SmoothQuant needed); B200 → NVFP4 (no rotation needed). **Hardware decides format; format decides algorithm.**
3. **Prefer FP8 for accuracy** (2×, ~0.3%), and only go to INT4 (AWQ, 4×) for on-device or on-board.
4. **Use W4A4 only under a memory ceiling**, and know that it brings no speed benefit on general-purpose GPUs.
5. **Prefer native FP4 hardware** (B200); otherwise FP4 must come with QAT.
6. **Long text → E5M2; code/math → E4M3.**
7. **Below INT2 is essentially unusable** — do not trade usability for a headline number.

**Tying it together**: distillation shrinks capacity, quantization compresses precision, KV eviction cuts length, tiered caching (HiSparse) puts all three into one memory budget — that is the full toolbox for squeezing 70B+ models into single-card long-context serving today.

*Note: this post consolidates the 2026-07-18 (KV Cache quant+evict), 2026-08-01 (FP8/INT4 deploy), and 2026-08-04 (FP4 QAT) daily technical tracks. On 2026-10-27 the deep dives were split into the five-part Quantization Notes series, and this post became the index and lineage table.*
