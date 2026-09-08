---
title: 'Operator Explainers (2): Model Quantization for Deployment — the precision-compression spectrum from FP16 to FP4'
description: 'A systematic thread tying together the scattered quantization notes: FP8 (E4M3/E5M2), INT4 (AWQ/GPTQ), FP4 quantization-aware training, KV Cache quantization and eviction, plus deployment choices and inference-engine landing.'
pubDate: 2026-09-08
series: Operator Explainers
lang: en
altLang: zh
altHref: /blog/op2-model-quantization
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

Model quantization is the art of trading precision for memory/bandwidth/compute — it is **orthogonal** to *distillation* (which trades capacity for size); the two stack (distill a small model, then quantize it for deployment). This post consolidates the FP8/INT4/FP4/KV-Cache quantization notes that were scattered across daily tracking into one spectrum, landing on real vLLM/SGLang and speculative-decoding deployments.

## 1. Precision Spectrum and Compression

| Precision | Bits | vs FP16 | Typical loss | Arena |
|-----------|------|---------|--------------|-------|
| FP16 / BF16 | 16 | 1× | baseline | training, high-fidelity inference |
| FP8 (E4M3 / E5M2) | 8 | **2×** | ~0.3% | weights+acts, KV Cache |
| INT8 | 8 | 2× | slightly above FP8 | edge, older HW |
| INT4 (mixed) | 4 | **4×** | ~0.5% | on-device / onboard |
| FP4 | 4 | 4× | needs QAT | next-gen train/infer |

- **BF16 vs FP16**: BF16 has more exponent bits (stable training range); FP16 has finer mantissa but overflows easily.
- **Two FP8 formats**: `E4M3` (4 exp / 3 mant, good for weights/acts, code-math friendly) vs `E5M2` (5 exp / 2 mant, larger range, good for long-text/multilingual) — the choice alone moves the accuracy drop.

## 2. Post-Training Quantization (PTQ): compression without gradients

PTQ quantizes a trained model directly; the core question is "how to assign scaling factors":

- **RTN (round-to-nearest)**: per-element round; fastest but large-activation channels get crushed.
- **GPTQ**: per-layer OBS-style second-order quantization — when quantizing one column, compensate the others via the Hessian, minimizing ‖WX − ŴX‖². Good for offline GPU layer-by-layer compression.
- **AWQ (activation-aware)**: compute activation saliency s = mean(|X|)^α, "amplify" important channels before quantizing; near-zero overhead, excellent fidelity on large-activation channels. The current LLM on-device default.
- **per-group**: `group_size=128` is the industry default granularity, balancing fidelity and lookup-table cost.

> VLA in practice: OpenVLA INT4 ≈ 4GB with near-lossless accuracy runs on Jetson Orin, leaving headroom for perception/planning/control — the only way "big models enter the physical world."

## 3. Quantization-Aware Training (QAT): put the error into backprop

PTQ loss grows at INT4/FP4, so QAT is needed:

- **FP4 QAT (DeepSeek V4 / V4-Flash line)**: apply FP4 pseudo-quantization to MoE routed-expert weights and the CSA indexer QK path — **pseudo-quantize to FP4 during training then dequantize back to high precision for the compute**, while sampling/rollout uses native FP4. Gradients thus see the true post-quant distribution; deployment is WYSIWYG.
- Prerequisite: mixed-precision training (BF16 AMP) + FP8 KV Cache. QAT usually pairs with Muon optimizer and SwiGLU Clamping to keep 4-bit drops usable.

## 4. KV Cache quantization and eviction: the long-context memory lifeline

The main bottleneck of long-context inference is not FLOPs but **KV Cache memory**. Two orthogonal paths:

- **Quantize (compress precision)**: FP16→FP8(E4M3/E5M2) 2× ~0.3%; FP16→INT4 mixed (K symmetric + V asymmetric) 4× ~0.5%; per-group(128) mainstream.
- **Evict (cut token count)**: H2O / ThinKV prune by importance, another 3~244×.
- **Frontier stacking**: RDKV (2026) unifies quant+eviction under rate-distortion, 2.48% cache → 97.81% accuracy; ThinKV (ICLR 2026 Oral) thought-adaptive KV compress, <5% KV kept near-lossless, 5.8× throughput.
- **Pitfalls**: E4M3 vs E5M2 choice, INT2 practical floor, quant+evict tuning, H2O's extra 5~10% compute.

## 5. Landing on inference engines and speculative decoding

- **vLLM / SGLang**: FP8 KV Cache + AWQ deployment is where quantization meets the engine; HiSparse (see sys6) pushes "HBM as cache, host DRAM as authoritative KV" — a system-level extension of the quant+tier idea.
- **Speculative decoding**: quantization frees KV headroom — bigger batch, heavier draft model, more end-to-end speedup.
- **Training side**: Megatron-LM `--fp8-format hybrid` + BF16 AMP is now standard for large-model training.

## 6. Deployment checklist

1. Locate the bottleneck first: memory wall → KV quant/evict; bandwidth wall → INT4/AWQ weights; compute wall → FP8 forward.
2. Prefer FP8 (2×, ~0.3%) ; drop to INT4 (AWQ, 4×) only on-device.
3. FP4 must come with QAT, or only for drop-insensitive scenarios.
4. Long text → E5M2; code/math → E4M3.
5. Below INT2 is basically unusable — don't trade usability for a headline number.

**Tying it together**: distillation shrinks capacity, quantization compresses precision, KV eviction cuts length, tiered caching (HiSparse) puts all three into one memory budget — that is the full toolbox for squeezing 70B+ models into single-card long-context serving today.

*Note: this post consolidates the 2026-07-18 (KV Cache quant+evict), 2026-08-01 (FP8/INT4 deploy), 2026-08-04 (FP4 QAT) daily technical tracks into a systematic version.*
