---
title: 'Inference Systems Infrastructure Notes (9): The KV Cache Landscape — Why It Decides Your Throughput, Latency and Concurrency'
description: 'Starting from redundant computation: why autoregressive generation can be cached and where the boundary is; how large KV Cache really is (32 GiB measured on Qwen3-32B); how it eats TTFT, TPOT and concurrency; plus the three-layer optimization taxonomy (token / model / system) and the five knobs: sequence length, head count, key bits, head dimension, and layers.'
pubDate: 2026-09-09
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys9-kv-cache-fundamentals-optimization
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

KV Cache is **space traded for time**: it pays once to compute the repeated work in autoregressive generation and caches the result, at the cost of GPU memory growing linearly with batch times sequence length, which eventually throttles your concurrency. This post covers three things — **why it can be cached at all, how big it actually gets, and which directions the industry is cutting it from**.

## 1. Meet the Enemy: Without KV Cache, What Are We Recomputing?

LLM inference is autoregressive: generating token n requires information from all previous n−1 tokens. Naively, every new token re-runs the **whole history** through the forward pass:

| Module | Where the redundancy lives |
|--------|----------------------------|
| Embedding | Historical token embeddings regenerated every step |
| K / V generation | K and V for historical tokens recomputed every step |
| QKᵀ | The attention score matrix recomputed, growing quadratically with sequence length |
| Softmax × V | The weighted sum recomputed |

Total cost for N tokens lands around **O(N²)** — the longer the sequence, the worse it gets, and it gets ugly fast.

<div class="fig">
<svg viewBox="0 0 640 250" role="img" aria-label="KV Cache behaviour during prefill and decode">
  <title>Figure 1: KV Cache during prefill and decode</title>
  <text x="16" y="26" font-size="13" fill="#6b7280">prefill: the whole prompt in one pass, emits the first token</text>
  <rect x="16" y="44" width="46" height="38" rx="4" fill="#f6f8fa" stroke="#30363d"/>
  <text x="39" y="69" font-size="13" fill="#1a1a1a" text-anchor="middle">tok1</text>
  <rect x="68" y="44" width="46" height="38" rx="4" fill="#f6f8fa" stroke="#30363d"/>
  <text x="91" y="69" font-size="13" fill="#1a1a1a" text-anchor="middle">tok2</text>
  <rect x="120" y="44" width="46" height="38" rx="4" fill="#f6f8fa" stroke="#30363d"/>
  <text x="143" y="69" font-size="13" fill="#1a1a1a" text-anchor="middle">tok3</text>
  <line x1="172" y1="63" x2="206" y2="63" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="206,63 198,59 198,67" fill="#2563eb"/>
  <rect x="214" y="38" width="196" height="50" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="312" y="30" font-size="12" fill="#2563eb" text-anchor="middle">KV Cache, 3 slots</text>
  <rect x="224" y="50" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <rect x="282" y="50" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <rect x="340" y="50" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <line x1="416" y1="63" x2="450" y2="63" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="450,63 442,59 442,67" fill="#2563eb"/>
  <rect x="458" y="44" width="46" height="38" rx="4" fill="#ecfdf5" stroke="#10b981"/>
  <text x="481" y="69" font-size="13" fill="#047857" text-anchor="middle">out</text>
  <text x="16" y="146" font-size="13" fill="#6b7280">decode: one token per step, but attention reads the entire history</text>
  <rect x="16" y="164" width="46" height="38" rx="4" fill="#f6f8fa" stroke="#30363d"/>
  <text x="39" y="189" font-size="13" fill="#1a1a1a" text-anchor="middle">tok4</text>
  <line x1="68" y1="183" x2="206" y2="183" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="206,183 198,179 198,187" fill="#2563eb"/>
  <rect x="214" y="158" width="262" height="50" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="345" y="150" font-size="12" fill="#2563eb" text-anchor="middle">KV Cache, 4 slots (+1)</text>
  <rect x="224" y="170" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <rect x="282" y="170" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <rect x="340" y="170" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <rect x="398" y="170" width="50" height="26" rx="3" fill="#dcfce7" stroke="#10b981"/>
  <line x1="482" y1="183" x2="516" y2="183" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="516,183 508,179 508,187" fill="#2563eb"/>
  <rect x="524" y="164" width="46" height="38" rx="4" fill="#ecfdf5" stroke="#10b981"/>
  <text x="547" y="189" font-size="13" fill="#047857" text-anchor="middle">out</text>
  <text x="16" y="234" font-size="12" fill="#6b7280">Green marks what this step adds. Decode work per step is constant (one token), but the KV it reads grows linearly — the source of memory-bound behaviour.</text>
</svg>
<figcaption>Figure 1: prefill builds the KV Cache for the whole prompt in one shot; decode adds one token per step, and attention reads the entire history.</figcaption>
</div>

## 2. Why Caching Is Legal: Module by Module

Caching rests on two properties: **causal masking** and **position independence**.

### Attention: causal masking freezes the past

- At inference, the query at position i only attends to 1..i — **it never sees the future**. Under unidirectional attention, the k₂ computed at `i=3` is **identical** to the k₂ computed at `i=4`.
- Put simply, K/V for historical tokens are already final. There is no reason to recompute them. Each step only needs K/V for **the newest token**, appended to the cache.

### FFN / LayerNorm / Linear: positions do not interact

- **FFN**: features at different positions do not mix, so output i depends only on input i (Y₁ depends only on X₁).
- **LayerNorm**: statistics run along d_model, so the output depends only on the current row of hidden states.
- **Linear (lm_head)**: by the nature of matrix multiplication, the last row of logits depends only on the last row of hidden states.
- **Softmax**: keeping prior results lets you merge with the new one.

### The one-line mathematical backing

Matrix multiplication is block-separable: split A into `[:s]` and `[s:]`, multiply each by B, concatenate — identical to multiplying A by B whole. Attention and FFN are both matrix multiplications, so caching `[:s]` and computing only the new `[s:]` is **exactly equivalent**. This is a lossless optimization.

## 3. Prefill vs Decode: Two Stages, Opposite Personalities

| Stage | What it does | Compute profile | Key metric |
|-------|--------------|-----------------|------------|
| **prefill** | Processes the whole prompt, builds the KV Cache, emits the first token | **Compute-bound** (large GEMMs, high parallelism) | TTFT |
| **decode** | Emits one token at a time, reading the full history each step | **Memory-bandwidth-bound** (little compute, lots of KV traffic) | TPOT |

This split — one wants FLOPs, the other wants bandwidth — is the entire reason PD disaggregation exists (see sys8).

## 4. How Big Does It Actually Get

The formula for standard MHA/GQA:

```
KV Cache = 2 (K and V) x layers x batch x seq_len x kv_dim x dtype_bytes
```

where `kv_dim = num_kv_heads x head_size`.

**Measured case (Qwen3-32B)**:

- 64 layers, GQA with effective KV dimension 1024, BF16
- At batch=4, seq=32768:
  `2 x 64 x 4 x 32768 x 1024 x 2B = 32 GiB`
- Raise batch to 8 → **64 GiB**, already comparable to the model weights themselves (roughly 64 GB in BF16)

**Takeaway**: KV Cache grows **linearly** with batch and sequence length. Under high concurrency and long contexts, it — not the weights — is what caps your serving concurrency.

## 5. How It Eats Performance: Three Metrics, Three Memory Regions

| Metric | Meaning | Driven by |
|--------|---------|-----------|
| **TTFT** | Time to first token | Mostly prefill cost; prefix cache hits cut it sharply |
| **TPOT** | Time per output token | Per-step decode cost; rises as KV grows (more traffic) |
| **Throughput** | Output tokens per second (the cost metric) | Larger batches help, until memory runs out |

Memory view:

```
peak memory = model weights (fixed) + KV Cache (grows with batch x seq) + activations
total latency = TTFT + TPOT x number of generated tokens
```

So shrinking KV Cache pays twice: **more concurrency on the same card** (memory saved) and **faster requests** (bandwidth saved).

## 6. The Optimization Landscape: Three Layers

The survey *A Survey on Large Language Model Acceleration based on KV Cache Management* organizes the field into three layers.

### 6.1 Token-level (no model change, no parallelism change)

Selecting, organizing, and compressing at token granularity:

| Direction | Approach | Examples |
|-----------|----------|----------|
| **Selection** | Keep only the most important tokens | H2O (heavy hitters), Keyformer, SnapKV, Quest |
| **Budget allocation** | Distribute cache budget across layers / heads | PyramidKV, PyramidInfer, AdaKV, DuoAttention |
| **Merging** | Merge similar or overlapping KV pairs | Intra / inter-layer merging, Prompt Cache |
| **Quantization** | Lower storage precision (FP16 to INT8 / INT4 / FP8) | KV cache quantization, see op2 |
| **Low-rank decomposition** | Compress KV matrices into lower-dimensional form | LoRA-style compression, one of the ideas behind MLA |

### 6.2 Model-level (architectural changes)

- **Attention grouping and sharing**: MQA (one shared KV head), GQA (grouped sharing)
- **Architecture redesign**: MLA (low-rank latent compression, see op1), YOCO, CLA, MLKV, NSA
- **Non-Transformer architectures**: linear attention, RWKV, Mamba — no KV Cache to speak of

### 6.3 System-level (scheduling and memory management)

- **Memory management**: PagedAttention (paging), virtual-memory adaptation, prefix sharing (RadixAttention)
- **Scheduling**: prefix-aware scheduling to raise hit rate, preemptive context switching
- **Hardware-aware design**: GPU / CPU / SSD tiered offloading (HiCache, AttentionStore, LMCache)

## 7. Five Knobs: Turning "Shrink the KV Cache" Into Quantifiable Actions

Going back to the formula, KV Cache size is a product of five factors, and each factor is an independent entry point:

| Knob | How to turn it | Cost |
|------|----------------|------|
| **Sequence length** | Sparsity (static window / dynamic eviction), prefix reuse | May drop critical tokens; long-range tasks degrade |
| **Head count** | MQA / GQA (fewer KV heads) | Slight expressiveness loss, now a mainstream default |
| **key_bits** | Quantize to INT8 / INT4 / FP8 | Accuracy loss, needs calibration (see op2) |
| **Head dimension** | MLA compresses KV into a low-rank latent; Double Sparsity exploits channel sparsity | Large structural change |
| **Layers** | Cache only some layers (YOCO / CLA / MLKV / LayerSkip) | Needs training support or accepts approximation |

**How to read this**: the knobs are not exclusive. Production stacks stack them — for example, "GQA (heads) + FP8 KV (bits) + prefix reuse (length) + PagedAttention (system)" is the most common combination today.

## 8. Links to the Rest of the Series

- This post is **fundamentals and taxonomy**: what KV Cache is, why it works, and how it can be cut.
- **sys5** covers the **evolution thread** of compression directions (MHA to MSA to HCA) — how the routes changed over time.
- **sys6 (HiSparse)**, **sys7 (DCP)**, and **sys1 (HiCache)** are three concrete system-level deployments.
- **op1 (MLA)** and **op2 (quantization)** map to the model-level and quantization knobs.
- **sys10**, the next post, moves from principles to **engineering implementation**: SGLang's two-level memory pool, vLLM's block management, and Baidu's AttentionStore measurements on Kunlunxin hardware.
