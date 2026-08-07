---
title: 'Operator Notes (1): MLA — Multi-head Latent Attention at the Operator Level'
description: 'Pull Multi-head Latent Attention (MLA) from concept to operator. Line-by-line pseudocode for c_kv / W_DKV / W_UK / W_UV, quantifying how it shrinks per-token KV cache from 2·n_h·d_h down to d_c (~1/64 on DeepSeek-V2), and why inference caches only the single low-rank latent c_kv. Opens the "Operator Notes" mini-series; later posts cover more attention / normalization / routing operators you can actually implement.'
pubDate: 2026-08-07
series: Operator Notes
lang: en
altLang: zh
altHref: /blog/op1-mla-operator
layout: ../../../layouts/BlogPost.astro
---

## 0. Why This "Operator Notes" Mini-Series

Earlier posts (mm5 efficient systems, fa4 DeepSeek V4) treated MLA as a *concept*: K/V are jointly compressed into a low-rank latent c, so the KV cache shrinks dramatically. But *how* exactly is it computed, what is actually stored in the cache, and how much smaller is it — those only become clear at the **operator level**.

This mini-series does exactly that: take the operators frontier architectures gloss over and write them as line-by-line pseudocode you could implement directly. We open with MLA because it is the shared foundation of the 2026 "compress KV"主线 (mm5 / fa4) and the first brick in understanding DeepSeek-family inference cost reduction.

## 1. What MLA Does (One-Line Recap)

Standard multi-head attention (MHA) stores the full K and V for every token in the KV cache:

- K dim = n_h · d_h (heads × head dim)
- V dim = n_h · d_h
- per-token cache = 2 · n_h · d_h

At n_h=128, d_h=128, that is 32768 numbers per token. At 1M context that is 32.77 billion numbers — the very source of "long-context cost explodes quadratically."

MLA (Multi-head Latent Attention) does not store K/V directly. Instead it **jointly projects K/V into a low-rank latent space** and caches only that latent c_kv. At inference, two up-projections restore K and V from c_kv.

## 2. Line-by-Line Operator Pseudocode

Dimension conventions (DeepSeek-V2 style):

- d_model: model hidden dim
- n_h: number of heads, d_h: per-head dim
- d_c: latent dim (post-KV-compression; far smaller than n_h·d_h)

```python
# Down-projection (KV compression) -- the only matrix that enters the KV cache
W_DKV : [d_model, d_c]         # compress x into a low-rank latent
# Up-projections -- restore K, V from the latent
W_UK  : [d_c, n_h * d_h]       # c_kv -> K
W_UV  : [d_c, n_h * d_h]       # c_kv -> V
# Query side (simplest version; real MLA also compresses Q to c_q first, see §4)
W_Q   : [d_model, n_h * d_h]
# Output projection
W_O   : [n_h * d_h, d_model]

# ---- Per-token Prefill ----
c_kv = x @ W_DKV              # [B, T, d_c]   <- the ONLY KV state cached

# ---- Restore K, V from c_kv (not cached; computed on demand) ----
K = c_kv @ W_UK               # [B, T, n_h * d_h]
V = c_kv @ W_UV               # [B, T, n_h * d_h]

# ---- Standard scaled-dot-product attention ----
Q     = x @ W_Q               # [B, T, n_h * d_h]
scores = Q @ K^T / sqrt(d_h)  # [B, n_h, T, T]
attn   = softmax(scores, axis=-1)
O     = attn @ V              # [B, T, n_h * d_h]
out   = O @ W_O               # [B, T, d_model]
```

Key point: **of the entire chain, the only thing that enters the KV cache is `c_kv` (d_c dims)**. `K` and `V` are "restored" right before attention and discarded after — they never occupy the cache.

## 3. How Much the Cache Shrinks

| Scheme | Per-token cache dim | 1M context (n_h=128, d_h=128) |
|---|---|---|
| Standard MHA | 2 · n_h · d_h = 32768 | ~32.77 billion |
| MLA (d_c=512) | d_c = 512 | ~0.51 billion |

The cache drops to about **1/64** (~98.4% reduction). This is what mm5 calls "compress the width": the **width** of KV is compressed to d_c, while attention quality is recovered by the up-projections W_UK / W_UV at restore time. The cost is one extra up-projection's compute (well worth the saved KV memory and bandwidth).

<div class="keybox">
<strong>Key:</strong> What MLA saves is <strong>memory and bandwidth</strong> (the KV cache), not compute. It trades "store full K/V" for "store a low-rank latent + two small matmuls," moving long-context cost from O(n·d_kv) down to O(n·d_c).
</div>

## 4. Detail: Query Is Also Compressed, but Inference Caches Only c_kv

In real MLA the Query is compressed too: first `c_q = x @ W_DQ` (d_c dim), then `Q = c_q @ W_UQ`. During training, compressing Q cuts **activation memory** (Q must be kept for forward/backward). At inference, Query is recomputed from the current input x and never enters the KV cache — so the only thing that must persist in the cache is the single latent c_kv.

That is also why MLA wins on both "training saves memory" and "inference saves KV": it compresses the same latent c, just for different purposes.

## 5. Ties to the Rest of the Series

- mm5 places MLA in the "attention five-piece set" conceptually and draws DeepSeek-V4's "MLA → NSA → DSA → CSA+HCA" four-stage evolution map — MLA is the starting point.
- fa4 (DeepSeek V4)'s CSA/HCA push further along this "compress KV" line: compressing KV down to 4-token blocks, even 128× heavy compression.

Once you internalize MLA's operator, V4's compressed attention reads as "the same latent-vector idea, taken further" — not a pile of alien terms.

> Next: **GQA / MQA operator comparison** (they compress "heads" not "width") and **parameter conservation from MHA to MLA** — follow the series for updates.
