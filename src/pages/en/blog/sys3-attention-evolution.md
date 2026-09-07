---
title: 'Inference Systems Infrastructure Notes (3): MSA / CSA / HCA — Three Attention Redesign Routes in One Picture'
description: 'Frontier long-context models in 2026 reshape attention along three main routes: MiniMax sparse attention (MSA), DeepSeek V4 compressed sparse attention (CSA), and heavily compressed attention (HCA). Where each one intervenes, what compression ratio it achieves, and which scenario it fits — explained with one comparison diagram and three schematics.'
pubDate: 2026-08-26
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys3-attention-evolution
layout: ../../../layouts/BlogPost.astro
---

## 0. Why Distinguish the Three?

Nearly every frontier model in 2026 is reshaping attention, but they cut in different places:

- **MSA (Sparse Attention)** — does not compress KV, it only changes **what you look at**; sparse selection.
- **CSA (Compressed Sparse Attention)** — compresses KV into blocks first, then applies sparse selection.
- **HCA (Heavily Compressed Attention)** — compresses KV extremely hard, then runs dense attention.

The three are not replacements for one another; they are **layered collaborators**. DeepSeek V4 even stacks all three (sliding window + CSA + HCA). Only by understanding where each one stops paying off can you see why V4 keeps KV down to ~10% at 1M tokens.

## 1. One-Line Positioning

| Mechanism | In one line | Representative model | Core action | Complexity |
|---|---|---|---|---|
| **MSA** | Compute only the important attention pairs and skip the rest | MiniMax M3 | Learned or fixed sparse pattern | ≪ O(n²) |
| **CSA** | Compress KV into blocks first, then take top-k over blocks | DeepSeek V4-Pro | 4-token block compression + FP4 indexer | O(n·c) |
| **HCA** | Compress 128 tokens into one entry, then run dense attention | DeepSeek V4 upper layers | 128× compression + differentiable fusion | O(n·c') |

> Notation: n is sequence length; c is the number of compressed entries (c ≈ n/4 for CSA, c ≈ n/128 for HCA).

## 2. Principle Comparison

<div class="fig">
<svg viewBox="0 0 680 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Comparison of MSA, CSA and HCA attention redesigns">
  <defs>
    <marker id="arrEn3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Three attention redesigns: where they cut and how hard they compress</text>

  <rect x="20" y="50" width="300" height="110" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.2"/>
  <text x="35" y="75" font-size="14" font-weight="700" fill="#166534">MSA - Sparse Attention</text>
  <text x="35" y="95" font-size="12" fill="#166534">MiniMax M3</text>
  <text x="35" y="115" font-size="12" fill="#444">KV is not compressed; only important positions are attended</text>
  <text x="35" y="135" font-size="12" fill="#444">Sparse pattern learned or fixed; skips 90%+ of compute</text>

  <rect x="360" y="50" width="300" height="110" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.2"/>
  <text x="375" y="75" font-size="14" font-weight="700" fill="#1e40af">CSA - Compressed Sparse Attention</text>
  <text x="375" y="95" font-size="12" fill="#1e40af">DeepSeek V4</text>
  <text x="375" y="115" font-size="12" fill="#444">Compress every 4 tokens into 1 KV block</text>
  <text x="375" y="135" font-size="12" fill="#444">FP4 indexer picks top-k blocks; local window as backstop</text>

  <rect x="190" y="190" width="300" height="110" rx="8" fill="#fff7ed" stroke="#f59e0b" stroke-width="1.2"/>
  <text x="205" y="215" font-size="14" font-weight="700" fill="#92400e">HCA - Heavily Compressed Attention</text>
  <text x="205" y="235" font-size="12" fill="#92400e">DeepSeek V4 upper layers</text>
  <text x="205" y="255" font-size="12" fill="#444">Compress every 128 tokens into 1 entry</text>
  <text x="205" y="275" font-size="12" fill="#444">Then run dense attention over all compressed entries</text>

  <line x1="170" y1="160" x2="260" y2="190" stroke="#6b7280" stroke-dasharray="4" marker-end="url(#arrEn3)"/>
  <text x="190" y="182" font-size="11" fill="#6b7280">add compression</text>
  <line x1="510" y1="160" x2="420" y2="190" stroke="#6b7280" stroke-dasharray="4" marker-end="url(#arrEn3)"/>
  <text x="424" y="182" font-size="11" fill="#6b7280">ratio up to 128x</text>

  <rect x="20" y="330" width="640" height="70" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="355" font-size="12.5" fill="#1a1a1a"><tspan font-weight="700">How they cooperate:</tspan> the nearest 128 tokens go through a sliding window (uncompressed) + mid-range CSA (4x compression + top-k) + far-range HCA (128x compression).</text>
  <text x="35" y="380" font-size="12" fill="#6b7280">MSA saves compute, CSA saves KV while keeping selection, HCA compresses to the extreme - only the combination carries 1M tokens.</text>
</svg>
<p class="cap">Figure: Where MSA / CSA / HCA intervene, and their compression ratios.</p>
</div>

## 3. MSA: Attend Only to What Matters

**Core idea.** The O(n²) of standard attention comes from "every query looks at every key." MSA makes each query look at only a small subset of keys, using a learned or fixed pattern.

```text
Full Attention:  Q_n x K_1...K_n      -> n^2 dot products
MSA:             Q_n x K_selected     -> only the selected positions
```

MiniMax M3 introduces a **learnable sparse gate** inside attention: the model itself decides which historical positions are worth attending to. Because KV is not compressed, the implementation is relatively direct — but the sparse index carries its own overhead.

**Pros.** No loss of original KV precision; fine-grained long-range information is preserved (as long as it gets selected).
**Cons.** The sparse index is irregular, so GPU memory access is non-contiguous; index overhead grows on very long sequences.

## 4. CSA: Compress First, Then Select Sparsely

**Core idea.** Compress KV in 4-token blocks into one "block vector," score all blocks with a lightweight FP4 indexer, pick the top-k blocks for exact attention, and finally keep local detail with a 128-token sliding window.

```text
Raw KV:   [t1][t2][t3][t4] [t5][t6][t7][t8] ...  -> length n
Compressed: [c1]            [c2]            ...  -> length n/4
FP4 index:  Q dotted with every c -> top-k blocks
Exact step: Q attends to top-k blocks + the most recent 128 tokens
```

**Key points:**

- Compression ratio m = 4, not extreme, so information loss stays controllable;
- The FP4 indexer is extremely fast; top-k selection is a tiny fraction of total time;
- The sliding-window branch preserves local precision, so short-range detail is not crushed.

**Pros.** KV cache drops to 1/4 directly, and compute also falls sharply after top-k; regular block compression is GPU-friendly.
**Cons.** You must train the compression function and the indexer; a poorly designed compression function loses long-range semantics.

## 5. HCA: Compress to the Extreme, Then Go Dense

**Core idea.** Fuse 128-token chunks into a single KV entry (compression ratio 128:1), then run **dense attention** over all compressed entries.

```text
Raw KV:   128 tokens  ->  1 directory entry
1M tokens ->  ~8000 directory entries
Q runs softmax attention over all 8000 entries
```

Why go **dense** rather than sparse after compression? At a scale of 8000 entries, a dense kernel has contiguous memory access and high warp utilization, which in practice beats irregular sparsity. HCA gives the model a coarse **outline of distant history**; CSA or the sliding window fills in the specifics.

**Pros.** KV compressed to the limit (1/128), so even 1M tokens fit in memory; dense attention is stable and efficient.
**Cons.** Per-token information is heavily abstracted, so it is a poor fit for tasks that depend on precise long-range detail; it must be paired with CSA or a sliding window.

## 6. All Three Together: V4's Layered Attention

<div class="fig">
<svg viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="DeepSeek V4 layered attention: sliding window plus CSA plus HCA">
  <rect x="0" y="0" width="680" height="220" fill="none"/>
  <text x="20" y="28" font-size="14" font-weight="700" fill="#1a1a1a">DeepSeek V4: near / mid / far attention working together</text>

  <rect x="20" y="55" width="180" height="60" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="110" y="82" font-size="13" fill="#047857" text-anchor="middle">Nearest 128 tokens</text>
  <text x="110" y="102" font-size="11" fill="#047857" text-anchor="middle">Standard window, uncompressed</text>

  <rect x="220" y="55" width="200" height="60" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="320" y="82" font-size="13" fill="#1d4ed8" text-anchor="middle">Mid-range history</text>
  <text x="320" y="102" font-size="11" fill="#1d4ed8" text-anchor="middle">CSA, 4x compression + top-k</text>

  <rect x="440" y="55" width="220" height="60" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="550" y="82" font-size="13" fill="#b45309" text-anchor="middle">Far history</text>
  <text x="550" y="102" font-size="11" fill="#b45309" text-anchor="middle">HCA, 128x compression + dense</text>

  <line x1="200" y1="85" x2="218" y2="85" stroke="#6b7280"/>
  <line x1="420" y1="85" x2="438" y2="85" stroke="#6b7280"/>

  <rect x="20" y="140" width="640" height="60" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="165" font-size="12.5" fill="#1a1a1a">Queries route to different branches by distance; the final output is summed or concatenated.</text>
  <text x="35" y="185" font-size="12" fill="#6b7280">Result: at 1M tokens, KV cache is about 10% of V3.2 and per-token compute about 27%.</text>
</svg>
<p class="cap">Figure: V4 stacks window + CSA + HCA so near and far are handled by different branches.</p>
</div>

## 7. Quick Selection Table

| Scenario | Recommended mechanism | Why |
|---|---|---|
| Precise local modeling | Sliding window / full attention | No compression, detail intact |
| Mid-range selective long context | CSA | Compression plus selection balances precision and efficiency |
| Far-range overview / coarse memory | HCA | Compressed to the limit, dense compute is fast |
| Building your own lightweight long-context model | Start with window + CSA, add HCA if needed | CSA is easier to tune than HCA |
| Agent or video history needing 1M+ tokens | Three-layer stack (the V4 approach) | Every single mechanism has a gap; only the stack covers them all |

## 8. Common Pitfalls

1. **Do not swap HCA for sparse.** At 8000 entries the dense kernel is more regular; sparse index overhead overtakes it.
2. **The CSA compression function must be differentiable.** Plain mean-pooling loses semantics; use gated or learned compression.
3. **Clamp the FP4 indexer.** Its numeric range is small, so unclamped dot products overflow easily.
4. **HCA must align its range with CSA.** If the HCA compression range does not match the CSA selectable range, you get holes: something in the directory that you cannot index, or vice versa.

## 9. Connection to the Investment Thread

All three attention redesigns point to the same conclusion: **the cost of long context is falling fast.** For embodied intelligence, that means a robot VLA can "remember" longer video history and plan more complex tasks without being bounded by onboard memory. Related names: edge inference chips (Horizon Robotics, 09660.HK), the Huatai-PineBridge robotics ETF (562500), and liquid cooling / compute infrastructure (TrendForce penetration 53% to 60%).

> Investment content is the author's personal observation and is not investment advice.
