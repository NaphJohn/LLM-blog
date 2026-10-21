---
title: "Operator Explainers (3): Ascend EagleQBSA — Quantized × Sparse Joint Attention, up to 7.6× Faster Attention Modules"
description: "Long-video generation pushes attention to million-token sequences. EagleQBSA in MindIE SD on Ascend 950 lands a distribution-aware hybrid quantization recipe (per-block INT8 QK + per-channel FP8 V/P) on top of AdaBSA block sparsity, then squeezes the Vector side with Cube-to-Vector direct transfer with transpose, columnized Softmax reduction, and bitwise FP8 conversion. Wan2.2 end-to-end 1.6×, MiniMax-H3 1.52×, single-operator up to 14.7×, VBench error under 1%."
pubDate: 2026-10-21
series: Operator Explainers
lang: en
altLang: zh
altHref: /blog/op3-eagleqbsa
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

Quantization and sparsity are two orthogonal routes to faster attention, but **stacking them naively makes them fight each other**: sparse selection reshapes the data distribution and amplifies quantization error, and once low-bit matmuls speed up the Cube, the Vector side (Softmax and friends) becomes the new bottleneck. The answer from Ascend MindIE SD is EagleQBSA (Eagle Quant Block Sparse Attention): a **distribution-aware hybrid-precision recipe (per-block INT8 for QK, per-channel FP8 for V/P) plus Ascend 950 hardware-affinity work (transpose during Cube-to-Vector transfer, columnized Softmax reduction, bitwise FP8 conversion)**. Result: up to 7.6× faster attention modules, 1.6× end-to-end on Wan2.2, 1.52× on MiniMax-H3, with VBench error under 1%.

## 1. Background: Video Generation Pushes Attention into a Corner

DiT-style video models (Wan2.2, MiniMax-H3) run full attention over **joint spatio-temporal token sequences**: a 5s, 720p-class video easily reaches **hundreds of thousands to millions of tokens** after patchification. Attention then costs twice:

- **Compute**: the $O(N^2)$ QK^T and PV matmuls eat the bulk of FLOPs at long sequence lengths;
- **Memory traffic**: K/V reads grow quadratically with sequence length, so HBM bandwidth saturates before compute does.

Two optimizations become mandatory — **sparse attention** (skip unimportant KV connections) and **quantized attention** (compress QKV into FP8/INT8 and friends). This post dissects EagleQBSA, the operator-level combination of the two: built on the CANN BlockSparseAttention operator, targeting Ascend 950-series hardware, shipped with MindIE SD.

## 2. Two Orthogonal Acceleration Routes

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Sparse attention vs quantized attention">
  <defs>
    <marker id="arr3en1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Sparse vs quantized: two cuts, along different dimensions of attention</text>

  <!-- left: sparsity -->
  <text x="20" y="58" font-size="13" font-weight="700" fill="#047857">Sparse attention: cut the connection count</text>
  <g>
    <rect x="30" y="70" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="64" y="70" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="98" y="70" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="132" y="70" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="166" y="70" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="200" y="70" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>

    <rect x="30" y="100" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="64" y="100" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="98" y="100" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="132" y="100" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="166" y="100" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="200" y="100" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>

    <rect x="30" y="130" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="64" y="130" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="98" y="130" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="132" y="130" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="166" y="130" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="200" y="130" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>

    <rect x="30" y="160" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="64" y="160" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="98" y="160" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="132" y="160" width="34" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
    <rect x="166" y="160" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
    <rect x="200" y="160" width="34" height="30" fill="#f3f4f6" stroke="#d1d5db"/>
  </g>
  <text x="30" y="215" font-size="12" fill="#047857">Computed: only important blocks (attention mass concentrates on few links)</text>
  <text x="30" y="235" font-size="12" fill="#6b7280">Skipped: no KV access at all, saving compute and bandwidth</text>
  <text x="30" y="262" font-size="11" fill="#6b7280">Block granularity: MindIE SD already ships adaptive block-sparse AdaBSA</text>

  <!-- divider -->
  <line x1="345" y1="50" x2="345" y2="280" stroke="#e5e7eb" stroke-width="1.5"/>

  <!-- right: quantization -->
  <text x="375" y="58" font-size="13" font-weight="700" fill="#1d4ed8">Quantized attention: cut bits per number</text>

  <text x="375" y="92" font-size="12" fill="#1a1a1a">FP16 / BF16</text>
  <rect x="490" y="80" width="160" height="16" rx="3" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="600" y="93" font-size="10" fill="#1e40af" text-anchor="middle">16 bit</text>

  <text x="375" y="122" font-size="12" fill="#1a1a1a">FP8 (E4M3)</text>
  <rect x="490" y="110" width="80" height="16" rx="3" fill="#bfdbfe" stroke="#3b82f6"/>
  <text x="530" y="123" font-size="10" fill="#1e40af" text-anchor="middle">8 bit</text>

  <text x="375" y="152" font-size="12" fill="#1a1a1a">INT8</text>
  <rect x="490" y="140" width="80" height="16" rx="3" fill="#93c5fd" stroke="#3b82f6"/>
  <text x="530" y="153" font-size="10" fill="#1e40af" text-anchor="middle">8 bit</text>

  <text x="375" y="182" font-size="12" fill="#1a1a1a">FP4 / INT4</text>
  <rect x="490" y="170" width="40" height="16" rx="3" fill="#60a5fa" stroke="#3b82f6"/>
  <text x="510" y="183" font-size="10" fill="#1e40af" text-anchor="middle">4 bit</text>

  <text x="375" y="215" font-size="12" fill="#1a1a1a">Upside: low-bit matmul throughput</text>
  <text x="375" y="235" font-size="12" fill="#1a1a1a">Lower memory footprint · lower bandwidth pressure</text>
  <text x="375" y="262" font-size="11" fill="#6b7280">Cost: quantization error, strongly distribution-dependent</text>
</svg>
<p class="cap">Figure 1: the orthogonality of sparsity and quantization — one cuts which connections are computed, the other cuts how many bits each connection uses. The two pipelines are independent, so gains stack in theory.</p>
</div>

- The model-side version of the sparsity route already appeared in fa3 (MiniMax M3 with MSA); EagleQBSA uses the **inference-side, operator-level** adaptive block-sparse AdaBSA. Same idea, different landing spot.
- The full quantization spectrum appeared in op2 of this series (FP8/INT4/AWQ·GPTQ/FP4 QAT/KV Cache quantization). op2 answers "which precision tiers exist"; this post answers "which tier each matrix should use in a real operator, and why".

## 3. Why Naive Stacking Backfires

The sparsity and quantization pipelines are naturally orthogonal, but combining them directly hits two snags:

**Algorithmic layer: sparse selection reshapes the distribution and amplifies quantization error.**
The attention output P = softmax(QK^T) is **per-block dynamic**: the sparse mask differs block by block, so the score distribution of retained blocks is not the same before and after masking. If scales are computed from the pre-sparsity distribution, quantization error gets amplified by the nonlinearity of softmax, directly degrading visual generation quality (broken frames, drifting subjects).

**Implementation layer: after low-bit matmul speeds up the Cube, the Vector side becomes the new bottleneck.**
Once the QK/PV matmuls move to low-precision Cube paths, the **Vector-unit** work — Softmax (max, sum, exp, normalize), quantizing P, dequantization — grows as a share of runtime. The speedup no longer depends on matmul throughput but on whether the Vector pipeline can keep up.

EagleQBSA has two design pillars answering exactly these two snags: **distribution-aware hybrid-precision quantization** (algorithmic) and **Ascend 950 hardware-affinity implementation** (implementation).

## 4. The Distribution-Aware Recipe: INT8 QK / FP8 PV

Measured analysis of whole-network QKV matrices in video-generation workloads revealed three facts that drive the format choices:

**Fact one: QK is concentrated within each block (spatio-temporal locality).**
Although QK distributions differ widely across the whole sequence, within a single block the local sequence distribution is relatively concentrated — so QK suits **per-block quantization along the sequence dimension** (per-block scale).

**Fact two: INT8 beats FP8 for QK by an order of magnitude.**
Running whole-network QK attention with FP8 vs INT8 quantization against a BF16 baseline: **QK FP8 quantization shows 3–4× the single-operator error of INT8**. The fine-grained per-block INT8 scale exactly compensates for FP8's wide dynamic range but uneven within-block mantissa usage.

**Fact three: V is scattered with a wide dynamic range beyond INT8's fit; P follows V.**
The V matrix has a scattered distribution whose dynamic range exceeds what INT8 adapts to, so it suits **FP8 per-channel quantization** (per-channel scale, cheap dequantization); the P matrix follows V into FP8, and the PV matmul runs in FP8. Measured: **P/V INT8 quantization increases single-operator error by 2–3 orders of magnitude versus P/V FP8**.

The final recipe in one line: **hybrid-precision quantization with INT8 QK (per-block) / FP8 PV (V per-channel, P following)**, stacked on AdaBSA block sparsity. A code skeleton of the math:

```python
# QK: per-block INT8 along the sequence dimension (block-internal concentration, spatio-temporal locality)
q_int8, q_scale = quant_per_block_int8(q, block=128)   # scale: [num_blocks]
k_int8, k_scale = quant_per_block_int8(k, block=128)
scores = (q_int8 @ k_int8.T) * (q_scale * k_scale)     # INT8 matmul + dequant scale

# AdaBSA block-sparse mask: keep only important blocks, skip the rest of the KV
scores = apply_block_sparse_mask(scores, sparsity=0.63)

# Softmax (Vector side, columnized reduction on Ascend 950, see section 5)
p = softmax(scores)

# Quantize P to FP8 (bitwise approximate conversion, see section 5); V per-channel FP8
p_fp8 = quant_fp8_bitwise(p)
v_fp8, v_scale = quant_per_channel_fp8(v)              # scale: [channels]
out = (p_fp8.to_fp16() @ v_fp8.to_fp16()) * v_scale    # FP8 PV matmul + dequant
```

<div class="fig">
<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="EagleQBSA pipeline: INT8 QK, block-sparse Softmax, FP8 P, FP8 PV">
  <defs>
    <marker id="arr3en2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">EagleQBSA pipeline: blue = Cube low-precision matmul, amber = Vector side</text>

  <!-- inputs -->
  <rect x="20" y="60" width="70" height="40" rx="6" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="55" y="84" font-size="13" fill="#1a1a1a" text-anchor="middle">Q / K / V</text>
  <text x="55" y="116" font-size="10" fill="#6b7280" text-anchor="middle">BF16 in</text>

  <!-- quant boxes -->
  <rect x="120" y="60" width="110" height="40" rx="6" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="175" y="78" font-size="12" fill="#1e40af" text-anchor="middle">QK per-block INT8</text>
  <text x="175" y="95" font-size="10" fill="#1e40af" text-anchor="middle">block-local concentration</text>

  <rect x="120" y="150" width="110" height="40" rx="6" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="175" y="168" font-size="12" fill="#1e40af" text-anchor="middle">V per-channel FP8</text>
  <text x="175" y="185" font-size="10" fill="#1e40af" text-anchor="middle">scattered, wide range</text>

  <!-- QK matmul -->
  <rect x="265" y="60" width="110" height="40" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
  <text x="320" y="78" font-size="12" fill="#1e40af" text-anchor="middle">INT8 QK^T</text>
  <text x="320" y="95" font-size="11" fill="#1e40af" text-anchor="middle">Cube matmul</text>

  <!-- dequant + sparse -->
  <rect x="265" y="115" width="110" height="40" rx="6" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="320" y="133" font-size="12" fill="#92400e" text-anchor="middle">dequant × AdaBSA</text>
  <text x="320" y="150" font-size="10" fill="#92400e" text-anchor="middle">block-sparse mask</text>

  <!-- Softmax -->
  <rect x="410" y="115" width="110" height="40" rx="6" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="465" y="133" font-size="12" fill="#92400e" text-anchor="middle">Softmax</text>
  <text x="465" y="150" font-size="10" fill="#92400e" text-anchor="middle">columnized reduction</text>

  <!-- P FP8 -->
  <rect x="410" y="180" width="110" height="40" rx="6" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="465" y="198" font-size="12" fill="#92400e" text-anchor="middle">P to FP8</text>
  <text x="465" y="215" font-size="10" fill="#92400e" text-anchor="middle">bitwise conversion</text>

  <!-- PV matmul -->
  <rect x="555" y="150" width="105" height="40" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
  <text x="607" y="168" font-size="12" fill="#1e40af" text-anchor="middle">FP8 PV</text>
  <text x="607" y="185" font-size="11" fill="#1e40af" text-anchor="middle">Cube matmul</text>

  <!-- output -->
  <rect x="555" y="225" width="105" height="40" rx="6" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="607" y="249" font-size="13" fill="#1a1a1a" text-anchor="middle">Output O</text>

  <!-- arrows -->
  <line x1="90" y1="80" x2="118" y2="80" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr3en2)"/>
  <line x1="90" y1="170" x2="118" y2="170" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr3en2)"/>
  <line x1="230" y1="80" x2="263" y2="80" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr3en2)"/>
  <line x1="320" y1="100" x2="320" y2="113" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr3en2)"/>
  <line x1="375" y1="135" x2="408" y2="135" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr3en2)"/>
  <line x1="465" y1="155" x2="465" y2="178" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr3en2)"/>
  <line x1="520" y1="200" x2="553" y2="185" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr3en2)"/>
  <line x1="175" y1="190" x2="175" y2="225" stroke="#4b5563" stroke-width="1.5"/>
  <line x1="175" y1="225" x2="607" y2="225" stroke="#4b5563" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="607" y1="225" x2="607" y2="192" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr3en2)"/>

  <text x="20" y="290" font-size="11" fill="#6b7280">Key: both matmul stages run on low-precision Cube paths; the Vector side keeps only Softmax and P quantization, both specially optimized (see figure 3).</text>
</svg>
<p class="cap">Figure 2: the EagleQBSA pipeline. The INT8 QK / FP8 PV hybrid is not a guess — it follows directly from the three measured facts in section 4.</p>
</div>

## 5. Ascend 950 Hardware Affinity: Squeezing the Vector Side

Once quantization moves matmuls to the Cube, Vector-unit memory traffic and compute become the new bottleneck. Ascend 950-series provides two key hardware capabilities: **direct data transfer between Cube and Vector** (no round-trip through memory) and **register-level (Regbase) Vector programmability**. All three EagleQBSA optimizations build on these:

**Optimization one: transpose during transfer — columnize the Softmax reduction.**
The operator transposes the matmul result while transferring it from Cube to Vector. Softmax needs a reduction over **one row** (max, then sum) — one row at a time, low data parallelism. After transposition, the row reduction becomes a **pairwise reduction across many columns** (tree-style halving), maximizing Vector parallelism.

<div class="fig">
<svg viewBox="0 0 680 290" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Softmax reduction optimization: row reduction to pairwise column reduction">
  <defs>
    <marker id="arr3en3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Softmax reduction optimization: transpose during Cube-to-Vector transfer turns row reduction into pairwise column reduction</text>

  <!-- left: before -->
  <text x="20" y="56" font-size="13" font-weight="700" fill="#92400e">Before: row-wise reduction (low parallelism)</text>
  <rect x="30" y="70" width="200" height="26" rx="3" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="130" y="87" font-size="11" fill="#92400e" text-anchor="middle">row i: max / sum sequentially over one row</text>
  <rect x="30" y="100" width="200" height="26" rx="3" fill="#f9fafb" stroke="#e5e7eb"/>
  <rect x="30" y="130" width="200" height="26" rx="3" fill="#f9fafb" stroke="#e5e7eb"/>
  <rect x="30" y="160" width="200" height="26" rx="3" fill="#f9fafb" stroke="#e5e7eb"/>
  <line x1="130" y1="96" x2="130" y2="126" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3 3"/>
  <text x="30" y="205" font-size="11" fill="#6b7280">One Vector reduction consumes a single row,</text>
  <text x="30" y="222" font-size="11" fill="#6b7280">so lane utilization stays low</text>

  <!-- middle: transpose -->
  <text x="270" y="56" font-size="13" font-weight="700" fill="#1a1a1a">Transpose in flight</text>
  <rect x="285" y="80" width="60" height="90" rx="6" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="315" y="120" font-size="11" fill="#1e40af" text-anchor="middle">S^T</text>
  <text x="315" y="138" font-size="10" fill="#1e40af" text-anchor="middle">direct</text>
  <line x1="230" y1="128" x2="283" y2="125" stroke="#4b5563" stroke-width="1.5" marker-end="url(#arr3en3)"/>

  <!-- right: after -->
  <text x="390" y="56" font-size="13" font-weight="700" fill="#047857">After: pairwise column reduction (tree halving)</text>
  <rect x="400" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <rect x="426" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <rect x="452" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <rect x="478" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <rect x="504" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <rect x="530" y="70" width="22" height="100" rx="3" fill="#d1fae5" stroke="#10b981"/>
  <text x="556" y="120" font-size="11" fill="#047857">col 0..N</text>
  <line x1="411" y1="70" x2="411" y2="52" stroke="#10b981" stroke-width="1.5"/>
  <line x1="437" y1="70" x2="437" y2="52" stroke="#10b981" stroke-width="1.5"/>
  <line x1="424" y1="52" x2="424" y2="40" stroke="#10b981" stroke-width="1.5"/>
  <line x1="463" y1="70" x2="463" y2="52" stroke="#10b981" stroke-width="1.5"/>
  <line x1="489" y1="70" x2="489" y2="52" stroke="#10b981" stroke-width="1.5"/>
  <line x1="476" y1="52" x2="476" y2="40" stroke="#10b981" stroke-width="1.5"/>
  <line x1="424" y1="40" x2="476" y2="40" stroke="#10b981" stroke-width="1.5"/>
  <text x="600" y="45" font-size="11" fill="#047857">pair</text>
  <text x="600" y="60" font-size="11" fill="#047857">wise</text>
  <text x="400" y="195" font-size="11" fill="#6b7280">Many columns reduce max/sum pairwise in the same cycle,</text>
  <text x="400" y="212" font-size="11" fill="#6b7280">finishing in log2(N) steps with Vector lanes saturated</text>

  <text x="20" y="262" font-size="11" fill="#6b7280">Note: Softmax = max reduction → exp → sum reduction → normalize; columnization benefits both reductions.</text>
</svg>
<p class="cap">Figure 3: columnized Softmax reduction. The transpose happens "for free" during transfer, buying a step change in Vector reduction parallelism.</p>
</div>

**Optimization two: quantize P with a bitwise approximate FP16-to-FP8 conversion.**
Compared with the standard numeric FP16-to-FP8 conversion (exponent bias + mantissa truncation + overflow handling), EagleQBSA implements an approximate conversion with **bitwise operations**, lowering overhead and further relieving the Vector compute bottleneck.

**Optimization three: Regbase register-level development.**
Together with Ascend 950 Regbase capability, the Vector-side kernel organizes data at register level, cutting intermediate memory round-trips.

## 6. Measured Performance and Accuracy

**Single-operator performance** (B=1, N=40, D=128, covering 8K–128K sequence lengths):

| Baseline | Configuration | Speedup |
|---|---|---|
| BF16 block-sparse BSA operator | same sparsity | **~1.6×** |
| non-sparse float attention FaScore | 64K sequence, 0.9 sparsity | **up to 14.7×** |

Read the second row carefully: **up to 14.7× versus dense float attention at 0.9 sparsity** — that is the stacked upside of "sparsity cuts connections × quantization cuts bits"; the 1.6× versus BF16 BSA at equal sparsity isolates the pure quantization contribution.

**End-to-end performance** (only the self-attention module swapped to EagleQBSA; linear layers unchanged):

| Model | Task | Sparsity | End-to-end speedup |
|---|---|---|---|
| Wan2.2-T2V-A14B | text-to-video | 0.63 | **1.64×** |
| Wan2.2-I2V-A14B | image-to-video | 0.63 | **1.61×** |
| MiniMax-H3 | text-to-audio-video | 0.8 | **1.52×** |

Attention modules are up to **7.6×** faster; 5s video generation lands at 1.5–1.6× end-to-end — attention's share of whole-network time caps the end-to-end gain, the typical "operator fast, network discounted" distribution.

**Generation quality** (VBench, five dimensions: subject consistency / imaging quality / aesthetic quality / overall consistency / motion smoothness):

- Both tasks show **overall accuracy error under 1%** versus the BF16 baseline;
- Text-to-video used 48 `t2v_sora` text prompts;
- Under identical prompts, EagleQBSA matches the baseline in subject and style;
- Official caveat: results reflect this test configuration only and do not imply zero accuracy loss for all content and models.

## 7. Wiring: Where This Sits on the Site Map

- **op2 (quantization deployment spectrum) → op3**: op2 maps the FP8/INT4/FP4 tiers; op3 demonstrates on a real operator that "tier choice is a data-distribution question" — QK is block-concentrated so INT8 per-block, V is scattered with wide range so FP8 per-channel, P follows V. **There is no globally optimal quantization, only per-matrix optimal.**
- **fa3 (MiniMax M3 sparse attention) ↔ op3**: the same sparsity idea — fa3 reshapes the architecture (sparse at training time), op3 reshapes the operator (training untouched, plug-and-play). The two routes fit very different scenarios.
- **fw9 (Wan2.2 dual-expert MoE) → op3**: fw9 dissects the Wan2.2 model structure and the vllm-omni engine; op3 shows the same model's attention acceleration path on Ascend. Systems optimization for video generation is becoming a battlefield of its own.
- **sys series (KV Cache / attention optimization)**: EagleQBSA handles only the **attention slice of compute and memory traffic**; KV cache compression and tiered caching are the orthogonal systems-layer line, two sides of the same coin with this post.

## 8. Summary

The EagleQBSA methodology compresses into three sentences:

1. **Recipes follow distributions**: INT8 per-block for QK plus FP8 per-channel for V/P — each matrix's format is decided by measured data distribution, not one size fits all;
2. **Bottlenecks move**: once the Cube goes low-bit, the Vector Softmax and quantization become the new bottleneck — optimization must chase the bottleneck;
3. **Hardware affinity is the amplifier**: transpose-on-transfer, columnized reduction, and bitwise FP8 conversion — these "non-algorithmic" details account for the gap between 1.6× and 7.6×.

For practitioners on domestic compute stacks, there is also a signal here: Ascend 950 plus MindIE SD has turned "quantized × sparse joint attention" into a ready-to-use operator feature (shipped with MindIE SD), moving domestic inference optimization for video generation from "it runs" to "chasing multiples".

---

**Series Navigation**

- Previous: **op2 Model Quantization for Deployment — the precision-compression spectrum from FP16 to FP4** (the map of quantization tiers; this post is "tier choice inside a real operator");
- Series opener: **op1 MLA Operator-Level** (another operator-perspective sample: compressing KV width);
- Related: **fa3 MiniMax M3 sparse attention** (model-side sparsity, read against the inference-side sparsity here), **fw9 Wan2.2 dual-expert MoE and vllm-omni** (the systems-side dissection of the same model), **sys9 the KV Cache Landscape** (the orthogonal systems-layer optimization line).

*Sources: this post distills the public Ascend community / MindIE SD article "Up to 7.6× faster: EagleQBSA quantized sparse attention optimization practice on Ascend" (CANN operator source: gitcode.com/cann/ops-transformer → attention/block_sparse_attention; open-source PR: gitcode.com/Ascend/MindIE-SD/pull/608). All performance and accuracy figures follow the test configurations disclosed in the original article and are not universal conclusions.*
