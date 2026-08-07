---
title: 'Frontier Architecture Decoding Notes (3): MiniMax M2.7 → M3 — The Leap of Sparse Attention'
description: 'Following the roadmap, the second model torn apart is MiniMax. First the M2.7 baseline (standard attention, 200K, text-only, API-only), then how M3 uses MiniMax Sparse Attention (MSA) to pull context from 200K to 1M while cutting per-token compute at 1M to ~1/20 of M2.7 (prefill ~9×, decode ~15× faster). M3 also adds native multimodal (text+image+video) and computer use, and open-weights.'
pubDate: 2026-08-06
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa3-minimax-m3
layout: ../../../layouts/BlogPost.astro
---

## 0. Why Watch MiniMax's "Evolution"

Kimi K3 "adds a delta on MLA" — a gentle change. MiniMax gives a more直观 lesson: **how one series goes from standard attention to sparse, step by step**. See the M2.7 starting point clearly, and the M3 leap becomes obvious.

## 1. Starting Point: MiniMax M2.7 (baseline)

- **standard softmax attention**, no sparsity;
- **200K context**, text-only, no native multimodality;
- **API-only**, weights not open;
- mostly text training data.

It represents the "previous generation" dense / standard-attention model: capable enough, but both context and compute hit a ceiling.

### 1.1 Backstory: From MiniMax-Text-01's Lightning Attention to M2.7's "Revert to Full Attention"

M2.7 is not MiniMax's first brush with "non-standard attention." Earlier, **MiniMax-Text-01** used a hybrid **Lightning Attention** (a linear / retentive-attention variant, interleaved with standard softmax layers) to push context all the way to 4M. Linear attention's upside is that compute grows **linearly** with length, not quadratically; the cost is limited expressiveness and training stability, and it underperforms full attention on long-range exact retrieval and complex reasoning.

MiniMax then made a key pivot in later iterations: revert to **standard softmax full attention** as the safe baseline (the 200K standard attention of M2.7), saving the "ultra-long context" bet for a more controllable scheme; then attack 1M with **sparse attention (MSA)** rather than linear attention in M3. In other words, M2.7's "revert to full attention" is not a retreat — it moves "long context" off a risky route (linear attention) onto a more controllable one (full-attention base, then sparsify), clearing the ground for M3's MSA.

> This "linear-attention exploration → revert to full attention → sparsify" curve, alongside Kimi K3 (delta on MLA) and DeepSeek V4 (compress KV directly), forms the three different kinds of courage in 2026's "reshape attention."

## 2. The Leap: MiniMax M3 + MSA

M3's core innovation is **MiniMax Sparse Attention (MSA)**:

- **context 200K → 1M** (5×);
- at 1M context, **per-token compute is ~1/20 of M2.7**;
- **prefill ~9×, decode ~15× faster**;
- trained on **100T tokens**, **interleaved data**.

The intuition of sparse attention: not every query must attend to all historical keys. MSA uses a sparse-selection mechanism so each position only attends to the "truly relevant" fraction of history, holding compute down at 1M length.

<div class="keybox">
<strong>Key:</strong> sparse attention's payoff comes from "computing less" — at 1M length M3's per-token compute is only <strong>1/20</strong> of M2.7's, buying order-of-magnitude prefill/decode speedups. This is the canonical "context grows but cost doesn't explode" solution.
</div>

### 2.1 Engineering Side: KV-Block-Major Prefill — the KV Layout that Carries MSA

MSA lets each query pick only a few KV blocks, turning attention from "fully connected" into "block selection." That raises new requirements on the inference engine's KV-cache layout.

Standard paged-attention KV is **token-major**: page → token → head → dim. When attention is block-selective (each query picks a different set of KV blocks), token-major makes "gathering the blocks a query selected" a scattered cross-page gather — non-contiguous memory access during prefill, hard-to-write kernels.

**KV-Block-Major** flips the layout to **block → token → head → dim**: tokens within the same block are contiguous in memory. Then each query's selected blocks can be read out as whole contiguous chunks, gather collapses into a few large block copies, and the prefill kernel both coalesces memory access and parallelizes per block. It is exactly this class of layout support that lets block-sparse-attention models like MSA be served at 1M context by engines like vLLM without rewriting the whole attention backend (one of the keys to so-called "Day-0 support").

> One line: MSA changes *which* to compute; KV-Block-Major changes *how to store* so the algorithm hits peak throughput — algorithm and systems engineering are two sides of the same coin.

### 2.2 Where Sparse Attention Comes From: BigBird / Longformer to MSA

MSA did not appear from nowhere. The idea of block-sparse attention traces at least to several foundational works around 2020:

- **Longformer** (2020): combined "sliding window + global tokens + task-specific sparsity" to bring attention from O(n²) down to O(n), proving you can work well without every token seeing all history.
- **BigBird** (2020): used three sparse pattern types — "random blocks + window blocks + global blocks" — still linear complexity yet theoretically Turing-complete, the canonical block-sparse attention work.
- Later **Linear Attention / Retentive Network / Gated Linear Attention** took a different route: kernel tricks or recurrence to write attention as linear, at the cost of some expressiveness (see the Lightning Attention recap in 1.1).

MSA stands on these shoulders but makes two key upgrades: **(1) the selection is learned**, not a fixed window / random pattern — the model itself decides which history blocks each block should see; **(2) it is deeply coupled with native multimodal and long-context training**, not a sparse trick bolted on afterward. So it is neither Longformer's fixed window nor BigBird's random blocks, but a modern version of "learnable block-sparse + end-to-end long-context training."

## 3. Beyond Attention: Native Multimodal + Computer Use

M3 doesn't only change attention; it fills in "usability":

- **native multimodal**: text + image + video modeled uniformly, not a text model with vision bolted on;
- **computer use**: can operate GUIs / browsers, fitting Agent scenarios;
- **open weights**: vs M2.7's "API-only," M3 opens weights, helping ecosystem and secondary development.

## 4. Benchmarks (official)

| Benchmark | M3 score |
|---|---|
| SWE-Bench Pro | 59.0% |
| Terminal-Bench 2.1 | 66.0% |
| MCP Atlas | 74.2% |

These agentic / tool-call / terminal-operation benchmarks map exactly to M3's computer-use and long-context positioning.

## 5. Contrast With Kimi K3

- Kimi K3 "**changes attention internals**" (KDA delta + gate), keeping the MLA base;
- MiniMax M3 "**changes attention's computation scope**" (sparsify), jumping straight from standard attention to MSA;
- common: both aim to **hold 1M context without exploding compute**; differ in where they cut.

> Next: DeepSeek V4, more aggressive than sparsity — compressing KV itself.
