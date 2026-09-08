---
title: 'Inference Systems Infrastructure Notes (7): DCP — Decode Context Parallelism, sharding KV cache along the sequence dimension'
description: 'Memory and communication optimization for the long-context Decode phase: DCP shards redundantly replicated KV cache along the sequence dimension and reuses a Scratch Buffer for all_gather/reduce_scatter, synergizing with Prefix Caching, speculative decoding and P/D disaggregation.'
pubDate: 2026-09-08
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys7-dcp-decode-context-parallel
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

In long-context serving the Decode-phase bottleneck is not FLOPs but **redundant KV-cache replication across GPUs**. DCP (Decode Context Parallelism) shards the KV cache along the **sequence dimension** across GPUs and reuses a Scratch Buffer for collective comms — saving memory and cutting latency, with up to **1.5%~4%** end-to-end throughput gain. This post links it with HiCache (sys1), HiSparse (sys6) and P/D disaggregation from this series.

## 1. Why DCP: the KV wall in Decode

Under standard Tensor Parallelism (TP), each layer places its KV cache on every GPU that participates in that layer. The longer the sequence, the larger the KV — but TP splits along the **hidden dimension**, not the sequence dimension, so:

- Every GPU keeps a full copy of the KV cache → **massive redundant replication**, memory wasted for nothing.
- Single-GPU memory runs out first, Batch Size can't grow → throughput capped, long context OOMs outright.

Decode is token-by-token, compute-light but memory-bound, so "fits in memory, fetched fast" matters more than matmul throughput. DCP targets exactly this.

## 2. Core mechanism: KV sharded along the sequence dimension

DCP changes KV from "full copy per GPU" to "each GPU keeps only its slice":

```
TP split:  KV replicated across hidden dim → every GPU holds full-sequence KV (redundant)
DCP split: KV sharded across sequence dim   → every GPU holds only [start:end] (no redundancy)
```

- Each GPU holds only its KV slice; memory drops linearly with shard count.
- At Decode, GPUs process their KV slices in parallel, then a collective comms reassembles the full attention result.
- Orthogonal to TP: TP does hidden-split of weights/activations, DCP does sequence-split of KV; the two stack.

## 3. Communication optimization: reuse the Scratch Buffer

Sharding costs cross-GPU comms every Decode step. DCP's key engineering trick is **reusing a Scratch Buffer** for `all_gather` / `reduce_scatter`:

- No per-comm temporary allocation + data copy → drops frequent malloc/free and redundant copies.
- Lower comms latency → higher end-to-end throughput (framework measured up to **1.5%~4%**).
- Same philosophy as many vLLM operator optimizations: turn "allocate + copy" into "reuse one buffer".

## 4. Decode and prefill phases it touches

DCP changes more than Decode — it reaches preprocessing too:

- **Decode (core)**: per-token generation shards KV + collective comms reassemble attention, directly faster.
- **Chunked Prefill**: very long inputs are chunked and prefilled in parallel, lowering Time-To-First-Token (TTFT).
- **Cached Prefill**: reuse already-computed KV slices, avoiding re-prefilling the whole context.

## 5. Where it pays off, and what it synergizes with

DCP's value is sharpest when:

- **Long-context inference**: hundreds of thousands to millions of tokens (agents, long-doc analysis) make KV huge; sharding fits longer sequences into finite memory.
- **Higher throughput**: freed memory → larger Batch Size → higher overall throughput.
- **Advanced features**:
  - **Prefix Caching**: KV slices of a shared prefix are reusable; DCP stores, Prefix Caching hits.
  - **Speculative Decoding**: DCP's KV headroom lets the draft model be heavier and the batch larger, more speedup.
  - **P/D Disaggregation**: Prefill and Decode on separate instances; DCP keeps sharding KV on the Decode side — orthogonal and complementary.

## 6. Position among this series' optimizations

| Optimization | Solves | Dimension |
|--------------|--------|-----------|
| HiCache (sys1) | which NUMA node hosts KV, how NIC joins | HW topology / data link |
| HiSparse (sys6) | HBM as cache, host DRAM holds full KV, GPU keeps hot slots | tiered cache (capacity wall) |
| **DCP (sys7)** | KV sharded on sequence dim + Scratch Buffer comms | parallel sharding (redundancy / comms wall) |
| P/D disaggregation | Prefill and Decode instances decoupled | architecture split |

Three complementary angles: HiCache decides *where to put*, HiSparse decides *how much to keep*, DCP decides *how to shard in parallel*.

## 7. ⚠️ Disambiguation: three meanings of DCP

In high-performance inference frameworks (vLLM / SGLang), DCP almost always means **Decode Context Parallelism**. Elsewhere it can also mean:

- **Dynamic Compressing Prompts**: intelligently drop redundant tokens to compress the input prompt.
- **Dual-Cue Pruning**: for multimodal (vision-language) models, combines text + visual cues to pick important visual tokens.

Always confirm the context before writing/reading — don't confuse "parallel sharding" with "prompt compression" or "visual pruning".

## 8. Takeaway

DCP is the memory/comms optimization for long-context Decode: shard KV along the sequence dimension to kill redundant replication, reuse a Scratch Buffer to cut comms latency, and synergize natively with Prefix Caching, speculative decoding and P/D disaggregation. Together with HiSparse and HiCache it forms the three faces of long-context serving — capacity wall, data link, parallel sharding.

*Note: compiled from the user-provided DCP technical brief on 2026-09-08. The 1.5%~4% throughput is a framework-measured range; real gains vary with sequence length, batch size and comms topology.*
