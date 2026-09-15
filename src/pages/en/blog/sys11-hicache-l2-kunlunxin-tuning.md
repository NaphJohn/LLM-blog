---
title: "Tuning SGLang HiCache L2 on Kunlunxin P800/P900: Layout, DMA, NUMA and TP-Shared KV"
description: "Take SGLang HiCache L2 (Host DRAM prefix cache) from merely working to fast on Kunlunxin P800/P900: from the tiered architecture and net-benefit formula, down to layout choice (page_first_direct to layer_first), DMA segment count, async transfer, NUMA affinity, transparent huge pages, PCIe IDO ordering, write-policy semantic fix and Ghost List, and TP group shared Host KV; ending with measured end-to-end TTFT dropping from 28.83s to 18.45s (and to 14.47s with cache-aware plus dp-aware routing)."
pubDate: 2026-09-15
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys11-hicache-l2-kunlunxin-tuning
layout: ../../../layouts/BlogPost.astro
---

## 0. The one-line main thread

> **Whether tiered caching pays off depends not only on how high the hit rate can go, but also on how low the cross-tier transfer cost can be pushed.**

HiCache extends prefix KV from HBM (L1) to Host DRAM (L2) and then to external storage (L3), trading a slower but far larger store for prefix reuse. But on Kunlunxin this mechanism defaults to a net loss — whenever the extra cross-tier transfer overhead outweighs the compute saved by hits, the whole thing slows down. Building on the NUMA x PCIe x NIC topology from sys1, this post breaks down the full path of tuning L2 from working to fast on P800/P900.

## 1. Why tiered prefix caching is needed

### 1.1 The problem: L1 capacity vs prefix reuse

The Prefill stage of LLM inference repeats a lot of computation. In Agent and multi-turn traffic, system prompts, tool definitions, and history overlap heavily across requests, so prefix duplication is high. Radix Cache reuses that computation: keep computed prefix KV in HBM, skip recompute on the next hit.

But Radix Cache lives on XPU HBM, its capacity hard-capped by device memory — under high concurrency the cache is evicted within minutes. Device memory cannot grow, yet two idle tiers sit unused: **hundreds of GB to TB of Host DRAM per node**, and **cluster-level external storage**. HiCache is about extending prefix caching into those two tiers.

### 1.2 The tiered design

HiCache splits the prefix index from the KV data location: the Radix tree only answers which node a token sequence maps to; where that node's KV actually lives is a separate concern.

<div class="fig">
<svg viewBox="0 0 680 432" width="100%" font-size="13" font-weight="400" role="img" aria-label="HiCache three-tier prefix cache architecture">
<title>HiCache three-tier prefix cache architecture</title>
<defs>
<marker id="ar1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<text x="340" y="34" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#1a1a1a">HiCache Three-Tier Prefix Cache</text>
<text x="340" y="54" text-anchor="middle" dominant-baseline="central" fill="#374151">Radix tree indexes, KV data placed per tier</text>
<rect x="200" y="78" width="280" height="60" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text x="340" y="102" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#26215C">L1 · XPU HBM</text>
<text x="340" y="122" text-anchor="middle" dominant-baseline="central" fill="#3C3489">10 GB · direct hit · no overhead</text>
<rect x="200" y="168" width="280" height="60" rx="10" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="340" y="192" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#04342C">L2 · Host DRAM</text>
<text x="340" y="212" text-anchor="middle" dominant-baseline="central" fill="#085041">100 GB – TB · PCIe (D2H / H2D)</text>
<rect x="200" y="258" width="280" height="60" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="340" y="282" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#042C53">L3 · External Storage</text>
<text x="340" y="302" text-anchor="middle" dominant-baseline="central" fill="#0C447C">TB – PB · net+IO · via L2</text>
<path d="M312 138 L312 168" stroke="#5F5E5A" stroke-width="1.5" fill="none" marker-end="url(#ar1)"/>
<path d="M368 168 L368 138" stroke="#5F5E5A" stroke-width="1.5" fill="none" marker-end="url(#ar1)"/>
<text x="300" y="154" text-anchor="end" dominant-baseline="central" fill="#444441">evict down</text>
<text x="380" y="154" text-anchor="start" dominant-baseline="central" fill="#444441">load up</text>
<path d="M312 228 L312 258" stroke="#5F5E5A" stroke-width="1.5" fill="none" marker-end="url(#ar1)"/>
<path d="M368 258 L368 228" stroke="#5F5E5A" stroke-width="1.5" fill="none" marker-end="url(#ar1)"/>
<text x="300" y="244" text-anchor="end" dominant-baseline="central" fill="#444441">evict</text>
<text x="380" y="244" text-anchor="start" dominant-baseline="central" fill="#444441">load</text>
<rect x="120" y="346" width="440" height="64" rx="10" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="370" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#2C2C2A">net benefit = compute saved by hit-rate − cross-tier transfer</text>
<text x="340" y="392" text-anchor="middle" dominant-baseline="central" fill="#444441">hit-rate up and transfer cost down together decide the sign</text>
</svg>
<p class="cap">Fig 1: L1/L2/L3. When a node is evicted from L1, its KV can sink to L2 instead of being dropped; on the next prefix hit it loads back to L1. L3 is the next-level backstop for L2 and can be shared across instances.</p>
</div>

After HiCache, the Radix tree is unchanged: a token sequence still maps to one node, prefix match / refcount / eviction logic stay the same. Only where the node's KV lives changes. That defines a metric running through this whole post:

```
net benefit = compute time saved by higher hit rate − extra cost of cross-tier data movement
```

Every later change either raises the hit rate (bigger L2 / better L2 efficiency / L3) or lowers cross-tier transfer cost — which splits into four parts: 1) transfer bandwidth, 2) layout-transpose overhead, 3) operator-launch overhead, 4) interference of extra transfers with the main inference loop.

### 1.3 The adaptation surface on Kunlunxin

Landing HiCache on P800/P900 touches three layers:

- **Operator layer**: upstream cross-tier copy kernels are CUDA; a functionally identical set must be rebuilt on Kunlunxin. Until then it will not run at all.
- **Framework layer**: which thread submits transfers, what granularity to split, what data to write, how many copies. Does not block correctness, but directly decides the sign of net benefit.
- **Host layer**: NUMA topology, page-table scale, PCIe ordering. Usually hidden by drivers and hardware under CUDA, but on Kunlunxin it shows up directly as performance.

## 2. L2 adaptation and tuning

### 2.1 Layout choice: page_first_direct to layer_first

HiCache L2 has three KV layouts: `layer_first`, `page_first`, `page_first_direct`. The first build chose `page_first_direct` — whole page contiguous, single layer page_size tokens contiguous, suited to direct (DMA) copy.

A quick test was discouraging: with HiCache off, the first request took 5.95s; with it on, 15.44s, and even the all-hit Replay request got slightly worse. The cost tracks the amount of data written back, not a fixed one-time overhead.

The real problem is the **DMA submission count**. `page_first_direct` dims are `(page_num, layer_num, page_size, 1, kv_dim)`, page outermost. On the host side, different layers within one page are adjacent and can merge, but the same layer across different pages is always `layer_num × page_size` apart and can never merge; on the device side, different layers are different buffers and inherently non-contiguous. Combined, the segment count is fixed at `page_num × layer_num`.

Switching to `layer_first` (dims `(layer, size, 1, kv_dim)`), same-layer pages become contiguous and merge into one segment. At 32K tokens, page_size 64, 78 layers: 512 pages × 78 = 39,936 segments; with layer_first the ideal is **78 contiguous copies** — roughly two orders of magnitude fewer. Each segment needs its own address computation and a DMA descriptor; the count itself decides whether single-layer transfer reaches near-sequential bandwidth.

<div class="fig">
<svg viewBox="0 0 680 400" width="100%" font-size="13" font-weight="400" role="img" aria-label="KV layout and DMA segment count">
<title>KV layout vs DMA segment count</title>
<defs>
<marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<text x="340" y="34" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#1a1a1a">KV Layout Decides DMA Segment Count</text>
<text x="340" y="54" text-anchor="middle" dominant-baseline="central" fill="#374151">eg: 32K tokens · page_size 64 · 78 layers → 512 pages</text>
<rect x="40" y="80" width="270" height="200" rx="10" fill="#FAECE7" stroke="#993C1D" stroke-width="0.5"/>
<text x="175" y="104" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#4A1B0C">page_first_direct</text>
<text x="175" y="124" text-anchor="middle" dominant-baseline="central" fill="#712B13">page outer dim · cross-layer within page</text>
<rect x="70" y="142" width="48" height="120" rx="4" fill="#F0997B" stroke="#993C1D" stroke-width="0.5"/>
<rect x="124" y="142" width="48" height="120" rx="4" fill="#F0997B" stroke="#993C1D" stroke-width="0.5"/>
<rect x="178" y="142" width="48" height="120" rx="4" fill="#F0997B" stroke="#993C1D" stroke-width="0.5"/>
<rect x="232" y="142" width="48" height="120" rx="4" fill="#F0997B" stroke="#993C1D" stroke-width="0.5"/>
<text x="153" y="284" text-anchor="middle" dominant-baseline="central" fill="#712B13">each page split into layer_num segments</text>
<text x="175" y="318" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#4A1B0C">segments = page_num × layer_num</text>
<text x="175" y="342" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#993C1D">512 × 78 = 39,936</text>
<path d="M330 190 L350 190" stroke="#5F5E5A" stroke-width="1.5" fill="none" marker-end="url(#ar2)"/>
<text x="340" y="176" text-anchor="middle" dominant-baseline="central" fill="#444441">↓ ~2 orders of magnitude</text>
<rect x="370" y="80" width="270" height="200" rx="10" fill="#EAF3DE" stroke="#3B6D11" stroke-width="0.5"/>
<text x="505" y="104" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#173404">layer_first</text>
<text x="505" y="124" text-anchor="middle" dominant-baseline="central" fill="#27500A">layer outer dim · pages merged</text>
<rect x="400" y="146" width="210" height="22" rx="4" fill="#97C459" stroke="#3B6D11" stroke-width="0.5"/>
<rect x="400" y="176" width="210" height="22" rx="4" fill="#97C459" stroke="#3B6D11" stroke-width="0.5"/>
<rect x="400" y="206" width="210" height="22" rx="4" fill="#97C459" stroke="#3B6D11" stroke-width="0.5"/>
<rect x="400" y="236" width="210" height="22" rx="4" fill="#97C459" stroke="#3B6D11" stroke-width="0.5"/>
<text x="505" y="284" text-anchor="middle" dominant-baseline="central" fill="#27500A">same layer pages contiguous merged</text>
<text x="505" y="318" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#173404">segments = layer_num</text>
<text x="505" y="342" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#3B6D11">78 segments</text>
<rect x="120" y="356" width="440" height="34" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="373" text-anchor="middle" dominant-baseline="central" fill="#2C2C2A">both sides same layout → no transpose → net benefit turns positive (cold 10.7s → 7.9s)</text>
</svg>
<p class="cap">Fig 2: Layout choice decides the DMA segment count. layer_first makes host and device layouts identical, so no transpose is needed inside the operator; segments drop from page_num×layer_num to layer_num.</p>
</div>

The main route is set: **layer_first + bidirectional async**. With host layout switched to layer_first, the copy operators become same-layout `transfer_kv_per_layer_mla` (H2D per layer) / `transfer_kv_all_layer_mla` (D2H whole); cold-start first request drops from 10.74s to 7.91s.

### 2.2 Move transfer submission out of the scheduler thread

In the first build, the CPU-side submission (Kernel Launch) of H2D/D2H copy operators was inlined in the scheduler thread's `cache_unfinished_req` path. For D2H, a single `transfer_kv_all_layer_direct_lf_pf` submission typically took 1.3s, and the scheduler loop could not continue until it returned. One request writes back multiple chunks, enough to explain the extra 9 seconds.

The fix: give D2H and H2D each a daemon thread consuming a queue. `start_writing()` no longer inlines submission but enqueues the request; a backup thread submits on a dedicated stream. `start_loading()` does the same into a load queue. Async removes the scheduler blockage, but the non-hit path was still ~26% slower than no HiCache — so the cost is not only about who submits.

### 2.3 The real problem: DMA submission count

Async moves submission off the main thread, but the amount of data moved per writeback does not shrink. The cost is in the DMA submission count (see 2.1). Switching to layer_first cuts segments by ~two orders of magnitude, so single-layer transfer can approach sequential bandwidth. This is what flips the L2 main route from net loss to net gain.

### 2.4 The physical form of Host memory

Async fixes who submits, but two remaining costs come from the physical form of Host memory:

- **NUMA affinity**: P800 cards across a node land on different NUMA nodes. If the Host KV pool is on a remote node, every D2H/H2D crosses NUMA and contends with local RDMA registered memory. Pin each card to its NUMA node and disable framework auto-detection.
- **Transparent huge pages (THP)**: a 100GB Host buffer with 4KB pages needs ~26M page-table entries, taxing CPU page tables, IOMMU, and device DMA translation. 2MB huge pages cut entries ~512x. Use anonymous mmap + `madvise(MADV_HUGEPAGE)`, and `madvise` must come before `cudaHostRegister` (registration pins pages, after which THP can no longer merge).
- **LIFO free list**: default FIFO pushes freed slots to the tail and scans the whole pool; LIFO reuses the most recently freed slot, concentrating writes on a few hot pages for a pure locality win (no change to hit rate or eviction).

Measured: registered memory size is itself a cost — 80GB to 15GB, degradation falls monotonically from +20.6% to +8.8%; binding Host KV to the local NUMA per XPU topology removes most degradation (+11.5% to -2.1%).

### 2.5 Platform-side PCIe ordering: IDO

Hygon has a hardware constraint: of 128 high-speed PCIe lanes, 64 per CPU are fixed for CPU interconnect, leaving only 4 PCIe x16. If each P800 took a dedicated x16, the main NIC would have no PCIe left — so **two XPUs share one PCIe x16 uplink**. The direct consequence: binding strictly by P800 NUMA topology uses only half the machine memory (a typical 1.5T box gives ~750GB; split across 8 cards minus component overhead, ~75GB per card max), becoming a capacity bottleneck.

Forcing all NUMA nodes (e.g. H3C topology `1 1 3 3 5 5 7 7`, SGLang set to `0 1 2 3 4 5 6 7`) uses all memory but D2H bandwidth collapses. Root cause: the two XPUs under one Switch should write memory independently, but **strict PCIe ordering makes their writes block each other**.

<div class="fig">
<svg viewBox="0 0 680 380" width="100%" font-size="13" font-weight="400" role="img" aria-label="NUMA topology and PCIe ordering">
<title>NUMA topology and PCIe ordering</title>
<defs>
<marker id="ar3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<text x="340" y="34" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#1a1a1a">NUMA Topology and PCIe Ordering (Hygon P800)</text>
<text x="340" y="54" text-anchor="middle" dominant-baseline="central" fill="#374151">two XPUs share one Switch x16 uplink</text>
<rect x="180" y="80" width="320" height="120" rx="10" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="100" text-anchor="middle" dominant-baseline="central" fill="#2C2C2A">PCIe Switch (SW0)</text>
<rect x="60" y="230" width="200" height="64" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text x="160" y="254" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#26215C">XPU0 → NUMA0</text>
<text x="160" y="274" text-anchor="middle" dominant-baseline="central" fill="#3C3489">D2H write txn ID=A</text>
<rect x="420" y="230" width="200" height="64" rx="10" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="520" y="254" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#04342C">XPU1 → NUMA1</text>
<text x="520" y="274" text-anchor="middle" dominant-baseline="central" fill="#085041">D2H write txn ID=B</text>
<rect x="280" y="230" width="120" height="64" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="340" y="254" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#042C53">NIC1</text>
<text x="340" y="274" text-anchor="middle" dominant-baseline="central" fill="#0C447C">RDMA contention</text>
<path d="M160 294 L260 200" stroke="#534AB7" stroke-width="1.5" fill="none" marker-end="url(#ar3)"/>
<path d="M520 294 L420 200" stroke="#0F6E56" stroke-width="1.5" fill="none" marker-end="url(#ar3)"/>
<path d="M340 294 L340 200" stroke="#185FA5" stroke-width="1.5" fill="none" marker-end="url(#ar3)"/>
<rect x="60" y="316" width="560" height="44" rx="8" fill="#FAECE7" stroke="#993C1D" stroke-width="0.5"/>
<text x="340" y="334" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#4A1B0C">strict ordering: ID=A and ID=B block → D2H bandwidth down</text>
<text x="340" y="352" text-anchor="middle" dominant-baseline="central" fill="#712B13">use IDO (ID-Based Ordering): different-source IDs no longer wait → bandwidth back</text>
</svg>
<p class="cap">Fig 3: Two XPUs share one Switch x16 uplink. Strict PCIe ordering blocks the two D2H writes against each other; switching to ID-Based Ordering (IDO) lets different-source-ID writes proceed independently and D2H bandwidth recovers.</p>
</div>

The fix is to flip the PCIe RC communication policy to **ID-Based Ordering (IDO)** via a CPU register, so writes from different source IDs no longer wait for each other. Applies to NUMA one-to-one binding scenarios; after enabling, bandwidth recovers as expected.

### 2.6 Write-policy semantic fix + Ghost List

L1 to L2 write policy defaults to `write_back` / `write_through` / `write_through_selective` (backup only when L1 hit count ≥ 2, filtering one-shot prefixes). But under PD-disaggregated Prefill, the same request is inserted into the Radix tree twice (`cache_unfinished_req` once, `cache_finished_req` once), so a single request pushes `hit_count` to 2, and `write_through_selective` silently degrades to `write_through` — a semantic surprise.

Two fixes:

1. **Suppress same-request double counting**: skip `hit_count` accumulation at the finished stage, so `hit_count` again means distinct requests.
2. **Ghost List**: the fix introduces a new problem — if the reuse gap exceeds the prefix's L1 residence, the second arrival finds the node already evicted and `hit_count` reset, dropping L2 hit rate. Ghost List keeps an extra bounded Hash Map of page hashes evicted from L1 (page granularity, not node, because chunk boundaries and partial-hit lengths differ across requests; the page-boundary hash accumulates from root and encodes both content and position). On rebuild, scan backward; the first hit is the longest reuse prefix; write only the proven-reused segment. Key is 64-bit, OrderedDict gives O(1) add/lookup, ~120B per entry.

<div class="fig">
<svg viewBox="0 0 680 420" width="100%" font-size="13" font-weight="400" role="img" aria-label="Write policy Ghost List TP shared">
<title>Write policy / Ghost List / TP shared</title>
<rect x="40" y="80" width="600" height="96" rx="10" fill="#FAEEDA" stroke="#854F0B" stroke-width="0.5"/>
<text x="60" y="104" text-anchor="start" dominant-baseline="central" font-weight="500" fill="#412402">1 · write_through_selective semantic fix</text>
<text x="60" y="128" text-anchor="start" dominant-baseline="central" fill="#633806">PD disagg inserts same req twice → hit_count inflated → silently degrades to write_through</text>
<text x="60" y="152" text-anchor="start" dominant-baseline="central" fill="#633806">fix: skip accumulation at finished stage → hit_count again counts distinct requests</text>
<rect x="40" y="188" width="600" height="96" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text x="60" y="212" text-anchor="start" dominant-baseline="central" font-weight="500" fill="#26215C">2 · Ghost List</text>
<text x="60" y="236" text-anchor="start" dominant-baseline="central" fill="#3C3489">page-granularity hash (64-bit) · OrderedDict LRU · O(1) add/lookup · ~120B each</text>
<text x="60" y="260" text-anchor="start" dominant-baseline="central" fill="#3C3489">scan backward, first hit = longest reuse prefix · write only proven segment · bounded</text>
<rect x="40" y="296" width="600" height="96" rx="10" fill="#EAF3DE" stroke="#3B6D11" stroke-width="0.5"/>
<text x="60" y="320" text-anchor="start" dominant-baseline="central" font-weight="500" fill="#173404">3 · TP group shared Host KV</text>
<text x="60" y="344" text-anchor="start" dominant-baseline="central" fill="#27500A">MAP_SHARED buffer · only tp0 does D2H · all ranks read same indices</text>
<text x="60" y="368" text-anchor="start" dominant-baseline="central" fill="#27500A">MLA/NSA replicated latent bit-identical → L2 effective cap ×8 · D2H traffic ÷8</text>
</svg>
<p class="cap">Fig 4: Three software optimizations. Write-policy semantic fix plus Ghost List cut useless writebacks; TP group shared Host KV merges N redundant copies into one, enlarging effective capacity and shrinking D2H traffic.</p>
</div>

This policy is off by default — its benefit is conditional: whether TTFT saved by fewer D2H writes beats TTFT lost to lower hit rate depends on L2 capacity and the reuse-distance distribution. The tighter L2 is, the more filtering pays (it keeps one-shot prefixes out so limited L2 holds only proven-reusable high-value KV).

### 2.7 TP group shared Host KV

This is the single biggest win. For MLA/NSA, the Host-side KV is a replicated latent — every TP rank holds the bit-identical same data (even with dp attention, redundancy remains when TP≠1). So TP=8 means: 8 identical Host memory copies (L2 effective capacity only 1/8 of nominal) + 8x D2H traffic (while PCIe bandwidth, IOMMU/TLB budget, and kernel-launch quota are per-device or host-shared).

Mechanism: one TP group shares a `MAP_SHARED` Host buffer; only rank 0 in the group does D2H; all ranks read with the same indices. Media constraints dictate `/dev/shm` + `MADV_HUGEPAGE` (XPU driver cannot register hugetlbfs, but THP-backed tmpfs registers fine), with `MPOL_INTERLEAVE` temporarily set across all online nodes during registration. A side effect: under shared mode, NUMA binding becomes optional (L2 memory is already interleaved across nodes). Correctness relies on an implicit invariant: host pool alloc/free must be SPMD-deterministic across ranks, with an optional self-check for cross-rank hash consistency.

## 3. Results

End-to-end test on GLM5 Workbuddy 24K–64K, prefix reuse rate 0.82, PD disagg, concurrency 10:

| Stage | Key change | TTFT | Prefix hit rate | vs Base |
|---|---|---|---|---|
| Base | HiCache off, L1 Radix only | 28.83s | 46.28% | — |
| Main route | layer_first + kernel backend + bidirectional async | 22.08s | 69.23% | -23.42% |
| + Host mem form | NUMA / hugepage / LIFO | 20.86s | 69.23% | -27.66% |
| + TP shared Host KV | drop N copies, L2 to 400GB | 19.12s | 72.79% | -33.69% |
| + L2 capacity tune | back to 200GB, hit-rate saturates | 18.45s | 71.94% | -36.01% |

<div class="fig">
<svg viewBox="0 0 680 420" width="100%" font-size="13" font-weight="400" role="img" aria-label="End-to-end TTFT waterfall">
<title>End-to-end TTFT waterfall</title>
<defs>
<marker id="ar5" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<text x="340" y="34" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#1a1a1a">End-to-End TTFT Waterfall</text>
<text x="340" y="54" text-anchor="middle" dominant-baseline="central" fill="#374151">GLM5 Workbuddy 24K–64K · reuse rate 0.82 · concurrency 10</text>
<rect x="180" y="78" width="500" height="36" rx="6" fill="#F09595" stroke="#A32D2D" stroke-width="0.5"/>
<text x="188" y="96" text-anchor="start" dominant-baseline="central" fill="#501313">Base · L1 Radix only</text>
<text x="672" y="96" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#501313">28.83s</text>
<rect x="180" y="124" width="383" height="36" rx="6" fill="#F7C1C1" stroke="#A32D2D" stroke-width="0.5"/>
<text x="188" y="142" text-anchor="start" dominant-baseline="central" fill="#501313">+ layer_first + bidirectional async</text>
<text x="555" y="142" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#501313">22.08s · -23.4%</text>
<rect x="180" y="170" width="362" height="36" rx="6" fill="#FAC775" stroke="#854F0B" stroke-width="0.5"/>
<text x="188" y="188" text-anchor="start" dominant-baseline="central" fill="#412402">+ Host mem form (NUMA/hugepage/LIFO)</text>
<text x="534" y="188" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#412402">20.86s · -27.7%</text>
<rect x="180" y="216" width="332" height="36" rx="6" fill="#C0DD97" stroke="#3B6D11" stroke-width="0.5"/>
<text x="188" y="234" text-anchor="start" dominant-baseline="central" fill="#173404">+ TP group shared Host KV</text>
<text x="504" y="234" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#173404">19.12s · -33.7%</text>
<rect x="180" y="262" width="320" height="36" rx="6" fill="#97C459" stroke="#3B6D11" stroke-width="0.5"/>
<text x="188" y="280" text-anchor="start" dominant-baseline="central" fill="#173404">+ L2 capacity tune (200GB)</text>
<text x="492" y="280" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#173404">18.45s · -36.0%</text>
<rect x="180" y="308" width="251" height="36" rx="6" fill="#639922" stroke="#3B6D11" stroke-width="0.5"/>
<text x="188" y="326" text-anchor="start" dominant-baseline="central" fill="#ffffff">+ cache-aware + dp-aware routing</text>
<text x="423" y="326" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#ffffff">14.47s · -42.6%</text>
<line x1="180" y1="64" x2="180" y2="360" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="174" y="96" text-anchor="end" dominant-baseline="central" fill="#444441">0</text>
<rect x="60" y="364" width="560" height="34" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="381" text-anchor="middle" dominant-baseline="central" fill="#2C2C2A">hit-rate 46% to 72% is the main gain; the 12.6pt from -23% to -36% is all transfer-path optimization</text>
</svg>
<p class="cap">Fig 5: End-to-end TTFT waterfall. From 28.83s down to 18.45s; with cache-aware + dp-aware routing, TTFT drops further to 14.47s.</p>
</div>

Two key conclusions:

- **Gain comes from two sides**: hit rate rising from 46.28% to ~72% (+25pt) is the main body, from L2 enlarging effective capacity by an order of magnitude; the 12.6pt from -23.42% to -36.01% is entirely transfer-path optimization — no model-compute change, no higher hit-rate ceiling, purely pushing down cross-tier transfer overhead. This confirms the net-benefit formula.
- **Bigger is not better**: the last row is capacity turned down — with hit rate roughly flat, the smaller L2 is actually faster, because registered memory size is itself a cost. The right approach is to try capacity tier by tier from small, watch the marginal gain in hit rate, and stop at the plateau; capacity beyond the plateau only adds registered-memory cost that slows DMA and RDMA.

The all-hit fit: `T(n) = 779.1ms + 13.539us · n`, where 779ms ≈ 735ms (shared deployment floor) + 44ms (L2-path-only). Measured H2D bandwidth ~7.40 GB/s. Break-even is around 3K tokens: only hits above 3K tokens pay off.

## 4. Summary and outlook

HiCache L2 on Kunlunxin P800/P900 is already practically useful, with an implementation centered on **layout optimization, async transfer, NUMA affinity, and TP group sharing**. But every number here is measured in a specific environment and should not be extrapolated to other platforms or workloads — when switching, re-attribute in the same order: confirm the host side (ordering, NUMA, huge pages) leaves no obvious single-transfer cost, then probe hit-rate marginal gain tier by tier, and only then touch dataset-sensitive switches like write policy.

The future is **L3**: Host DRAM alone is not enough for longer context and higher efficiency. The fastest, cheapest path is to pool the **D node Host DRAM via Mooncake** in PD disagg — D nodes run Decode and their Host DRAM is long underused; pooling needs no new hardware and gives Prefill instances sizable L3 capacity, still DRAM so cross-tier cost is far below disk; it also turns single-node private cache into cluster-shared cache, freeing hit rate from single-instance locality. Remaining questions: RDMA interference with L2 writeback, and pool-metadata consistency at scale.

## 5. Series navigation

- **sys1 (NUMA x PCIe x NIC under P900 + HiCache)** sets the hardware topology — the background for why L2 transfer needs parallelism, pinning, and IDO.
- **sys9 / sys10 (KV Cache landscape and production)** cover principles, taxonomy, and SGLang / vLLM / AttentionStore implementations; this post is the Kunlunxin L2 tuning follow-up.
- One line threading them: **sys1 sees the link → sys9/sys10 explain the principle and engines → sys11 makes L2 fast on Kunlunxin.**
