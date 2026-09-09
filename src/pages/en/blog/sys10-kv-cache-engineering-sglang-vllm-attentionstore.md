---
title: 'Inference Systems Infrastructure Notes (10): KV Cache in Production — SGLang Two-Level Pools, vLLM Block Management, and AttentionStore Tiered Caching'
description: 'From principles to code: SGLang req_to_token_pool to token_to_kv_pool two-level mapping and Radix Tree prefix reuse, HiCache three-tier storage with prefetch overlap; vLLM PagedAttention, profiling to size blocks, cross-layer unified layout (PR 27743), the LRU block pool and chained prefix hashing; plus Baidu AttentionStore measured on Kunlunxin P800 (6.2x lower TTFT at 64K context, 5.4x multi-turn throughput).'
pubDate: 2026-09-09
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys10-kv-cache-engineering-sglang-vllm-attentionstore
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

sys9 covered how KV Cache should be trimmed. This post looks at **how two leading engines and one industrial caching system actually manage it**: SGLang uses a **two-level pool plus a radix tree** for token-granular prefix reuse; vLLM uses **paging plus an LRU block pool** to minimize fragmentation; Baidu AttentionStore widens the view from one card to the **whole cluster**, giving the scheduler eyes to see where cache already lives.

## 1. SGLang: The Two-Level Memory Pool

### 1.1 Context first: the Scheduler event loop

KV Cache management is spread across the scheduler, so start with the main loop:

```
Event Loop, forever:
  process_input_requests   -> receive, classify, enqueue to waiting_queue
  get_next_batch_to_run    -> form a batch (prefill first, else refresh the decode batch)
  run_batch                -> forward (generation / idle / embedding)
  process_batch_result     -> handle outputs, update state, release or cache KV
```

Under memory pressure, prefill requests get **chunked** and decode requests get **retracted**, then pushed back into the waiting queue.

### 1.2 The two pools

SGLang splits "request to token to actual KV data" into two mapping levels:

<div class="fig">
<svg viewBox="0 0 660 250" role="img" aria-label="SGLang two-level memory pool mapping">
  <title>Figure 1: SGLang two-level pools and tree_cache</title>
  <rect x="14" y="56" width="112" height="44" rx="6" fill="#f6f8fa" stroke="#30363d"/>
  <text x="70" y="76" font-size="12" fill="#6b7280" text-anchor="middle">Request</text>
  <text x="70" y="92" font-size="13" fill="#1a1a1a" text-anchor="middle">token seq ABC</text>
  <line x1="130" y1="78" x2="164" y2="78" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="164,78 156,74 156,82" fill="#2563eb"/>
  <rect x="170" y="42" width="196" height="72" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="268" y="62" font-size="12" fill="#2563eb" text-anchor="middle">req_to_token_pool</text>
  <text x="268" y="80" font-size="11" fill="#1a1a1a" text-anchor="middle">[req_pool_idx][pos]</text>
  <text x="268" y="98" font-size="11" fill="#1a1a1a" text-anchor="middle">to out_cache_loc</text>
  <line x1="370" y1="78" x2="404" y2="78" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="404,78 396,74 396,82" fill="#2563eb"/>
  <rect x="410" y="30" width="236" height="96" rx="6" fill="#ecfdf5" stroke="#10b981"/>
  <text x="528" y="50" font-size="12" fill="#047857" text-anchor="middle">token_to_kv_pool</text>
  <text x="528" y="68" font-size="11" fill="#1a1a1a" text-anchor="middle">[layer][out_cache_loc]</text>
  <text x="528" y="84" font-size="11" fill="#1a1a1a" text-anchor="middle">[head][head_dim]</text>
  <text x="528" y="102" font-size="11" fill="#1a1a1a" text-anchor="middle">to cache_k, cache_v</text>
  <text x="528" y="118" font-size="11" fill="#047857" text-anchor="middle">the real bulk in HBM</text>
  <rect x="170" y="164" width="196" height="60" rx="6" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="268" y="184" font-size="12" fill="#854f0b" text-anchor="middle">tree_cache (RadixCache)</text>
  <text x="268" y="202" font-size="11" fill="#1a1a1a" text-anchor="middle">token id to out_cache_loc</text>
  <text x="268" y="218" font-size="11" fill="#1a1a1a" text-anchor="middle">prefix reuse across requests</text>
  <line x1="268" y1="164" x2="268" y2="118" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="14" y="234" font-size="11" fill="#6b7280">Level one tracks whose token at which position; level two maps that index to real KV; tree_cache decides what can be reused.</text>
</svg>
<figcaption>Figure 1: level one maps a request to KV indices, level two maps indices to real KV data, and tree_cache handles cross-request prefix reuse.</figcaption>
</div>

| Pool | Shape | Indexed by | Returns |
|------|-------|-----------|---------|
| **req_to_token_pool** | max_running_requests × context_len | `[req_pool_idx][pos]` | `out_cache_loc` (a KV index) |
| **token_to_kv_pool** | layers × max_tokens × heads × head_dim | `[layer_id][out_cache_loc][head][dim]` | `cache_k`, `cache_v` |

A forward pass usually fetches **a whole layer at once**, because that layer needs every historical token of the request.

### 1.3 tree_cache: reuse across requests

- Keys are **token ids** (the KV of a given token is request-independent); values are `out_cache_loc`.
- Implementations include `ChunkCache` (the simplified path when radix cache is off) and `PrefixCache` (which includes **RadixCache** and **HiCache**).

### 1.4 The prefill flow (request ABC)

**1. match_prefix**: find the longest prefix in the radix tree. If the tree holds `AFG` and the request is `ABC`, it matches `A` and splits `AFG` into `A` and `FG`.

**2. prepare_for_extend**:
- `req_to_token_pool`: allocate `req_pool_indices`, write in the prefix part
- `token_to_kv_pool`: allocate slots only for **unmatched tokens** → 3 input tokens minus 1 matched = **only 2 allocated** (B, C)

**3. run_batch to forward_extend**: the attention backend writes K/V for B and C into the newly allocated `out_cache_loc`; Q is B and C, while K/V come from A (cache hit) plus B and C (new).

**4. cache_unfinished_req**: attach `BC` as a child of `A`, incrementing the lock refcount.

### 1.5 The decode flow

- `prepare_for_decode`: allocate only **batch_size × 1** slots per step (one token per step per sequence)
- `run_batch to forward_decode`: Q is the single new token, K/V is the whole history
- `cache_finished_req` runs only when the request completes; **in-flight decode requests need no extra cache work** — the generated sequence is appended to the tree at the end

### 1.6 HiCache: three tiers of storage

| Module | Role |
|--------|------|
| **HiRadixTree** | GPU / CPU two-level prefix tree with native KV sync between tiers |
| **Storage Backend** | Pluggable layer, currently integrating 3FS, Mooncake, NIXL; unified `batch_get/set/exists` with zero-copy |
| **Global KVManager** | Unified metadata management for the distributed filesystem |
| **3FS** | DeepSeek open-source high-performance distributed filesystem (RDMA plus NVMe SSD, TiB/s aggregate read bandwidth) |

Two key optimizations:

1. **Prefetch overlaps with waiting**: `prefetch_from_storage` fires as soon as a request is enqueued, moving KV from storage into host memory during the queue wait. Three policies:
   - `best_effort`: if prefetch is still running when scheduled, abort it and run inference
   - `timeout`: abort only past a threshold, otherwise skip the request this round
   - `wait_complete`: only schedule once prefetch finishes
2. **Loading overlaps with compute**: host-to-GPU transfer runs on a **separate CUDA stream, layer by layer** (`load_to_device_per_layer`), so layer i can start as soon as its KV lands instead of waiting for all layers.

The effect is that blocking I/O hides inside queue waiting and GPU compute: larger effective cache capacity with minimal TTFT damage.

### 1.7 Two flavours of sparsity

**SWA (sliding window)**: each step attends only to the last W tokens, and KV outside the window is **genuinely freed**. It works through a trio — a small pool for SWA layers, immediate window recycling, and tombstones that preserve prefix matching. Compute uses a window mask; memory uses dual pools, dual LRU, and tombstones. Result: real reclamation outside the window, prefixes still shareable.

**DeepSeek NSA**: **keep all KV, read only part of it** — the opposite of SWA. An indexer scores each page and selects Top-K, fusing three paths in parallel: compressed global summary, selected pages, and a sliding local window. KV is never deleted, but attention read cost drops from O(seq) to O(K·page + W + seq/L), cutting both compute and bandwidth at long context.

## 2. vLLM: Paging, Block Pools and Connectors

### 2.1 PagedAttention

The OS virtual-memory trick applied to KV Cache:

- Each request's KV is cut into fixed-size **logical blocks** (default `block_size = 16` tokens)
- A **Block Table** records the logical-to-physical mapping
- Physical blocks may be **non-contiguous** in HBM, eliminating fragmentation

| Concept | OS analogy |
|---------|------------|
| Request | Process |
| Logical KV block | Virtual page |
| Block Table | Page table |
| Physical KV block | Physical frame |

### 2.2 Initialization: compute how many blocks fit

1. **Build dummy data**: from `max_num_seqs` and `max_num_batched_tokens`, synthesize fake requests (for example 10 tokens over 3 seqs gives lengths 4, 3, 3)
2. **Run one simulated forward** to measure peak usage:
   `KV Cache budget = GPU free memory minus (weights plus activations) minus CUDA Graph reserve`
3. **Count blocks**: `num_blocks = KV Cache budget / sum of per-layer block sizes`, where
   `per_layer_page_size = block_size × num_kv_heads × head_size × dtype_size × 2`
4. **Pre-allocate** one empty tensor that stays resident in HBM (CPU side works the same way, default 4 GiB)

### 2.3 Cross-layer unified layout (PR 27743)

The old layout gave every layer its own block and split K from V. Harmless for compute, **devastating for KV offload** — effective blocks were too small, so transfer efficiency collapsed.

PR 27743 makes one logical block span all layers contiguously:

```
old: (2, num_blocks, block_size, num_kv_heads, head_size)
new: (num_layers, 2, num_blocks, block_size, num_kv_heads, head_size)
```

Stride order then depends on the attention backend (NHD makes all layers of a block contiguous; HND favours batched access per head). **Measured block size changes**:

| Model | Old block | New block |
|-------|-----------|-----------|
| Llama-3.1-8B | 32 KB | 2 MB |
| Qwen3-32B (TP=2) | 16 KB | 2 MB |
| DeepSeek-V2-Lite (bs=64) | 72 KB | 1.9 MB |
| Qwen3-8B | 28 KB | 1.97 MB |

Offloading Connector throughput improves by **an order of magnitude**.

### 2.4 Block management components

```
KVCacheManager            <- top level, talks to the Scheduler
  └─ KVCacheCoordinator     <- coordinates multiple KV Cache groups
       └─ SingleTypeKVCacheManager  <- allocation for one block type
            └─ BlockPool           <- the physical block pool
                 └─ FreeKVCacheBlockQueue  <- LRU doubly linked list
```

`KVCacheSpec` branches by architecture: `FullAttentionSpec` (includes GQA, most common), `MLAAttentionSpec` (DeepSeek-V3, K/V merged into a latent), `SlidingWindowSpec`, `MambaSpec`, and others.

Each physical block is a `KVCacheBlock` (metadata only; data lives in the GPU tensor):

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int          # physical block number
    ref_cnt: int = 0       # refcount, 0 means free and reclaimable
    _block_hash: ...       # set only for full blocks with prefix caching on
    prev_free_block / next_free_block   # LRU list pointers
    is_null: bool = False  # the placeholder block with block_id 0
```

`BlockPool` maintains LRU order via `FreeKVCacheBlockQueue`: allocation pops the **oldest** from the head, freeing appends to the tail, and a prefix hit calls `touch()` to remove it from the middle in O(1).

### 2.5 Chained hashing for prefix caching

A block hash **depends on its predecessor**:

```python
hash_block_tokens(hash_fn, parent_block_hash, curr_block_token_ids, extra_keys)
```

So two blocks share a hash only when they sit at the same position **and** every preceding token matches. Multimodal inputs, LoRA, and `cache_salt` enter through `extra_keys` so different request types cannot collide.

Full path: `get_computed_blocks()` finds hits → `touch()` protects them → `allocate_slots()` adds new blocks → inference → `cache_full_blocks()` registers hashes for completed blocks → `free_blocks()` at the end (**keeping the hash**, evicting only when free space runs short).

### 2.6 KV Connector

A unified abstraction for saving, loading, and transferring KV between instances. It is the foundation of **PD disaggregation** and **KV offload** (LMCache, Mooncake, and similar implementations).

## 3. AttentionStore: From Migration to System

### 3.1 Three problems that must be solved

Moving KV to CPU or SSD is not enough; production hits three walls:

1. **Scheduling blind spot**: the scheduler cannot see cache distribution, so requests land on nodes without cache and trigger a full prefill recompute, wiping out the offload gain
2. **Slow data paths**: movement across HBM / DRAM / SSD lacks targeted optimization, and transfer latency eats the reuse benefit
3. **Cache dies with the process**: cache is tightly coupled to the inference process, so a restart or upgrade invalidates everything

### 3.2 Global index plus cache-aware scheduling

- **Global KV index**: aggregates KV Block metadata from every node (`BlockHash`, host medium HBM/DRAM/SSD), tracking create and destroy events to build a Host to Blocks map
- **Scheduling upgrade**: from "resource health only" to "resource plus cache scoring" — first narrow to a set of high-hit-rate nodes, then score by hit length and medium read efficiency

The goal shifts from "is it available" to "**is it optimal**".

### 3.3 Tiered caching and transfer optimization

Flow: check HBM first → on a miss, migrate from another node's host memory via node-level pooling → only compute when still missing. Prefill output goes to the decode node and is asynchronously written back to DRAM/SSD; decode increments are written back asynchronously too.

| Optimization | Approach | Effect |
|---------------|----------|--------|
| **Kunlunxin native adaptation** | XPU-native APIs for data movement, cache access and execution scheduling; a unified hardware abstraction layer | Smooth operation across hardware |
| **Read acceleration** | Fast and slow media issue transfers **in parallel** instead of serially; shared memory marked as huge pages; pinned for the full lifecycle | DRAM to HBM efficiency **4x over baseline** |
| **Transfer acceleration** | A C++ SDK moves serialization, packing and cross-node transfer out of the main process into an async thread pool; write-back and transfer split and run in parallel | KV transfer pipelines with model compute |

AttentionStore also **runs as an independent process**, decoupled from the inference engine — KV survives restarts, recovery and version upgrades, and can be restored quickly from a local index table.

### 3.4 Measured results (DeepSeek R1 671B on Kunlunxin P800)

Setup: 2 prefill nodes, TP4 / DP4.

| Scenario | Gain |
|----------|------|
| Context above 8K | TTFT improved by a steady **50% to 80%** |
| Multi-turn conversation | Overall throughput up **5.4x** |
| 64K long context | TTFT **6.2x lower** than default Chunk-Prefill |

## 4. Comparing the Three

| Dimension | SGLang | vLLM | AttentionStore |
|-----------|--------|------|----------------|
| **Core abstraction** | Two-level pool plus radix tree | Paged virtual memory plus LRU block pool | Cluster-wide global index over tiered media |
| **Reuse granularity** | Token level (prefix tree) | Block level (chained hash) | Globally addressable KV Blocks |
| **Memory governance** | Pooling plus tiering (HiCache) | Paging removes fragmentation | HBM / DRAM / SSD, three tiers |
| **Main problem solved** | Prefix reuse rate, long-context capacity | Fragmentation, scheduling efficiency | Cluster hit rate, process decoupling |
| **Typical dependencies** | 3FS / Mooncake / NIXL | KV Connector (LMCache and others) | Distributed FS plus RDMA |

One-line distinction: **SGLang and vLLM solve how a single machine manages its memory; AttentionStore solves how a cluster routes a request to cache that already exists.**

## 5. Links to the Rest of the Series

- **sys9** is principles and taxonomy; this post is its engineering landing.
- **sys1** (P900 plus HiCache over NUMA / PCIe / NIC) covers the hardware topology, which explains why HiCache tiering needs parallel transfers and page pinning.
- **sys8** (PD disaggregation) quantifies KV transfer cost (roughly 600 MB for 70B at 4k tokens in FP16) — exactly what KV Connector and AttentionStore optimize.
- **fw2** (vLLM internals) covers PagedAttention at the concept level; this post extends it to block management and cross-layer layout.
- **ag1** argues that Agentic workloads turn KV Cache from a memory problem into a scheduling problem — AttentionStore's global index is the industrial confirmation of that claim.
