---
title: "Inference Systems Infrastructure Notes (14): Extensible KV Cache — Sizing from Measurement Instead of Estimate, and How Virtual Addresses Decouple Capture from Capacity"
description: "The three layers behind vLLM PR #56492: why CUDA graph capture pins down when the KV cache must be allocated (a graph stores pointers, not tensors), why the 8 percent gpu_memory_utilization reserve equals roughly 15 GiB idle on a 192 GiB card, and how cuMemAddressReserve plus cuMemMap let sizing move after warmup without recapturing. Includes why each term in the sizing formula looks the way it does, the measured gains across eight models on GB200, why DeepSeek-V4-Flash gains only 4.1 percent, why 433k to 672k on H200 is not a general expectation, the capacity-latency mechanism behind the 6 to 18 percent TPOT increase, and the correct rollback (0.9 is not a rollback)."
pubDate: 2026-11-25
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys14-extensible-kv-cache-measured-sizing
layout: ../../../layouts/BlogPost.astro
---

## 0. The one-line thesis

The old way vLLM sized its KV cache was to **make an irreversible decision at the moment of least information**. The cache had to exist before CUDA graph capture, because a captured graph bakes in addresses. Everything that only appears after capture and warmup — the graph private pool, attention workspaces, sampler and logprobs buffers, speculative decode drafts — could only be guessed from a single profiling pass and absorbed by an 8 percent reserve.

PR `#56492` splits the decision in two. The address range is reserved before capture using the driver virtual memory API, and physical pages are committed only after warmup, once the actually free memory is known. Addresses never move, so the captured graphs stay valid. Capacity is measured, so the 8 percent reserve can go back to work.

One sentence is enough to place the feature: **it is not an allocator optimization, it is a reordering in time**.

## 1. Where the constraint comes from

### 1.1 A CUDA graph stores pointers, not tensors

Capturing a graph snapshots a sequence of kernel launches: grid and block dimensions, and **the device address of every tensor argument**. Replay does not re-resolve tensors. It reads from the addresses in the snapshot.

So the order cannot be swapped:

```text
KV cache must exist first (to own a fixed VA range)
    ↓
only then can capture run (addresses go into the graph)
    ↓
capture and warmup themselves need more memory
    ↓
but by then the cache size is already frozen
```

That is the awkward part. **The KV cache is the largest allocation in memory, and it is also the earliest one decided.**

### 1.2 Three guesses the old path had to make

One profiling sample only sees the model forward. It knows nothing about what happens later, so three things were inferred:

| What needs reserving | Old situation | Cost of guessing wrong |
|---|---|---|
| **CUDA graph memory** | Either ignored, or measured with a throwaway capture pass | Ignored: runtime OOM. Measured: tens of seconds of wasted startup |
| **Warmup transients** (attention workspaces, sampler and logprobs, spec-decode drafts) | Profiling never exercises these paths | Absorbed by the reserve only |
| **Runtime peaks** (long prefill, large decode batch, chunked context on prefix cache hits) | Completely unknown | Same |

### 1.3 What the 8 percent actually costs

`gpu_memory_utilization=0.92` leaves 0.08 of the device. On a 192 GiB card that is **about 15 GiB**. This is not designed redundancy. It is the size of the estimation gap, expressed as a percentage because nobody could compute the gap directly.

On HBM, 15 GiB of KV cache is on the order of **hundreds of thousands of tokens** for a 70B-class model with GQA and fp8 KV. Every machine running the old path was holding that capacity idle for years.

## 2. The fix: separate address from capacity

### 2.1 The VMM primitives

The CUDA driver virtual memory API splits reservation from backing:

```text
cuMemAddressReserve(&va, size, align, 0, 0)   // address only, zero physical memory
cuMemCreate(&handle, size, &prop, 0)          // allocate a physical allocation
cuMemMap(va + offset, size, 0, handle, 0)     // map it into the reserved range
cuMemSetAccess(va, size, &desc, 1)            // set read/write access
```

Release goes through `cuMemUnmap` plus `cuMemRelease`, and the address range itself can be reused. ROCm has the `hipMem*` equivalents.

These calls have existed for as long as unified virtual addressing has. They were simply never applied to the KV cache, because **nothing previously needed address-first, capacity-later semantics**.

### 2.2 ExtensibleTensor hides all of it

vLLM wraps this as `ExtensibleTensor` in `vllm/utils/extensible_tensor.py`, exposed to PyTorch through DLPack. To the layers above it looks like an ordinary device tensor: shape, strides, and `data_ptr` all behave normally.

Three implementation details matter:

- **Segment split.** The full capacity is divided into per-layer (or per-tensor) segments, each committing independently. Every layer view lands in its own contiguous VA range, so no kernel code changes.
- **Driver granule granularity** for commits, typically 2 MiB.
- **Newly committed blocks are zeroed.** A physical page may have been recycled from another process. Not zeroing it means reading stale data, which is both a correctness and a security problem.

<div class="fig">
<svg viewBox="0 0 680 236" width="100%" font-size="12" role="img" aria-label="Reserved virtual address range with demand-committed physical pages">
<title>Figure 1: One reserved virtual range, with physical pages only behind the segments that were touched</title>
<defs>
<marker id="m1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<text x="34" y="20" font-size="11" fill="#6b7280">Reserved virtual address range (capacity ceiling, fixed before capture)</text>
<rect x="34" y="28" width="612" height="40" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="0.8" stroke-dasharray="4 3"/>
<line x1="34" y1="48" x2="646" y2="48" stroke="#2563eb" stroke-width="0.5" stroke-dasharray="2 4" opacity="0.4"/>
<text x="340" y="60" font-size="10" fill="#1d4ed8" text-anchor="middle">layer 0 · layer 1 · layer 2 · … · layer L−1    (each segment contiguous, addresses never move)</text>

<text x="34" y="92" font-size="11" fill="#6b7280">Physical pages behind it (decided only after warmup)</text>

<rect x="34" y="100" width="150" height="34" rx="3" fill="#ecfdf5" stroke="#10b981" stroke-width="0.8"/>
<text x="109" y="121" font-size="10" fill="#047857" text-anchor="middle">committed</text>
<rect x="190" y="100" width="150" height="34" rx="3" fill="#ecfdf5" stroke="#10b981" stroke-width="0.8"/>
<text x="265" y="121" font-size="10" fill="#047857" text-anchor="middle">committed</text>
<rect x="346" y="100" width="150" height="34" rx="3" fill="#f9fafb" stroke="#9ca3af" stroke-width="0.8" stroke-dasharray="3 3"/>
<text x="421" y="121" font-size="10" fill="#6b7280" text-anchor="middle">unmapped, no pages</text>
<rect x="502" y="100" width="144" height="34" rx="3" fill="#f9fafb" stroke="#9ca3af" stroke-width="0.8" stroke-dasharray="3 3"/>
<text x="574" y="121" font-size="10" fill="#6b7280" text-anchor="middle">unmapped</text>

<line x1="109" y1="140" x2="109" y2="158" stroke="#10b981" stroke-width="1" marker-end="url(#m1)"/>
<line x1="265" y1="140" x2="265" y2="158" stroke="#10b981" stroke-width="1" marker-end="url(#m1)"/>

<rect x="34" y="162" width="274" height="26" rx="3" fill="#fff" stroke="#10b981" stroke-width="0.8"/>
<text x="171" y="179" font-size="10" fill="#047857" text-anchor="middle">blocks warmup actually touched  →  pages committed</text>

<rect x="34" y="200" width="612" height="28" rx="4" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="48" y="218" font-size="10" fill="#854f0b">Final step extend_kv_cache: map the measured remaining budget into the same VA range. Pointers unchanged, graphs stay valid.</text>
</svg>
<div class="cap">Figure 1: Reserve and commit are separate operations. The dashed region carries zero physical memory, which is why reserving early costs almost nothing.</div>
</div>

### 2.3 Stable addresses are the hard requirement

This is the one point that cannot be compromised. If the final capacity did not fit the reservation and the cache had to move, then:

- every captured CUDA graph becomes invalid and must be recaptured, which gives back the entire startup win;
- every component holding a pointer into it breaks: attention backend workspace views, KV connector registrations, sleep mode snapshots.

So the reservation takes the profiling upper bound, conservatively large. The final step can only **add** pages into it, never exceed it. That directionality is what makes extension a mapping operation rather than a relocation.

The same stable range also solves three other problems for free:

- **Sleep mode** unmaps physical pages on sleep and remaps at the same addresses on wake, so both capacity and layout survive.
- **KV connectors** are created on the worker side after the final extend, so they register the final memory. Segment boundaries are aligned to the driver granule so an RDMA registration never spans two physical chunks.
- **Weight updates.** In RL loops, `reload_weights` needs transient memory warmup cannot measure. The default flow pauses and clears the KV cache first, at which point the cache physical memory can be released for the update and remapped afterwards.

## 3. Sizing: every term has a reason

```python
budget = util * device_mem - weights
peak   = max(profiling_peak, warmup_peak)
margin = max(1 GiB, 0.75 * peak)
blocks = floor((budget - peak - margin) / bytes_per_block)
final  = min(blocks)   # over all workers and all DP engines
```

### 3.1 Why peak is a max, not a sum

The profiling peak and the warmup peak measure the same quantity at two moments. They do not coexist. Taking the maximum is correct. Adding them double counts and throws away capacity for nothing.

Taking the maximum means trusting the largest observed value. The bias is deliberate: **under-provision rather than OOM during a long prefill**.

### 3.2 Why `max(1 GiB, 0.75 * peak)`

The two floors cover two different failure modes:

- **The 1 GiB absolute floor** handles small models where the computed margin would be tens of megabytes, and one unexpected allocation hits the wall.
- **The 0.75 relative floor** acknowledges that bigger peaks jitter more. A configuration with a 15 GiB peak cannot live on a 1 GiB margin.

The 0.75 is an empirical coefficient, not a derivation. It prices the paths warmup does not cover. That is exactly why the feature falls back automatically whenever a configuration cannot use it, rather than proceeding anyway.

### 3.3 Three new warmup probes

Measurement accuracy depends entirely on warmup actually exercising the path. Three steps were added:

| New step | What it measures | Skip it and |
|---|---|---|
| Full-budget prefill | Attention workspace peak at long context | The first long prompt OOMs |
| Prompt-logprobs chunk | Logprobs temporary buffers | It blows up when a client asks for logprobs |
| Chunked context prefill for MLA | The MLA context up-projection on the prefix cache path | Cache hits cost more memory than misses |

The third one deserves attention. **MLA context up-projection only runs when there is cached context**, which an ordinary warmup never produces. That is precisely the path the old reserve was silently insuring, and getting the insurance wrong meant OOM at runtime.

### 3.4 Why the minimum across all ranks

TP and DP ranks step in lockstep. Every step must process the same number of blocks, or the collective stalls. Capacity therefore has to converge on the smallest value across all workers, delivered with a single `extend_kv_cache` RPC.

It also explains why having each rank measure its own free memory is not enough: a rank that measures more would compute more capacity, and it would still be cut down to the slowest one.

<div class="fig">
<svg viewBox="0 0 680 214" width="100%" font-size="12" role="img" aria-label="Comparison of memory budget allocation">
<title>Figure 2: The old 8 percent slack becomes a measured peak plus a computed margin</title>

<text x="34" y="18" font-size="11" fill="#6b7280">Old · gpu_memory_utilization = 0.92 (KV from estimate)</text>
<rect x="34" y="26" width="43" height="34" fill="#f3f4f6" stroke="#d1d5db" stroke-width="0.5"/>
<rect x="77" y="26" width="435" height="34" fill="#d1d5db"/>
<text x="294" y="48" font-size="10" fill="#374151" text-anchor="middle">KV cache (estimated)</text>
<rect x="512" y="26" width="86" height="34" fill="#9ca3af"/>
<rect x="598" y="26" width="48" height="34" fill="#fecaca"/>
<text x="622" y="48" font-size="9" fill="#b91c1c" text-anchor="middle">slack</text>
<text x="34" y="74" font-size="9.5" fill="#9ca3af">weights</text>
<text x="516" y="74" font-size="9.5" fill="#9ca3af">graph / workspace estimate</text>
<text x="598" y="74" font-size="9.5" fill="#b91c1c">8% idle ≈ 15 GiB</text>

<text x="34" y="108" font-size="11" fill="#6b7280">New · utilization defaults to 1.0 when unset (KV from measurement)</text>
<rect x="34" y="116" width="43" height="34" fill="#f3f4f6" stroke="#d1d5db" stroke-width="0.5"/>
<rect x="77" y="116" width="470" height="34" fill="#93c5fd"/>
<text x="312" y="138" font-size="10" fill="#1e3a8a" text-anchor="middle">KV cache (measured)</text>
<rect x="547" y="116" width="61" height="34" fill="#a7f3d0"/>
<rect x="608" y="116" width="38" height="34" fill="#ddd6fe"/>
<text x="34" y="164" font-size="9.5" fill="#9ca3af">weights</text>
<text x="547" y="164" font-size="9.5" fill="#047857">measured peak</text>
<text x="604" y="164" font-size="9.5" fill="#6d28d9">margin</text>

<rect x="34" y="178" width="612" height="26" rx="4" fill="#f6f8fa" stroke="#e5e7eb" stroke-width="0.5"/>
<text x="48" y="195" font-size="10" fill="#374151">The saving is not a smarter algorithm. It is no longer pricing the unknown as a percentage. A margin is computable; slack is not.</text>
</svg>
<div class="cap">Figure 2: Proportions follow the Qwen3-8B direction on GB200 (150.2 to 161.6 GiB), not per-card data.</div>
</div>

## 4. Gains, and where they stop

GB200, 189 GiB per GPU, everything else default. All eight models gain:

| Model | TP | Old sizing | New default | Token gain | Engine init | End to end |
|---|---|---|---|---|---|---|
| Qwen3-8B | 1 | 150.2 GiB | 161.6 GiB | +7.5% | −29% | −18% |
| Qwen3-8B + EAGLE3 | 1 | 147.3 GiB | 157.4 GiB | +6.8% | −32% | −14% |
| Qwen3-32B | 1 | 103.1 GiB | 113.0 GiB | +9.5% | −30% | −12% |
| gpt-oss-20b | 1 | 150.5 GiB | 161.0 GiB | +7.0% | −28% | −11% |
| Qwen3.6-35B-A3B | 1 | 97.6 GiB | 106.2 GiB | +8.8% | −21% | −9% |
| MiniMax-M2.7 | 4 | 109.7 GiB | 118.4 GiB | +7.9% | −31% | −6% |
| Qwen3-235B-A22B-FP8 | 4 | 108.1 GiB | 117.1 GiB | +8.2% | −34% | −17% |
| DeepSeek-V4-Flash | 4 | 119.9 GiB | 124.9 GiB | +4.1% | −30% | −9% |

Two observations.

**The startup win comes from capturing one fewer time, not from capturing faster.** The old path ran a throwaway capture pass solely to estimate graph memory. The real capture costs the same in both variants.

**DeepSeek-V4-Flash at 4.1 percent is the row to study.** Its graphs and workspaces genuinely need about 15 GiB after warmup, which means **the old 8 percent slack was already being consumed**. There was little left to reclaim. That gives you a way to predict your own benefit:

> Your gain is roughly how much the old path overestimated you.
> Large graph pools and workspaces (big batches, many graph sizes, spec decode, MLA) mean the slack was already in use. Do not expect 10 percent.

**On the 433k to 672k figure for MiniMax-M3 on H200 (up 55 percent):** this is the release note headline, but it comes from a different GPU and model with no breakdown. The plausible reading is that the old estimate was far more conservative for that configuration than for the GB200 sample. Treat it as **evidence that the ceiling exists**, not as an expectation.

## 5. The cost: capacity and latency are one knob

### 5.1 The mechanism

A larger cache means the scheduler no longer has to preempt and recompute evicted sequences to stay within memory. It converts the extra capacity into **more concurrently running sequences**. Decode memory traffic per step is the sum of context lengths across the batch, and KV reads dominate. Bigger batch, longer per-token time.

This is the textbook capacity-latency trade: **you did not get free memory, you converted memory into concurrency**. Throughput and TTFT benefit, TPOT pays.

The reported figure is **6 to 18 percent higher TPOT at concurrency 128 on some H200 models**.

### 5.2 A caveat worth flagging

By the mechanism above, TPOT should rise mainly when concurrency is pushed up. But the measurement holds concurrency fixed at 128 and still shows the increase. With `max_num_seqs` saturated, the batch cannot grow, so something else is involved. Plausible explanations: fewer preemptions make the decode batch more homogeneous, and a pure 128-way decode batch is the most bandwidth-hungry shape there is; or a larger block pool spreads KV across more physical chunks and costs attention kernel efficiency.

The PR reports the effect without decomposing it. This paragraph is inference. **Operationally it does not matter. Measure your own P99.**

### 5.3 Who should turn it off

| Situation | Recommendation |
|---|---|
| Offline batch, throughput first, embedding and rerank | Keep the default, delete `--gpu-memory-utilization` |
| Interactive chat or voice, TPOT sensitive | Measure P50 and P99 first; set 0.9 if it misses |
| Multiple engines per GPU, or other processes on the card | Must set explicitly. This is the parameter main remaining job |
| Unified memory systems such as DGX Spark | Must leave room for the OS and CPU workloads |
| Memory dashboard alarms | Ignore. The engine deliberately fills available memory with cache. Near 100 percent is by design |

## 6. Compatibility, and the correct way to roll back

Configurations that cannot use the feature fall back to estimate-based sizing automatically, without error. Only an **explicit** `--enable-extensible-kv-cache` is an error in those branches:

| Configuration | Behavior |
|---|---|
| Platforms other than CUDA and ROCm (XPU, CPU) | Off. No VMM primitive |
| V1 model runner | Off. V1 is removed, so moot |
| `kv_cache_memory_bytes` or `num_gpu_blocks_override` | Off. You gave a size, so measurement is moot |
| HiSparse | Off. It allocates its own KV cache and must route through the hook |
| Elastic EP | Off. Scale-up rank launches skip warmup |
| LMCache MP connector | Off. It shares through legacy CUDA IPC and cannot export VMM memory |
| Mooncake NVLink or BAREX (`custom_mem_pool`) | Off. The cache must come from the connector pool |
| ROCm with NIXL, Mooncake or MoRIIO | Off. UCX and ROCr cannot yet register HIP VMM memory |
| Models with no KV cache | Off. Nothing to size |

<div class="fig">
<svg viewBox="0 0 680 106" width="100%" font-size="12" role="img" aria-label="Explicit utilization is a cap, not a rollback">
<title>Figure 3: 0.9 is a cap, not a rollback</title>
<rect x="34" y="14" width="612" height="34" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="1"/>
<text x="48" y="30" font-size="11" fill="#b91c1c" font-weight="600">--gpu-memory-utilization 0.9 is NOT a rollback</text>
<text x="48" y="43" font-size="10" fill="#7f1d1d">It is only a cap. Measured graph and warmup memory is still subtracted from that 90 percent, yielding 0.6 to 3.7 percent less cache than before.</text>
<rect x="34" y="56" width="612" height="34" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="1"/>
<text x="48" y="72" font-size="11" fill="#047857" font-weight="600">--no-enable-extensible-kv-cache is the rollback</text>
<text x="48" y="85" font-size="10" fill="#065f46">Returns to profiling plus the 8 percent slack. The utilization default stays at 0.92 in that path.</text>
</svg>
<div class="cap">Figure 3: Two switches at different layers. One caps the budget, the other selects the sizing algorithm.</div>
</div>

If you set an explicit value between 0.8 and 1.0 that withholds more than 1 GiB, vLLM prints a startup warning naming exactly how much memory is being kept out of the cache.

A few side effects worth knowing:

- **The memory dashboard sits near 100 percent.** Not a leak.
- **Weight hot updates** in RL loops need headroom from `gpu_memory_utilization` when the reload happens without a cache-clearing pause. A direct `reload_weights` call fails before loading if even the largest layer cannot fit.
- **KV connector registration timing changed.** The worker-side connector is created after the final extend and registers the final memory. Custom connectors should confirm they are not sizing against the reserved capacity.

## 7. Pre-flight checklist

```text
[ ] Image version at or above 0.34 (contains #56492)
[ ] Legacy --gpu-memory-utilization 0.9 or 0.92 removed (throughput deployments)
[ ] --kv-cache-memory-bytes is not set (it disables the whole feature)
[ ] Not using Mooncake NVLink or BAREX, LMCache MP, HiSparse, or Elastic EP
[ ] Baseline recorded from the startup log: block count and engine init time
[ ] Interactive services: P50 and P99 TPOT benchmarked against the old version
[ ] Memory alerts retuned, accepting a long-term figure near 100 percent
[ ] The rollback plan says --no-enable-extensible-kv-cache, not a utilization value
```

## 8. How this connects to the rest of the site

- **`sys9` (The KV Cache Landscape)** gives the static view: how big the cache is and the five tunable knobs. This post adds the **dynamic** view: at what point in startup the knob is turned, and why it used to be turned blindly.
- **`sys10` (KV Cache in Production)** covers vLLM paged block management and the cross-layer layout. Extensible KV cache does not change paging semantics. It changes **where the block count comes from**. The blocks are the same blocks; `num_gpu_blocks` moved from guessed to measured.
- **`fw7` (CUDA Graph)** is required reading for the premise here: a graph stores address snapshots, which is why any reallocation scheme fails. It is also why the core of this technique is stable addresses rather than faster allocation.
- **`sys12` (Mooncake)** and **`sys11` (HiCache L2)** both move KV across layers or machines. This feature is **orthogonal but in conflict**. It governs capacity inside HBM, they govern placement outside it. Mooncake `custom_mem_pool` and LMCache MP IPC export are both incompatible with VMM memory, hence the fallbacks.
- **`sys6` (HiSparse)** is the opposite answer to the same shortage: treat HBM as a cache and layer the KV. HiSparse **accepts that the working set does not fit and trades layers for capacity**. Extensible cache **first measures what actually fits**. In production the two could stack, starting from a correct measurement and layering on top, but on the current code path HiSparse turns Extensible off.
- **`sys13` (MoE Expert Parallelism)** covers compute-side memory and bandwidth through Grouped GEMM and EPLB. This post covers the cache side. During decode they compete for the same HBM bandwidth, which is why TPOT on large MoE models is more sensitive to cache capacity changes.

---

## Series navigation

**Inference Systems Infrastructure Notes**

- [(13) MoE Expert Parallelism, Communication and Compute](/en/blog/sys13-moe-expert-parallel-comms)
- [(12) Mooncake: A KVCache-centric PD-Disaggregated Architecture](/en/blog/sys12-mooncake-kvcache-centric-serving)
- [(11) Tuning SGLang HiCache L2 on Kunlunxin P800/P900](/en/blog/sys11-hicache-l2-kunlunxin-tuning)
- [(10) KV Cache in Production](/en/blog/sys10-kv-cache-engineering-sglang-vllm-attentionstore)
- [(9) The KV Cache Landscape](/en/blog/sys9-kv-cache-fundamentals-optimization)

**Related**

- [(7) CUDA Graph](/en/blog/fw7-cuda-graph): where the address snapshot lives
- [(6) HiSparse Hierarchical KV Cache](/en/blog/sys6-hisparse-hierarchical-kv-cache): the opposite direction
- [(1) MLA operator](/en/blog/op1-mla-operator): where the context up-projection memory comes from
