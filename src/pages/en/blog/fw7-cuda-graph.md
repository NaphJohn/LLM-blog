---
title: '(7) Graph Mode: CUDA Graphs and How vLLM / SGLang Implement Them'
description: "Why decode is CPU-bound, the capture/replay mechanics and three hard constraints of CUDA Graphs, plus vLLM's piecewise+full dual-mode dispatch and SGLang's CudaGraphRunner with default-on PCG."
pubDate: 2026-08-04
series: vLLM & SGLang Serving Notes
lang: en
altLang: zh
altHref: /blog/fw7-cuda-graph
layout: ../../../layouts/BlogPost.astro
---

## 1. Why Graph Mode: Decode Is CPU-Bound

Previous posts covered paged KV cache, prefix reuse, and continuous batching — all optimizing **memory and FLOPs**. Once those are fixed, engines hit a third wall: **CPU kernel-launch overhead**.

Decode has a peculiar shape: each step computes `batch × 1` tokens. The matrices are flat and small, so each kernel runs on the GPU for only **a few microseconds**, while the CPU-side cost of launching a kernel — parameter marshalling, driver calls, stream submission — is **5–10 μs**. How many kernels per forward pass? A 60-layer model with RMSNorm, QKV projection, RoPE, attention, residual adds, and three or four GEMMs per layer adds up to **hundreds to over a thousand**.

Do the math: 1,000 kernels × 7 μs launch overhead ≈ **7 ms of CPU time**, while the GPU might finish the same kernels in 2–3 ms. The counterintuitive conclusion: **decode is bottlenecked not by the GPU but by how fast the CPU can feed it**. The GPU spends most of its time idle, and the utilization curve looks like a sawtooth full of "bubbles."

<div class="fig">
<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#374151">Eager: CPU launches one by one, GPU idles</text>
  <text x="16" y="52" font-size="11" fill="#6b7280">CPU</text>
  <g fill="#dbeafe" stroke="#3b82f6">
    <rect x="52" y="38" width="30" height="18" rx="3"/><rect x="102" y="38" width="30" height="18" rx="3"/><rect x="152" y="38" width="30" height="18" rx="3"/><rect x="202" y="38" width="30" height="18" rx="3"/><rect x="252" y="38" width="30" height="18" rx="3"/><rect x="302" y="38" width="30" height="18" rx="3"/><rect x="352" y="38" width="30" height="18" rx="3"/><rect x="402" y="38" width="30" height="18" rx="3"/><rect x="452" y="38" width="30" height="18" rx="3"/><rect x="502" y="38" width="30" height="18" rx="3"/><rect x="552" y="38" width="30" height="18" rx="3"/><rect x="602" y="38" width="30" height="18" rx="3"/>
  </g>
  <text x="16" y="90" font-size="11" fill="#6b7280">GPU</text>
  <rect x="52" y="74" width="580" height="18" rx="3" fill="none" stroke="#d1d5db" stroke-dasharray="4 3"/>
  <g fill="#a7f3d0" stroke="#10b981">
    <rect x="52" y="74" width="12" height="18" rx="2"/><rect x="102" y="74" width="12" height="18" rx="2"/><rect x="152" y="74" width="12" height="18" rx="2"/><rect x="202" y="74" width="12" height="18" rx="2"/><rect x="252" y="74" width="12" height="18" rx="2"/><rect x="302" y="74" width="12" height="18" rx="2"/><rect x="352" y="74" width="12" height="18" rx="2"/><rect x="402" y="74" width="12" height="18" rx="2"/><rect x="452" y="74" width="12" height="18" rx="2"/><rect x="502" y="74" width="12" height="18" rx="2"/><rect x="552" y="74" width="12" height="18" rx="2"/><rect x="602" y="74" width="12" height="18" rx="2"/>
  </g>
  <text x="52" y="112" font-size="10" fill="#9ca3af">dashed = GPU bubbles: kernels run for μs, launches cost 5–10 μs — the GPU waits on the CPU</text>

  <text x="16" y="148" font-size="12" font-weight="700" fill="#374151">Graph mode: one launch replays the whole graph</text>
  <text x="16" y="182" font-size="11" fill="#6b7280">CPU</text>
  <rect x="52" y="166" width="46" height="18" rx="3" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="75" y="179" font-size="10" fill="#1e3a8a" text-anchor="middle">launch</text>
  <text x="112" y="179" font-size="10" fill="#9ca3af">← CPU then idles, free to schedule the next batch</text>
  <text x="16" y="220" font-size="11" fill="#6b7280">GPU</text>
  <rect x="52" y="204" width="580" height="18" rx="3" fill="#a7f3d0" stroke="#10b981"/>
  <text x="342" y="217" font-size="10" fill="#065f46" text-anchor="middle">all kernels in the graph run back-to-back, no bubbles</text>
</svg>
</div>

## 2. CUDA Graphs: Capture Once, Replay Forever

CUDA Graphs (introduced in CUDA 10) take a direct approach:

- **Capture**: record a sequence of kernels — together with each kernel's parameters and memory addresses — into a DAG object;
- **Replay**: for every subsequent execution, the CPU submits the entire graph with **a single launch**, and the GPU driver schedules the kernels back-to-back. CPU overhead drops from O(#kernels) to **O(1)**;
- Graphs can be partially updated (`cudaGraphExecUpdate`), but inference engines rarely bother — they swap data through static buffers instead.

There is no free lunch. Graph mode imposes three hard constraints:

1. **Fixed shapes**: grid/block dimensions and tensor shapes are frozen at capture time;
2. **Fixed memory addresses**: replay uses the pointers recorded during capture, so inputs/outputs must live in fixed-address buffers;
3. **No CPU↔GPU sync, no dynamic control flow**: a graph contains no `if batch > 32`, and you cannot `.item()` a value back to the host mid-graph.

So every inference engine converges on the same engineering recipe:

| Recipe | Constraint it solves |
|---|---|
| **Static input/output buffers** | Fixed addresses: copy real inputs in before replay, read results out after |
| **Capture a set of graphs at discrete sizes** | Fixed shapes: one graph each for bs = 1, 2, 4, 8, … |
| **Pad up at runtime** | Actual bs = 100 → replay the bs = 128 graph, discard 28 extra rows |
| **Fall back to eager beyond the largest size** | You can't capture infinitely many graphs, and large batches are GPU-bound anyway |

Padding sounds wasteful, but decode kernels are memory-bound — computing a few extra padded rows costs far less than CPU launch overhead. The trade is almost always worth it.

## 3. vLLM: Piecewise Graphs as the Base, Full Graphs to Close, Runtime Dispatch

vLLM V1 ties graph mode to torch.compile's piecewise compilation in a single unified framework.

### 3.1 PIECEWISE: Split at Attention

V1's default compilation level splits the forward FX graph **at every attention op** and captures CUDA graphs only for the **segments between attentions** (embedding, RMSNorm, QKV projection, MoE/MLP, residuals — all token-wise computation), while **attention itself runs eagerly**.

Why split there? Attention is the hardest op to make graph-compatible: variable sequence lengths, paged KV block-table indexing, varlen `cu_seqlens` — all need dedicated backend support. The computation between attentions, by contrast, is purely token-wise with shapes determined solely by token count — naturally graph-friendly. To enable fine-grained memory reuse across piecewise graphs, V1 also rewrote attention so that **the output tensor is passed in as an input**: attention writes into a graph-managed buffer instead of allocating its own output.

### 3.2 FULL: Attention Inside the Graph Too

For **perfectly uniform pure-decode batches** (every request generates exactly one token, no prefill mixed in), a graph-compatible attention backend (FlashAttention / FlashInfer decode kernels, which read `seq_lens` from a GPU buffer and resolve addresses inside the kernel) lets you capture **the entire forward pass, attention included**, eliminating even attention's launch overhead. The win is most visible for small models and MoE low-latency serving.

### 3.3 Five cudagraph_mode Levels, Dispatched at Runtime

vLLM composes these capabilities into five modes:

| Mode | Behavior | Best for |
|---|---|---|
| `NONE` | Fully eager | Debugging, graph-incompatible models |
| `PIECEWISE` | Piecewise graphs only | Pooling models, etc. |
| `FULL` | Full graphs only | Small models / short-prompt pure-decode workloads |
| `FULL_DECODE_ONLY` | Full graph for uniform decode, eager otherwise | Decode instances in P/D disaggregation; saves piecewise-graph memory |
| `FULL_AND_PIECEWISE` (default) | Full graph for uniform decode, piecewise otherwise | Strongest general setting; highest memory use |

The last two are **dual modes**: at runtime a **dispatcher picks per batch composition** — uniform decode batches take the full graph, mixed/prefill batches take piecewise, special cases like cascade attention are forced to PIECEWISE, and anything unsupported degrades to eager. If an attention backend doesn't support the configured mode, the framework automatically downgrades to the nearest supported one.

### 3.4 Capture Sizes

Which batch sizes to capture is controlled by `cudagraph_capture_sizes` (auto-generated by default, up to `max_num_seqs`):

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 48]}'
```

The tuning rule of thumb: **make sure your typical concurrency levels appear in the capture list**. One reported case gained 8% throughput just by adding bs=16 to the list, and another 25% by switching to `FULL_DECODE_ONLY`. The cost: each extra size means more GPU memory and longer capture time. V1 graphs use noticeably more memory than V0 — trim the list when memory is tight, or use `enforce_eager` to disable graphs entirely.

## 4. SGLang: Full Graphs for Decode + Piecewise Graphs for Prefill, Two Independent Paths

SGLang skips the "unified dispatcher" design and builds a dedicated runner for each phase.

### 4.1 Decode: Full-Graph Capture via CudaGraphRunner

SGLang's decode attention kernels are **natively graph-compatible** — dynamic metadata like sequence lengths and KV write locations all live in static on-GPU buffers, and kernels resolve addresses internally after launch, with zero CPU-side dynamic decisions. So SGLang captures **the whole decode forward as a single graph**, with no "leave attention outside" compromise.

Key components:

- **GraphInputBuffers**: pre-allocated static input buffers — `input_ids`, `positions`, `seq_lens`, `out_cache_loc` (KV cache write positions) — with permanent addresses;
- **get_batch_sizes_to_capture()**: builds the capture list from 1 to `--cuda-graph-max-bs` (dense at small bs, sparse at large bs); `--cuda-graph-bs` overrides it explicitly;
- **Largest-first capture + a global shared memory pool**: the biggest graph is captured first, and smaller graphs reuse its pool memory, substantially cutting total footprint;
- **Three steps at runtime**: (1) `bisect` for the smallest captured size ≥ actual bs; (2) copy the real batch into static buffers and pad the rest; (3) `graph.replay()`, then read results from the static output buffer.

If the actual bs exceeds `cuda_graph_max_bs`, or the batch uses graph-incompatible features (some speculative-decoding paths), the runner falls back to a normal eager forward.

### 4.2 Prefill / Extend: PiecewiseCudaGraphRunner (PCG)

Prefill's difficulty is that **token counts vary every step**, so you can't enumerate shapes by batch size the way decode does. SGLang's PCG (Piecewise CUDA Graph — **enabled by default in current releases**; the old `--enable-piecewise-cuda-graph` flag is deprecated) takes the same idea as vLLM's piecewise graphs but implements it differently:

1. **Trace**: use `torch.compile` with a custom `SGLangBackend` to obtain the FX graph of the model forward;
2. **Split**: cut the graph at registered **split ops** (attention, all-reduce, MoE dispatch — dynamic or communication ops), which stay eager;
3. **Capture each piece**: every segment is captured at a set of **token-count levels**. The default schedule gets sparser as sizes grow — step 4 for 4–32, 16 for 48–256, 32 for 288–512, 64 for 576–1024, 256 for 1280–4096, 512 beyond — capped by `--piecewise-cuda-graph-max-tokens` (defaults to `chunked_prefill_size` for non-MLA, 2048 for MLA);
4. **Runtime**: the actual token count is **rounded up via binary search** to the nearest captured level, copied into static buffers with zero padding, each piece is replayed, and outputs are sliced back to the true length. Beyond the largest level, it falls back to the normal path.

For memory, PCG likewise uses a **global shared pool + largest-first capture**, and holds the last segment's output tensors as weak references to maximize reuse.

One detail that matters to kernel developers: PCG relies on torch.compile tracing, and **newly written CUDA kernels are not traceable by default** (JIT compilation, file IO, dynamic loading all break tracing). They must be wrapped with `register_custom_op` as opaque nodes to live inside a piecewise graph.

### 4.3 Stacking with torch.compile

SGLang also has an independent `--enable-torch-compile` flag (default for small bs ≤ 32): inside the CUDA graph, Inductor fuses kernels further so each graph contains fewer, larger kernels. The two compose — they are not alternatives.

## 5. Side-by-Side

| Dimension | vLLM V1 | SGLang |
|---|---|---|
| Decode graph granularity | Dual mode: full graph for uniform decode, piecewise otherwise, dispatched per batch | CudaGraphRunner always full-graph (decode kernels are natively graph-compatible) |
| Prefill graphs | Piecewise (PIECEWISE, unified with torch.compile piecewise compilation) | PCG piecewise graphs (separate runner, on by default) |
| Split points | Attention ops | Split ops: attention / all-reduce / MoE dispatch |
| Capture dimensions | Batch-size list `cudagraph_capture_sizes` | Decode by bs (`--cuda-graph-max-bs`); prefill by token-count levels |
| Memory strategy | Compiler backend manages intermediate buffers; footprint is on the high side, trim the size list | Global shared pool + largest-first capture + weak-ref outputs |
| Fallback | Automatic mode downgrade (FULL → PIECEWISE → NONE) | Over max level / incompatible → eager |
| Turn it off | `enforce_eager=True` or `cudagraph_mode: NONE` | `--disable-cuda-graph` / `--disable-piecewise-cuda-graph` |

The design-philosophy difference is worth savoring: vLLM chose **one framework with runtime dispatch** — maximally expressive, using whatever graph fits the batch, at the cost of complexity and the highest memory footprint. SGLang chose **two independent, individually optimized paths** — a dead-simple full graph for decode and PCG for prefill; less elegant, but each piece is easier to maintain. On the fundamentals — padding, static buffers, shared memory pools — the two are identical.

## 6. Engineering Trade-offs and Common Pitfalls

1. **Memory for CPU**: graph mode is a textbook memory-for-overhead trade. Community figures for SGLang: ~100–500 MB of input buffers per captured size + 10–50 MB of graph metadata + a 1–2 GB one-time shared pool; vLLM V1 graphs use more memory than V0. When you OOM, trimming capture sizes should be your first move.
2. **Capture time lands in startup**: dozens of sizes × (one warmup + one capture) each can add tens of seconds to minutes on large models. Pre-warm and freeze production images.
3. **Padding isn't free**: padding bs=100 to 128 wastes 28% of decode compute — usually still far cheaper than launch overhead. The real trap is a capture list so sparse that you permanently pad 2×.
4. **Disable graphs when debugging**: graphs obscure real kernel call stacks and make errors cryptic. For numerical or crash investigations, `enforce_eager` / `--disable-cuda-graph` should be your first reaction.
5. **Interaction with speculative decoding**: draft-verify makes batch shapes more complex; both engines special-case (or restrict) graph compatibility on speculative paths. When speculative decoding is on, verify the graphs are actually being hit.

## 7. Summary

- Graph mode solves decode's **CPU launch bound**: capture once, replay forever — CPU overhead goes from O(n) to O(1);
- Three hard constraints (fixed shapes/addresses, no sync) imply the universal recipe: **static buffers + per-size capture + pad up**;
- **vLLM**: piecewise graphs unified with torch.compile, with FULL / PIECEWISE dual modes dispatched per batch;
- **SGLang**: full-graph CudaGraphRunner for decode, default-on PCG piecewise graphs for prefill — two independent paths;
- The cost is memory and startup time — graph mode is the quintessential "trade resources for deterministic latency" design in inference engines.

Next post we return to the scheduling layer: how Prefill/Decode Disaggregation splits two phases with completely different load profiles onto different GPUs, and how it stacks with graph mode and continuous batching.

## Further Reading

- [vLLM docs: CUDA Graphs design](https://docs.vllm.ai/en/latest/design/cuda_graphs.html) and [torch.compile integration](https://docs.vllm.ai/en/latest/design/torch_compile.html)
- [SGLang docs: Piecewise CUDA Graph](https://sgl-project.github.io/advanced_features/piecewise_cuda_graph.html)
- SGLang source: `python/sglang/srt/model_executor/cuda_graph_runner.py`, `piecewise_cuda_graph_runner.py`
- [NVIDIA CUDA Graphs programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
