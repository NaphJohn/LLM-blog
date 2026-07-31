---
title: '(2) vLLM Internals: From PagedAttention to Removing PagedAttention'
description: 'The virtual-memory analogy behind paged KV cache, continuous batching, the V1 process architecture, and why 0.25 removed PagedAttention while making Model Runner V2 the default.'
pubDate: 2026-08-01
series: vLLM & SGLang Serving Notes
lang: en
altLang: zh
altHref: /blog/fw2-vllm-internals
layout: ../../../layouts/BlogPost.astro
---

## 1. PagedAttention: OS Virtual Memory for the KV Cache

Naive KV caching preallocates by maximum length and reaches only 20%–40% utilization. vLLM's first big idea borrows directly from operating systems.

### 1.1 The Analogy

| Operating system | vLLM |
|---|---|
| Process virtual address space | Logical KV sequence of a request |
| Physical page (4 KB) | KV block (typically 16 tokens) |
| Page table | **Block table** |
| Demand paging, copy-on-write | On-demand blocks, prefix sharing |

A request's KV is **logically contiguous** (token 0, 1, 2, ...) but **physically scattered** anywhere in GPU memory, with a block table doing the mapping.

<div class="fig">
<svg viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#374151">Logical view (what the request sees)</text>
  <g font-size="10" text-anchor="middle">
    <rect x="16" y="28" width="60" height="26" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="46" y="45" fill="#1e3a8a">tok 0-15</text>
    <rect x="80" y="28" width="60" height="26" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="110" y="45" fill="#1e3a8a">tok 16-31</text>
    <rect x="144" y="28" width="60" height="26" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="174" y="45" fill="#1e3a8a">tok 32-47</text>
    <rect x="208" y="28" width="60" height="26" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="238" y="45" fill="#1e3a8a">tok 48-63</text>
  </g>
  <text x="286" y="46" font-size="11" fill="#6b7280">Request A (64 tokens)</text>

  <rect x="16" y="76" width="252" height="40" rx="5" fill="#fef9c3" stroke="#eab308"/>
  <text x="142" y="94" font-size="12" font-weight="700" fill="#854d0e" text-anchor="middle">Block Table</text>
  <text x="142" y="110" font-size="10" fill="#854d0e" text-anchor="middle">[0]-&gt;#7  [1]-&gt;#2  [2]-&gt;#9  [3]-&gt;#4</text>

  <line x1="46" y1="56" x2="60" y2="74" stroke="#9ca3af" stroke-width="1"/>
  <line x1="110" y1="56" x2="110" y2="74" stroke="#9ca3af" stroke-width="1"/>
  <line x1="174" y1="56" x2="174" y2="74" stroke="#9ca3af" stroke-width="1"/>
  <line x1="238" y1="56" x2="224" y2="74" stroke="#9ca3af" stroke-width="1"/>

  <text x="16" y="146" font-size="12" font-weight="700" fill="#374151">Physical GPU memory (block pool, scattered)</text>
  <g font-size="10" text-anchor="middle">
    <rect x="16" y="156" width="46" height="28" rx="3" fill="#f3f4f6" stroke="#d1d5db"/><text x="39" y="174" fill="#9ca3af">#0</text>
    <rect x="66" y="156" width="46" height="28" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="89" y="174" fill="#065f46">#1 B</text>
    <rect x="116" y="156" width="46" height="28" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="139" y="174" fill="#1e3a8a">#2 A1</text>
    <rect x="166" y="156" width="46" height="28" rx="3" fill="#f3f4f6" stroke="#d1d5db"/><text x="189" y="174" fill="#9ca3af">#3</text>
    <rect x="216" y="156" width="46" height="28" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="239" y="174" fill="#1e3a8a">#4 A3</text>
    <rect x="266" y="156" width="46" height="28" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="289" y="174" fill="#065f46">#5 B</text>
    <rect x="316" y="156" width="46" height="28" rx="3" fill="#f3f4f6" stroke="#d1d5db"/><text x="339" y="174" fill="#9ca3af">#6</text>
    <rect x="366" y="156" width="46" height="28" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="389" y="174" fill="#1e3a8a">#7 A0</text>
    <rect x="416" y="156" width="46" height="28" rx="3" fill="#f3f4f6" stroke="#d1d5db"/><text x="439" y="174" fill="#9ca3af">#8</text>
    <rect x="466" y="156" width="46" height="28" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="489" y="174" fill="#1e3a8a">#9 A2</text>
    <rect x="516" y="156" width="46" height="28" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="539" y="174" fill="#065f46">#10 B</text>
    <rect x="566" y="156" width="46" height="28" rx="3" fill="#f3f4f6" stroke="#d1d5db"/><text x="589" y="174" fill="#9ca3af">#11</text>
  </g>

  <rect x="380" y="76" width="284" height="40" rx="5" fill="#ecfdf5" stroke="#10b981"/>
  <text x="522" y="94" font-size="12" font-weight="700" fill="#047857" text-anchor="middle">Shared prefix: copy-on-write</text>
  <text x="522" y="110" font-size="10" fill="#047857" text-anchor="middle">If A and B share a prompt prefix, both point to the same block</text>

  <text x="16" y="212" font-size="11" fill="#6b7280">Fig 1: logical-to-physical mapping. Waste is bounded by one partially filled block, and identical prefixes share physical blocks.</text>
  <text x="16" y="232" font-size="11" fill="#6b7280">Result: effective KV utilization rises from ~20-40% to &gt;90%, multiplying concurrency at the same memory budget.</text>
</svg>
</div>

### 1.2 Three Direct Wins

1. **Fragmentation nearly vanishes**: waste is capped by the unfilled part of the last block — at most 15 tokens, not thousands.
2. **Free prefix sharing**: system prompts, few-shot examples, conversation history — identical prefixes point to the same physical blocks with reference counting and copy-on-write.
3. **Cheap parallel sampling**: best-of-n from one prompt stores the prompt KV once.

## 2. Continuous Batching

Paging solves memory; head-of-line blocking still needs fixing.

vLLM schedules at **iteration level**, not batch level:

```
After every decode step:
  - finished requests return immediately and free their blocks
  - queued requests are admitted into the freed slots
  - if memory runs short, preempt (swap out or recompute a request's KV)
```

Batch composition **flows dynamically**, so the GPU always runs active work instead of waiting for the slowest member. With **chunked prefill**, long prompts are split into chunks interleaved with decode steps, so a long prompt no longer monopolizes a step.

## 3. The V1 Architecture

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <rect x="150" y="14" width="380" height="34" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="340" y="36" font-size="13" font-weight="700" fill="#1d4ed8" text-anchor="middle">API Server (OpenAI-compatible frontend process)</text>

  <path d="M340 50 L340 66" stroke="#6b7280" stroke-width="1.5" marker-end="url(#a2)"/>
  <defs><marker id="a2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6b7280"/></marker></defs>
  <text x="352" y="63" font-size="10" fill="#6b7280">ZMQ / IPC</text>

  <rect x="150" y="68" width="380" height="34" rx="6" fill="#f0fdf4" stroke="#86efac"/>
  <text x="340" y="90" font-size="13" font-weight="700" fill="#047857" text-anchor="middle">EngineCore (separate process, busy loop)</text>

  <rect x="60" y="116" width="180" height="60" rx="6" fill="#fefce8" stroke="#fde047"/>
  <text x="150" y="136" font-size="12" font-weight="700" fill="#854d0e" text-anchor="middle">Scheduler</text>
  <text x="150" y="154" font-size="10" fill="#854d0e" text-anchor="middle">iteration-level, preemption</text>
  <text x="150" y="168" font-size="10" fill="#854d0e" text-anchor="middle">chunked prefill, priority</text>

  <rect x="256" y="116" width="168" height="60" rx="6" fill="#fefce8" stroke="#fde047"/>
  <text x="340" y="136" font-size="12" font-weight="700" fill="#854d0e" text-anchor="middle">KVCacheManager</text>
  <text x="340" y="154" font-size="10" fill="#854d0e" text-anchor="middle">block pool, block table</text>
  <text x="340" y="168" font-size="10" fill="#854d0e" text-anchor="middle">prefix cache, offload tiers</text>

  <rect x="440" y="116" width="180" height="60" rx="6" fill="#fefce8" stroke="#fde047"/>
  <text x="530" y="136" font-size="12" font-weight="700" fill="#854d0e" text-anchor="middle">Structured Output</text>
  <text x="530" y="154" font-size="10" fill="#854d0e" text-anchor="middle">grammar / JSON schema</text>
  <text x="530" y="168" font-size="10" fill="#854d0e" text-anchor="middle">streaming parser engine</text>

  <path d="M340 178 L340 194" stroke="#6b7280" stroke-width="1.5" marker-end="url(#a2)"/>

  <rect x="150" y="196" width="380" height="42" rx="6" fill="#fdf2f8" stroke="#f9a8d4"/>
  <text x="340" y="214" font-size="13" font-weight="700" fill="#9d174d" text-anchor="middle">Model Runner V2 (MRv2)</text>
  <text x="340" y="230" font-size="10" fill="#9d174d" text-anchor="middle">async-first, zero CPU-GPU sync, full-step CUDA Graph capture</text>

  <rect x="60" y="252" width="140" height="32" rx="6" fill="#f3f4f6" stroke="#d1d5db"/>
  <text x="130" y="272" font-size="11" fill="#374151" text-anchor="middle">Attention backends</text>
  <rect x="212" y="252" width="140" height="32" rx="6" fill="#f3f4f6" stroke="#d1d5db"/>
  <text x="282" y="272" font-size="11" fill="#374151" text-anchor="middle">FlashAttn / FlashInfer</text>
  <rect x="364" y="252" width="120" height="32" rx="6" fill="#f3f4f6" stroke="#d1d5db"/>
  <text x="424" y="272" font-size="11" fill="#374151" text-anchor="middle">Quant kernels</text>
  <rect x="496" y="252" width="124" height="32" rx="6" fill="#f3f4f6" stroke="#d1d5db"/>
  <text x="558" y="272" font-size="11" fill="#374151" text-anchor="middle">TP / PP / EP</text>
</svg>
</div>

Key design: **the API server and EngineCore run in separate processes**. CPU-heavy frontend work (tokenization, HTTP parsing, JSON serialization) no longer blocks the GPU scheduling loop; the two sides talk over ZMQ. This is also the foundation that made PD disaggregation natural later.

## 4. The Turn: Why 0.25 Removed PagedAttention

**vLLM v0.25.0** (July 11–12, 2026) did something counterintuitive: it **removed PagedAttention** (PR #47361) while making **Model Runner V2 the default for all dense models** (#39337).

It looks like self-sabotage. It is actually **an abstraction expiring**.

### 4.1 Why It Could Go

PagedAttention was originally an intermediate layer, because attention kernels of that era only read **contiguous** KV, so vLLM needed logical-to-physical translation above them.

By 2026, **modern attention backends (FlashAttention 3/4, FlashInfer) natively read block tables and gather paged KV inside the kernel**. Paging had sunk into the kernel itself.

That made vLLM's PagedAttention layer **pure redundant overhead** — an extra Python-side dispatch, an extra reshape, unnecessary synchronization. Removing it keeps paging semantics intact (block tables and KVCacheManager remain) and straightens the execution path.

<div class="keybox">
<strong>Two different things:</strong> what was removed is the internal <code>PagedAttention</code> <strong>operator abstraction</strong>. The <strong>paged KV cache mechanism still exists</strong>, now handled directly by attention backend kernels. This is abstraction sinking, not feature regression.
</div>

### 4.2 MRv2: Recording a Whole Step as One Graph

The same release promoted Model Runner V2, targeting the third bottleneck: **the GPU waiting on the CPU**.

MRv2 is **async-first with zero CPU-GPU synchronization**:

- every "copy back to CPU and branch" point on the forward path is eliminated, handled on-GPU instead;
- with no host sync points, **the entire decode step (including speculative draft and verify) can be captured as one complete CUDA Graph**;
- step N and N+1 overlap — the CPU prepares a full step ahead.

The effect: per-token kernel launch overhead collapses from **~300 µs to ~5 µs**.

<div class="fig">
<svg viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#b91c1c">MRv1: a sync point every step</text>
  <rect x="16" y="26" width="70" height="24" rx="3" fill="#bfdbfe" stroke="#3b82f6"/><text x="51" y="42" font-size="10" fill="#1e3a8a" text-anchor="middle">CPU prep</text>
  <rect x="90" y="26" width="90" height="24" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="135" y="42" font-size="10" fill="#065f46" text-anchor="middle">GPU forward</text>
  <rect x="184" y="26" width="54" height="24" rx="3" fill="#fecaca" stroke="#ef4444"/><text x="211" y="42" font-size="10" fill="#7f1d1d" text-anchor="middle">D2H</text>
  <rect x="242" y="26" width="60" height="24" rx="3" fill="#fed7aa" stroke="#f97316"/><text x="272" y="42" font-size="10" fill="#7c2d12" text-anchor="middle">CPU branch</text>
  <rect x="306" y="26" width="54" height="24" rx="3" fill="#fecaca" stroke="#ef4444"/><text x="333" y="42" font-size="10" fill="#7f1d1d" text-anchor="middle">H2D</text>
  <rect x="364" y="26" width="70" height="24" rx="3" fill="#bfdbfe" stroke="#3b82f6"/><text x="399" y="42" font-size="10" fill="#1e3a8a" text-anchor="middle">CPU prep</text>
  <rect x="438" y="26" width="90" height="24" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="483" y="42" font-size="10" fill="#065f46" text-anchor="middle">GPU forward</text>
  <text x="16" y="66" font-size="10" fill="#b91c1c">GPU idle window ~ half of each step</text>
  <rect x="184" y="54" width="176" height="6" fill="#fca5a5"/>

  <line x1="16" y1="84" x2="664" y2="84" stroke="#e5e7eb"/>

  <text x="16" y="106" font-size="12" font-weight="700" fill="#047857">MRv2: whole step in one CUDA Graph, CPU runs a step ahead</text>
  <rect x="16" y="114" width="70" height="24" rx="3" fill="#bfdbfe" stroke="#3b82f6"/><text x="51" y="130" font-size="10" fill="#1e3a8a" text-anchor="middle">prep N</text>
  <rect x="90" y="114" width="70" height="24" rx="3" fill="#bfdbfe" stroke="#3b82f6"/><text x="125" y="130" font-size="10" fill="#1e3a8a" text-anchor="middle">prep N+1</text>
  <rect x="164" y="114" width="70" height="24" rx="3" fill="#bfdbfe" stroke="#3b82f6"/><text x="199" y="130" font-size="10" fill="#1e3a8a" text-anchor="middle">prep N+2</text>

  <rect x="90" y="144" width="112" height="24" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="146" y="160" font-size="10" fill="#065f46" text-anchor="middle">graph N</text>
  <rect x="206" y="144" width="112" height="24" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="262" y="160" font-size="10" fill="#065f46" text-anchor="middle">graph N+1</text>
  <rect x="322" y="144" width="112" height="24" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="378" y="160" font-size="10" fill="#065f46" text-anchor="middle">graph N+2</text>
  <text x="442" y="160" font-size="10" fill="#047857">no bubbles</text>

  <text x="16" y="188" font-size="11" fill="#6b7280">Fig 3: launch overhead 300 µs to 5 µs. Gains are largest at small and medium batch sizes.</text>
</svg>
</div>

### 4.3 Costs and Pitfalls

<div class="warnbox">
<strong>0.25 is a wide-reaching change; full regression testing is mandatory:</strong>
<ul>
<li>custom operators and older GPUs fall back to the MRv1 path and see no gain;</li>
<li><strong>Transformers v4 is deprecated</strong> (#40389) — migrate to v5;</li>
<li>the build now requires <strong>C++20</strong>;</li>
<li>legacy partial-prefill flags were removed (#49244).</li>
</ul>
</div>

**v0.25.1** (July 14) is a mandatory two-commit patch:

- **#48330 mixed-dtype quant fusion guard** — fixes NVFP4 models emitting **silent garbage**. Root cause: FlashInfer's fused `allreduce + RMSNorm + static-quant` kernel assumed matching dtypes; with BF16 activations and FP32 RMSNorm weights it read 4-bit NVFP4 with the wrong bit pattern, corrupting hidden states and producing repeated `!!!!!`. The fix adds a dtype sentinel: mismatched dtypes take a safe path, matched ones keep the fusion.
- **#47888** — TorchCodec no longer blocks startup when FFmpeg is missing.

<div class="keybox">
The lesson from #48330: <strong>fused kernels genuinely save HBM round trips, but the implicit "dtypes always match" assumption can break.</strong> A dtype sentinel buys both speed and correctness — the classic tradeoff for aggressive kernel fusion.
</div>

## 5. Where vLLM Differentiates

Across many release cycles, vLLM's real moat is **breadth**:

- **Model breadth**: `Transformers backend parity` (since 0.25 the Transformers backend matches native speed) means **any new architecture with an HF implementation can be served at full speed on day zero**.
- **Hardware breadth**: CUDA / ROCm (AITER) / Intel XPU (DeepSeek-V4 `fuse_index_q` SYCL path) / TPU / CPU — the widest of any engine.
- **Quantization breadth**: FP8 / INT4 / AWQ / GPTQ / NVFP4 / MXFP4 / compressed-tensors.
- **Ecosystem breadth**: the KV Connector interface (Mooncake / NIXL / LMCache), a Rust router, and the two-tier KV TieringManager (PR #42285).

## 6. Summary

| Mechanism | Solves | Status |
|---|---|---|
| Paged KV + block table | Fragmentation, prefix sharing | Retained; operator layer sunk into attention backends |
| Continuous batching + chunked prefill | Head-of-line blocking | Stable foundation |
| API/Engine process split (V1) | CPU frontend blocking the GPU loop | Stable; basis for PD disaggregation |
| **Model Runner V2** | GPU idling on the CPU | **Default for all dense models since 0.25** |
| Full CUDA Graph capture | Kernel launch overhead | 300 µs to 5 µs |
| Transformers backend parity | Time-to-serve for new models | Day-zero at full speed |

Next: SGLang, which starts from a completely different question — not "how do we manage memory" but "**what does an LLM program actually look like**".
