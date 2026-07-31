---
title: '(1) Why You Need vLLM / SGLang: What Naive Inference Gets Wrong'
description: 'Starting from a plain HF generate call, this post explains the three bottlenecks — memory fragmentation, static batching, GPU stalls — and why inference engines exist.'
pubDate: 2026-08-01
series: vLLM & SGLang Serving Notes
lang: en
altLang: zh
altHref: /blog/fw1-why-inference-engine
layout: ../../../layouts/BlogPost.astro
---

## 0. What This Series Covers

This series reorganizes months of daily tracking of the vLLM / SGLang communities into one progressive storyline:

1. **Why these two frameworks exist** (this post)
2. **vLLM internals**: PagedAttention, continuous batching, the V1 architecture, Model Runner V2
3. **SGLang internals**: RadixAttention, structured output, zero-overhead Spec V2
4. **Shared frontier battles**: speculative decoding, killing sync stalls, PD disaggregation, low-bit quantization
5. **Version timeline**: what the 0.25 / 0.5.15 generation change actually did
6. **Model support and selection**: which engine for which workload

No prior reading required, but every post grounds the concrete facts from those daily reports (version numbers, PR IDs, performance figures) into the right conceptual slot.

## 1. Code That Runs But Cannot Serve

Almost everyone's first LLM inference code looks like this:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="cuda")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

inputs = tok("Explain speculative decoding", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=512)
print(tok.decode(out[0]))
```

This is **functionally correct**. One user, one request — it gives you the right answer.

But wrap it in an HTTP service and let 50 people hit it at once, and it collapses visibly: OOM, latency spiking into tens of seconds, and GPU utilization stuck somewhere in the teens.

The problem is not the model. It is **how you serve it**.

## 2. Three Fatal Bottlenecks

### 2.1 KV Cache Fragmentation

During autoregressive generation, every new token attends to the Key/Value vectors of all previous tokens. Caching them is the **KV cache**.

How big? Roughly:

```
KV memory = 2(K,V) x layers x kv_heads x head_dim x seq_len x batch x dtype_bytes
```

For a 32-layer, 8-KV-head (GQA), head-dim-128, FP16 8B model, that is about **128 KB per token**. An 8K-context request costs ~1 GB. Forty concurrent requests need 40 GB of KV alone.

The naive implementation **preallocates a contiguous block sized by `max_new_tokens`**. Set it to 2048 and it reserves 2048 tokens of space — even if the request stops after 30 tokens.

The result is memory full of reserved-but-unused holes. In practice, naive serving achieves only **20%–40% effective KV utilization**. Over 60% is pure waste.

<div class="fig">
<svg viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <text x="16" y="20" font-size="13" font-weight="700" fill="#b91c1c">Naive: contiguous preallocation</text>
  <rect x="16" y="32" width="290" height="30" rx="4" fill="#fef2f2" stroke="#ef4444"/>
  <rect x="16" y="32" width="70" height="30" rx="4" fill="#fca5a5" stroke="#ef4444"/>
  <text x="30" y="52" font-size="11" fill="#7f1d1d">used 30 tok</text>
  <text x="150" y="52" font-size="11" fill="#b91c1c">wasted (2048 reserved)</text>

  <rect x="16" y="70" width="290" height="30" rx="4" fill="#fef2f2" stroke="#ef4444"/>
  <rect x="16" y="70" width="120" height="30" rx="4" fill="#fca5a5" stroke="#ef4444"/>
  <text x="34" y="90" font-size="11" fill="#7f1d1d">used 400 tok</text>
  <text x="180" y="90" font-size="11" fill="#b91c1c">wasted</text>

  <rect x="16" y="108" width="290" height="30" rx="4" fill="#f3f4f6" stroke="#9ca3af" stroke-dasharray="4 3"/>
  <text x="80" y="128" font-size="11" fill="#6b7280">Request 3: out of memory, queued</text>
  <text x="16" y="162" font-size="12" fill="#b91c1c">Effective utilization ~20%-40%</text>

  <line x1="336" y1="20" x2="336" y2="200" stroke="#e5e7eb" stroke-width="1"/>

  <text x="360" y="20" font-size="13" font-weight="700" fill="#047857">Paged: on-demand blocks</text>
  <g fill="#a7f3d0" stroke="#10b981">
    <rect x="360" y="32" width="28" height="30" rx="3"/>
    <rect x="392" y="32" width="28" height="30" rx="3"/>
  </g>
  <text x="430" y="52" font-size="11" fill="#047857">Req A: 2 blocks</text>

  <g fill="#a7f3d0" stroke="#10b981">
    <rect x="360" y="70" width="28" height="30" rx="3"/>
    <rect x="392" y="70" width="28" height="30" rx="3"/>
    <rect x="424" y="70" width="28" height="30" rx="3"/>
    <rect x="456" y="70" width="28" height="30" rx="3"/>
    <rect x="488" y="70" width="28" height="30" rx="3"/>
  </g>
  <text x="526" y="90" font-size="11" fill="#047857">Req B: 5 blocks</text>

  <g fill="#bfdbfe" stroke="#3b82f6">
    <rect x="360" y="108" width="28" height="30" rx="3"/>
    <rect x="392" y="108" width="28" height="30" rx="3"/>
    <rect x="424" y="108" width="28" height="30" rx="3"/>
  </g>
  <text x="466" y="128" font-size="11" fill="#1d4ed8">Req C: admitted now</text>
  <text x="360" y="162" font-size="12" fill="#047857">Utilization &gt; 90%, several times more concurrency</text>

  <text x="16" y="196" font-size="11" fill="#6b7280">Fig 1: contiguous reservation vs paged allocation. Each small square is a fixed-size block (e.g. 16 tokens).</text>
</svg>
</div>

### 2.2 Static Batching: Head-of-Line Blocking

Naive services usually do **static batching**: collect N requests, run them together, **wait for the longest one**, return the whole batch, repeat.

Some requests finish in 20 tokens, others need 2000. The short ones **do not free their slots** — those slots idle until the longest finishes. And new arrivals must wait for the entire batch to complete.

### 2.3 GPU Waiting on CPU: The Invisible Killer

Within one decode step, the GPU may spend only a few hundred microseconds on matmuls, while the CPU schedules the next batch, prepares tensors, copies metadata, decides sampling results, checks stop conditions — often requiring device-to-host (D2H) and host-to-device (H2D) copies.

At every such synchronization point, the GPU simply **waits**. With a few hundred microseconds of compute and comparable launch plus sync overhead, **roughly half the wall time is idle**.

<div class="keybox">
<strong>In one line:</strong> naive inference wastes memory holes, idle slots, and idle GPU cycles. Inference engines exist to fill all three.
</div>

## 3. What Each Engine Solves

<div class="fig">
<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <rect x="20" y="16" width="200" height="200" rx="10" fill="#fef2f2" stroke="#fca5a5"/>
  <text x="120" y="40" font-size="14" font-weight="700" fill="#b91c1c" text-anchor="middle">Naive HF generate</text>
  <text x="36" y="68" font-size="12" fill="#7f1d1d">x Contiguous KV, fragmented</text>
  <text x="36" y="94" font-size="12" fill="#7f1d1d">x Static batch, HOL blocking</text>
  <text x="36" y="120" font-size="12" fill="#7f1d1d">x Per-step sync, GPU idle</text>
  <text x="36" y="146" font-size="12" fill="#7f1d1d">x No prefix reuse</text>
  <text x="36" y="172" font-size="12" fill="#7f1d1d">x No quant / parallelism</text>
  <text x="120" y="200" font-size="12" font-weight="700" fill="#b91c1c" text-anchor="middle">~15% utilization</text>

  <path d="M232 116 L268 116" stroke="#6b7280" stroke-width="2" marker-end="url(#ar1)"/>
  <defs><marker id="ar1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6b7280"/></marker></defs>

  <rect x="280" y="16" width="180" height="200" rx="10" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="370" y="40" font-size="14" font-weight="700" fill="#1d4ed8" text-anchor="middle">vLLM</text>
  <text x="294" y="68" font-size="12" fill="#1e3a8a">+ Paged KV / block table</text>
  <text x="294" y="94" font-size="12" fill="#1e3a8a">+ Continuous batching</text>
  <text x="294" y="120" font-size="12" fill="#1e3a8a">+ MRv2 zero sync</text>
  <text x="294" y="146" font-size="12" fill="#1e3a8a">+ Widest HW / model reach</text>
  <text x="294" y="172" font-size="12" fill="#1e3a8a">+ Full-step CUDA Graph</text>
  <text x="370" y="200" font-size="12" font-weight="700" fill="#1d4ed8" text-anchor="middle">General production base</text>

  <rect x="476" y="16" width="184" height="200" rx="10" fill="#ecfdf5" stroke="#6ee7b7"/>
  <text x="568" y="40" font-size="14" font-weight="700" fill="#047857" text-anchor="middle">SGLang</text>
  <text x="490" y="68" font-size="12" fill="#065f46">+ RadixAttention prefix tree</text>
  <text x="490" y="94" font-size="12" fill="#065f46">+ Fast structured output</text>
  <text x="490" y="120" font-size="12" fill="#065f46">+ Zero-overhead Spec V2</text>
  <text x="490" y="146" font-size="12" fill="#065f46">+ Large-scale EP / PD</text>
  <text x="490" y="172" font-size="12" fill="#065f46">+ Fast frontier adoption</text>
  <text x="568" y="200" font-size="12" font-weight="700" fill="#047857" text-anchor="middle">High-sharing / agent loads</text>

  <text x="20" y="238" font-size="11" fill="#6b7280">Fig 2: vLLM optimizes for breadth (models and hardware); SGLang for depth (prefix sharing and frontier throughput).</text>
</svg>
</div>

**vLLM** starts from memory: manage the KV cache like OS virtual memory (paging plus a block table), combine it with continuous batching, and push throughput up. It then grew into a general production base — the widest model coverage, the most hardware backends (CUDA / ROCm / XPU / TPU), the most quantization formats.

**SGLang** starts from program structure. It observed that real LLM applications — agents, multi-turn chat, batch evaluation, tree-of-thought — have **many requests sharing identical prefixes**. So it organizes prefix KV into a radix tree for cross-request reuse: RadixAttention. It also pushes aggressively on structured output and frontier throughput work.

## 4. The Metrics That Matter

Before choosing anything, be clear about the three numbers that matter:

| Metric | Meaning | Who cares |
|---|---|---|
| **TTFT** (Time To First Token) | Time to the first output token, dominated by prefill | Chat feel, agent first response |
| **TPOT / ITL** | Interval between subsequent tokens, dominated by decode | Streaming smoothness |
| **Throughput** | Total tokens per second across the server | Cost per million tokens |

These **trade off against each other**. Larger batches raise throughput and worsen per-request TPOT; PD disaggregation cuts TTFT but adds KV transfer. Every design decision picks a point inside this triangle.

<div class="warnbox">
<strong>Common mistake:</strong> selecting an engine on aggregate throughput alone. Offline batch jobs should indeed maximize total throughput, but online agent workloads are gated by TTFT and P99 TPOT — which live in the small-to-medium batch regime, exactly where the 2026 generation change (vLLM MRv2 / SGLang Spec V2) pays off most.
</div>

## 5. Prefill vs Decode

These two words recur throughout the series. One request has two phases with **opposite** compute characteristics:

| | Prefill | Decode |
|---|---|---|
| Work | Process the whole prompt at once, compute all KV | Process one new token |
| Parallelism | High (thousands of tokens at once) | Very low (one at a time) |
| Bottleneck | **Compute-bound** | **Memory-bound** |
| Determines | TTFT | TPOT |

This difference explains every optimization that follows:

- **Why decode needs large batches?** It is memory-bound; weights fetched once serve many requests, so bigger batches amortize better.
- **Why speculative decoding?** Decode emits one token per forward while compute units idle — so verify a few extra candidates for free.
- **Why PD disaggregation?** Opposite characteristics interfere on the same GPU; a prefill burst stretches decode latency. Separate them and give each its optimal parallelism and hardware.

## 6. Summary

| Naive problem | Engine solution | Origin |
|---|---|---|
| KV fragmentation | Paged KV cache + block table | vLLM PagedAttention |
| Head-of-line blocking | Continuous batching (iteration-level scheduling) | Both |
| Recomputing identical prefixes | Radix-tree prefix reuse | SGLang RadixAttention |
| GPU waiting on CPU | Zero-sync model runner / full CUDA Graph | vLLM MRv2 / SGLang Spec V2 |
| Idle decode compute | Speculative decoding (MTP / EAGLE / DFlash / DSpark) | Both |
| P/D interference | Prefill-decode disaggregation | Dynamo / Mooncake / NIXL |

Next: inside vLLM — from the virtual-memory analogy of PagedAttention to the seemingly contradictory act of **removing PagedAttention** in 0.25.
