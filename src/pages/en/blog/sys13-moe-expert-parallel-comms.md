---
title: "Inference Systems Infrastructure Notes (13): MoE Expert Parallelism, Communication and Compute — How Dispatcher, DeepEP, Grouped GEMM and EPLB Form One Pipeline, and How MegaMoE Welds It Into a Single Kernel"
description: "One MoE forward pass in five steps: route, dispatch, Grouped GEMM, combine, weighted reduction. Why the hard part of dispatch is variable length rather than getting data out, why DeepEP ships normal and low-latency kernels for two different SLAs, why the real win of Grouped GEMM is collapsing E tails into one rather than saving kernel launches, and why EPLB can only rebalance by statistics plus moving weights at inference time. Includes a decode step latency ledger and a parameter decision table, plus MegaMoE fusion and wave pipelining: the three classes of cost that fusion removes, why wave scheduling only rearranges time, and how the 6144 FLOPs/Byte hiding threshold falls out of 6hd over 3h, with a clear separation from MagiMoE."
pubDate: 2026-11-17
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys13-moe-expert-parallel-comms
layout: ../../../layouts/BlogPost.astro
---

## 0. The one-line thesis

A MoE forward pass follows a fixed data flow: **route decides where tokens go, dispatch sends them there, Grouped GEMM computes the whole batch at once, combine pulls results back, and a weighted sum reassembles them by gate weight**.

These four components are usually introduced one at a time, but their value only appears when you look at them together: **dispatch and combine set the floor on per-step latency, Grouped GEMM decides how much compute that batch actually uses, and EPLB decides whether the pipeline drifts after tens of thousands of steps**. Study any one in isolation and you reach the wrong conclusion — for instance that Grouped GEMM is just a way to save kernel launches, or that EPLB is optional operations tooling.

This post stays out of kernel internals (TMA, warp specialization, swizzling belong in a dedicated article) and covers the **systems-level semantics** instead: what each stage solves, whose latency budget pays for it, and how to pick the parameters.

## 1. Putting three terms in the right place

### 1.1 How EP differs from TP: what actually moves

| Strategy | What is sliced | Primitive | What moves | Traffic depends on |
|---|---|---|---|---|
| **TP** tensor parallel | Inside weight matrices (hidden / head dim) | all-reduce / all-gather | Weights are sliced, **activations stay put** | Fixed (twice per layer, scales with batch) |
| **PP** pipeline parallel | Layers | point-to-point activations | Activations flow one way | Fixed (stage boundaries only) |
| **EP** expert parallel | Experts (whole blocks on different GPUs) | **all-to-all** | Experts sit still, **tokens move** | **top-k, batch, and routing skew** |

That last row is the root of every EP headache. TP traffic is **knowable in advance** — give it hidden size and batch size and you can compute the bytes. EP traffic is **only knowable at runtime**: with the same 4096 tokens, top-8, and 128 experts, a uniform router gives each GPU 256 tokens, while a skewed one gives some GPU 800 and another 30.

This is why EP needs a purpose-built communication library (DeepEP) while TP gets by with NCCL all-reduce: **NCCL solves fixed-size collective communication, whereas EP needs variable-length, sparse communication whose destinations are decided by data**.

### 1.2 The word dispatcher means two different things

The dispatcher in `fw7` on this site is the **CUDA Graph mode dispatcher** — at runtime vLLM decides from the batch composition whether this batch replays the FULL graph or the PIECEWISE one. Its job is choosing which graph to replay.

The dispatcher in this post is the **MoE token dispatcher** — it moves hidden states to the GPUs holding the experts that routing selected. Its job is moving data.

Same name, unrelated mechanisms, and the only thing they share is the word. Do not read a hit in `fw7` as coverage of MoE dispatch.

## 2. One MoE forward pass: five steps and two boundaries

<div class="fig">
<svg viewBox="0 0 680 300" width="100%" font-size="13" font-weight="400" role="img" aria-label="The five-step data flow of one MoE forward pass">
<title>Figure 1: one MoE layer forward pass in five steps, with two communication segments in between</title>
<defs>
<marker id="m1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<text x="340" y="24" font-size="12" fill="#1a1a1a" font-weight="600" text-anchor="middle">One MoE layer: input x (local tokens per GPU)</text>

<rect x="30" y="46" width="104" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="0.8"/>
<text x="82" y="68" font-size="11" fill="#1d4ed8" font-weight="600" text-anchor="middle">1 - Route</text>
<text x="82" y="86" font-size="10" fill="#1a1a1a" text-anchor="middle">gate to top-k</text>

<rect x="150" y="46" width="120" height="56" rx="6" fill="#fef2f2" stroke="#ef4444" stroke-width="1.2"/>
<text x="210" y="68" font-size="11" fill="#b91c1c" font-weight="600" text-anchor="middle">2 - Dispatch</text>
<text x="210" y="86" font-size="10" fill="#1a1a1a" text-anchor="middle">all-to-all (variable)</text>

<rect x="286" y="46" width="132" height="56" rx="6" fill="#ecfdf5" stroke="#10b981" stroke-width="0.8"/>
<text x="352" y="68" font-size="11" fill="#047857" font-weight="600" text-anchor="middle">3 - Grouped GEMM</text>
<text x="352" y="86" font-size="10" fill="#1a1a1a" text-anchor="middle">E experts in one go</text>

<rect x="434" y="46" width="120" height="56" rx="6" fill="#fef2f2" stroke="#ef4444" stroke-width="1.2"/>
<text x="494" y="68" font-size="11" fill="#b91c1c" font-weight="600" text-anchor="middle">4 - Combine</text>
<text x="494" y="86" font-size="10" fill="#1a1a1a" text-anchor="middle">all-to-all (reverse)</text>

<rect x="570" y="46" width="82" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="0.8"/>
<text x="611" y="68" font-size="11" fill="#1d4ed8" font-weight="600" text-anchor="middle">5 - Weight</text>
<text x="611" y="86" font-size="10" fill="#1a1a1a" text-anchor="middle">sum gate_i * y_i</text>

<line x1="134" y1="74" x2="148" y2="74" stroke="#6b7280" stroke-width="1" marker-end="url(#m1)"/>
<line x1="270" y1="74" x2="284" y2="74" stroke="#6b7280" stroke-width="1" marker-end="url(#m1)"/>
<line x1="418" y1="74" x2="432" y2="74" stroke="#6b7280" stroke-width="1" marker-end="url(#m1)"/>
<line x1="554" y1="74" x2="568" y2="74" stroke="#6b7280" stroke-width="1" marker-end="url(#m1)"/>

<rect x="150" y="112" width="404" height="24" rx="4" fill="#fff5f5" stroke="#ef4444" stroke-width="0.5" stroke-dasharray="3 3"/>
<text x="352" y="128" font-size="10" fill="#b91c1c" text-anchor="middle">Pure communication, hideable only with chunk-level pipelining (see section 6)</text>

<rect x="30" y="164" width="622" height="52" rx="6" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="46" y="182" font-size="11" fill="#1a1a1a" font-weight="600">Three things dispatch quietly does along the way (easy to miss)</text>
<text x="46" y="200" font-size="10" fill="#6b7280">1) permute tokens by expert 2) concatenate variable segments into one buffer 3) emit offsets - exactly the layout Grouped GEMM wants</text>

<rect x="30" y="232" width="622" height="52" rx="6" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="46" y="250" font-size="11" fill="#854f0b" font-weight="600">Key observation: the Grouped GEMM input is not a designed format, it is a dispatch byproduct</text>
<text x="46" y="268" font-size="10" fill="#854f0b">So design them together. Fix the GEMM layout first, then the packing format; the other order costs an extra permute pass</text>
</svg>
</div>

Steps one and five are pure local compute (the gate matmul and the weighted sum). Steps two through four are where the problems live. Two boundaries are worth remembering:

1. **Before dispatch, tokens are ordered by request; after dispatch, they are ordered by expert.** That reordering is not a side effect, it is the precondition for Grouped GEMM to work at all.
2. **After combine you must have every shard before entering the next layer.** So combine is exposed latency with no compute left to hide it behind (unless you can hoist independent work from the next layer).

## 3. Dispatch: the hard part is variable length, not getting data out

### 3.1 Why a plain all-to-all falls short

Standard all-to-all assumes **every rank sends the same amount to every other rank**. Tensor parallel shard exchange fits that assumption, so NCCL `all_to_all` is enough.

MoE dispatch violates it outright. See Figure 2.

<div class="fig">
<svg viewBox="0 0 680 320" width="100%" font-size="13" font-weight="400" role="img" aria-label="The variable-length dispatch problem and its two solutions">
<title>Figure 2: every rank sends a different amount to every destination, so fixed-size all-to-all wastes bandwidth</title>

<text x="40" y="26" font-size="11" fill="#1a1a1a" font-weight="600">Real load: 4 ranks, tokens scattered to 4 destinations by routing</text>

<g font-size="9" fill="#6b7280">
<text x="66" y="48" text-anchor="middle">to R0</text>
<text x="126" y="48" text-anchor="middle">to R1</text>
<text x="186" y="48" text-anchor="middle">to R2</text>
<text x="246" y="48" text-anchor="middle">to R3</text>
</g>

<g font-size="10">
<text x="20" y="72" fill="#1a1a1a">R0 sends</text>
<text x="20" y="102" fill="#1a1a1a">R1 sends</text>
<text x="20" y="132" fill="#1a1a1a">R2 sends</text>
<text x="20" y="162" fill="#1a1a1a">R3 sends</text>
</g>

<rect x="50" y="58" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="84" y="58" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="118" y="58" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="152" y="58" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="186" y="58" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="220" y="58" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="254" y="58" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>

<rect x="50" y="88" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="84" y="88" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="118" y="88" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="152" y="88" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="186" y="88" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="220" y="88" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="254" y="88" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>

<rect x="50" y="118" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="84" y="118" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="118" y="118" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="152" y="118" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="186" y="118" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="220" y="118" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="254" y="118" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>

<rect x="50" y="148" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="84" y="148" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="118" y="148" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="152" y="148" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="186" y="148" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="220" y="148" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="254" y="148" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>

<text x="306" y="72" font-size="10" fill="#b91c1c">Hottest: R0 to R3, 4 tokens</text>
<text x="306" y="102" font-size="10" fill="#6b7280">Coldest: R1 to R0, 1 token</text>
<text x="306" y="132" font-size="10" fill="#6b7280">4x spread within one row</text>

<line x1="30" y1="186" x2="650" y2="186" stroke="#e5e7eb" stroke-width="1"/>

<text x="40" y="208" font-size="11" fill="#1a1a1a" font-weight="600">Option A: probe first (counts over a tiny message)</text>
<rect x="40" y="218" width="132" height="30" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="106" y="237" font-size="10" fill="#1d4ed8" text-anchor="middle">all-to-all(counts), 4 bytes</text>
<rect x="188" y="218" width="150" height="30" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="263" y="237" font-size="10" fill="#047857" text-anchor="middle">send exact lengths, zero waste</text>
<text x="356" y="237" font-size="10" fill="#6b7280">Cost: one extra sync, +1 RTT</text>

<text x="40" y="278" font-size="11" fill="#1a1a1a" font-weight="600">Option B: preallocate a ceiling (fixed slots)</text>
<rect x="40" y="288" width="132" height="26" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="106" y="305" font-size="10" fill="#1d4ed8" text-anchor="middle">N fixed slots per expert</text>
<rect x="188" y="288" width="150" height="26" rx="4" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="263" y="305" font-size="10" fill="#854f0b" text-anchor="middle">one shot, predictable latency</text>
<text x="356" y="305" font-size="10" fill="#6b7280">Cost: idle slots burn bandwidth, or drop tokens</text>
</svg>
</div>

Neither option is universally better. They map onto **two different SLAs**:

- **Prefill favours throughput** goes with A: the payload is large, so wasting 30 percent of bandwidth is expensive, and one extra RTT is irrelevant when the step already takes hundreds of milliseconds.
- **Decode favours interactivity** goes with B: the per-step payload is tiny, an extra RTT lands directly on TPOT, so waste bandwidth rather than wait.

### 3.2 DeepEP system semantics: two modes for two SLAs

DeepEP is essentially that conclusion turned into engineering. It ships two kernel families, and the difference is not algorithmic elegance but **which workload it assumes**:

| | **normal kernel** | **low-latency kernel** |
|---|---|---|
| Target workload | Prefill / training (large, variable batch) | Decode (tiny batch, every step) |
| Cross-node path | intra-node NVLink forwarding plus inter-node RDMA | straight RDMA (IBGDA) |
| Buffering | allocate on demand (Option A) | fixed slots (Option B) |
| CUDA Graph capture | No (shapes vary) | Yes (fixed shapes) - the reason DeepEP v2 ElasticBuffer let cross-node MoE decode enter the graph, as noted in `aihot-20260912` |
| SM footprint | many SMs move data | single-digit percentage of SMs, rest free for compute |
| Overlap style | chunk-level communication and compute pipelining | hook style: send now, `hook()` when the result is needed |

Two details deserve their own paragraph.

**First, normal-kernel forwarding is a shortcut, not a detour.** If every rank RDMA-sent directly to every destination rank, 64 GPUs would mean 64x64 connections and the inter-node fabric would saturate. DeepEP instead gathers tokens over intra-node NVLink onto the GPU responsible for each destination node, then issues a single RDMA from there. Cross-node traffic drops from O(GPUs squared) to O(nodes), at the cost of one extra NVLink hop inside the node. Since NVLink is an order of magnitude faster than RDMA, that trade is clearly worth it.

**Second, the only reason the low-latency kernel exists is that you cannot afford to wait.** A decode step carries tens to hundreds of tokens, small enough that bandwidth barely matters, but **the latency is added straight onto the critical path**. What you want is determinism and a small footprint, not bandwidth efficiency: fixed buffers keep shapes stable so the step can be captured in a graph, few SMs keep compute resources free, and hooks let the engine do other work while the transfer is in flight.

### 3.3 FP8 dispatch: shrink the payload before you send it

An easy optimization to overlook: dispatch moves hidden states, so if weights are already FP8, tokens can be quantized before sending and traffic halves.

The price is carrying scales, and they must be **per token**, otherwise accuracy falls apart. That in turn constrains the Grouped GEMM implementation, which must accept one scale per group rather than a single global scale. That coupling is the subject of the next section.

## 4. Grouped GEMM: turn E launches into 1, and E tails into 1

### 4.1 What the naive approach gets wrong

The MoE FFN is fundamentally E independent small matmuls: expert e multiplies its own `M_e` tokens by its own weight `W_e`. The most direct implementation is a loop:

```python
for e in range(num_experts):
    out[e] = x_e @ W[e]   # E kernel launches
```

That has two fatal problems, and the second matters far more than the first.

<div class="fig">
<svg viewBox="0 0 680 290" width="100%" font-size="13" font-weight="400" role="img" aria-label="Loop-per-expert GEMM versus grouped GEMM">
<title>Figure 3: the issue is not launch overhead, it is that every expert pays its own tail effect</title>

<text x="40" y="26" font-size="11" fill="#b91c1c" font-weight="600">Loop per expert: E launches, each with its own ramp-up and drain</text>

<g font-size="9">
<text x="20" y="56" fill="#6b7280">e0</text>
<text x="20" y="86" fill="#6b7280">e1</text>
<text x="20" y="116" fill="#6b7280">e2</text>
<text x="20" y="146" fill="#6b7280">e3</text>
</g>

<rect x="44" y="44" width="70" height="18" rx="2" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="116" y="44" width="10" height="18" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>
<text x="140" y="57" font-size="9" fill="#b91c1c">small M, underfilled</text>

<rect x="44" y="74" width="128" height="18" rx="2" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="174" y="74" width="10" height="18" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>

<rect x="44" y="104" width="30" height="18" rx="2" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="76" y="104" width="10" height="18" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>
<text x="100" y="117" font-size="9" fill="#b91c1c">30 rows: cannot fill a tile, spins idle</text>

<rect x="44" y="134" width="96" height="18" rx="2" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="142" y="134" width="10" height="18" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>

<text x="220" y="60" font-size="10" fill="#6b7280">Each red block is one ramp-up plus drain</text>
<text x="220" y="78" font-size="10" fill="#6b7280">E experts means E segments, none fills another gap</text>
<text x="220" y="112" font-size="10" fill="#b91c1c">Total time is roughly sum of max(compute, launch cost)</text>
<text x="220" y="130" font-size="10" fill="#b91c1c">Worse as E grows: 128 experts means 128 wasted tails</text>

<line x1="30" y1="172" x2="650" y2="172" stroke="#e5e7eb" stroke-width="1"/>

<text x="40" y="196" font-size="11" fill="#047857" font-weight="600">Grouped GEMM: one launch, all expert tiles in a single queue</text>

<rect x="44" y="210" width="180" height="18" rx="2" fill="#86efac" stroke="#10b981" stroke-width="0.5"/>
<text x="134" y="223" font-size="9" fill="#064e3b" text-anchor="middle">e0 tiles</text>
<rect x="226" y="210" width="240" height="18" rx="2" fill="#6ee7b7" stroke="#10b981" stroke-width="0.5"/>
<text x="346" y="223" font-size="9" fill="#064e3b" text-anchor="middle">e1 tiles</text>
<rect x="468" y="210" width="110" height="18" rx="2" fill="#86efac" stroke="#10b981" stroke-width="0.5"/>
<text x="523" y="223" font-size="9" fill="#064e3b" text-anchor="middle">e2 + e3</text>

<text x="44" y="244" font-size="10" fill="#047857">One continuous run, draining once at the end</text>
<text x="44" y="262" font-size="10" fill="#6b7280">Small experts (e2/e3, 30 rows each) get backfilled by later experts: grouped scheduling, not sequential math</text>
</svg>
</div>

- The surface problem: E launches, each with fixed overhead. 128 experts means 128 of them.
- **The real problem: computing each expert separately makes each pay its own tail effect.** A 30-row expert and an 800-row expert both go through ramp-up, tile fill, and drain. On the 30-row pass SM utilization might be 10 percent, and that idle capacity cannot be filled by other experts, because they live in a different launch.

Grouped GEMM solves both at once: **all tiles from all experts enter one global queue and whoever is free picks up the next tile**. Small experts get backfilled by later work automatically, and the batch drains once at the end. Tail effect goes from E occurrences to one.

The side benefit: launches drop from E to one, so the CPU stops being the bottleneck. Under graph mode this is not a nice-to-have, since only a single launch can be captured.

### 4.2 What the interface looks like: offsets are the entire contract

The Grouped GEMM interface looks odd at first: there is no batch size, only an offsets array.

```python
# counts: tokens received per expert (e0=3, e1=0, e2=5, e3=2)
counts  = torch.tensor([3, 0, 5, 2])
offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])   # [0, 3, 3, 8, 10]
```

`offsets[e]:offsets[e+1]` is the row range of expert e inside the **already sorted** token tensor. Note that e1 has `offsets` `[3, 3]`, an **empty group of length zero**: legal, and mandatory to support, because routing can absolutely starve an expert.

That explains two things:

1. **Why dispatch must do the permute itself.** Grouped GEMM needs each expert tokens stored contiguously, and dispatch is already packing tokens to send, so the packing order can simply be by expert. Do these as two passes and you add a full gather.
2. **Why this interface is naturally variable-length.** No capacity factor, no padding, no drop. Every expert computes exactly the tokens it actually received.

A verifiable equivalent implementation (`torch._grouped_mm` needs PyTorch 2.9 or newer; check the signature for your version, and older releases can use the third-party `grouped_gemm` package):

```python
import torch

counts  = torch.tensor([3, 0, 5, 2], device='cuda')
offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
T, H, I = int(counts.sum()), 1024, 512

x = torch.randn(T, H, dtype=torch.bfloat16, device='cuda')      # already sorted by expert
w = torch.randn(4, I, H, dtype=torch.bfloat16, device='cuda')   # weights of 4 experts

# grouped: one call, segmented internally by offsets
out = torch._grouped_mm(x, w.transpose(1, 2), offs=offsets)

# reference: the loop version, semantics must match exactly
ref = torch.empty_like(out)
for e in range(4):
    s, t = int(offsets[e]), int(offsets[e + 1])
    if t > s:                       # skip empty groups, do not compute them
        ref[s:t] = x[s:t] @ w[e].T

torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
```

One detail in that snippet is worth copying: **empty groups must be skipped explicitly**. The `t > s` guard is not defensive coding, it genuinely fires. Run 100k steps and some expert will see zero tokens.

### 4.3 Masked versus grouped: the legacy of capacity factor

Early MoE (GShard, Switch) gave each expert a fixed-capacity buffer via a **capacity factor**: easy to compute, easy to preallocate, easy to send as fixed-length messages. The cost is that tokens beyond the limit are **dropped** and unused slots are **zero-padded**.

The MegaBlocks paper (MLSys 2023) argues that both are waste and that MoE should be written directly as a **block-sparse GEMM** with no capacity ceiling at all. The numbers quoted in `aihot-20260925` on this site (40 percent faster than the best Tutel configuration, 2.4x faster than dense Megatron-LM, zero token dropping) refer to exactly this.

| | masked / fixed-capacity GEMM | grouped / variable-length GEMM |
|---|---|---|
| Capacity | Fixed capacity factor | No ceiling, real counts |
| Overflow tokens | **Dropped** (quality loss) | All computed |
| Empty slots | Zero-padded, still burn compute | Nonexistent |
| Communication | Fixed length, simple | Variable, needs counts exchange |
| Fit for inference | Low (quality unacceptable) | High |

**Inference should almost never use a capacity factor.** Dropping 5 percent of tokens in training can be recovered by the next batch. In serving, what you drop is part of a user answer. That is why modern inference stacks all take the variable-length grouped route, and why they must handle the variable-length communication from section 3.1.

### 4.4 The quantization coupling: per-group scales

Earlier I noted that FP8 dispatch needs per-token scales. On the Grouped GEMM side, quantization imposes a second requirement: **each expert needs its own activation scale**.

The reason is concrete. Across 384 experts the activation distributions differ; a finance-text expert and a code expert may differ by an order of magnitude in range. Share one scale across the whole model and the narrow experts get clipped badly. So the real FP8 grouped GEMM signature looks like this:

```text
out = grouped_gemm(x_fp8, w_fp8,
                   x_scale[group],     # one activation scale per group
                   w_scale[group],     # one weight scale per group
                   offsets)
```

This is why DeepGEMM supported both contiguous grouped and masked variants from day one and baked per-group scaling into the kernel. **The quantization scheme and the GEMM interface are designed together; you cannot choose them independently.**

## 5. Combine: the inverse operation, but a different latency account

Combine sends expert outputs back along the reverse path, then each GPU performs the top-k weighted sum with gate weights. Formally it is the inverse of dispatch, but three things are asymmetric:

| | dispatch | combine |
|---|---|---|
| Direction | tokens to the expert GPU | results back to the origin GPU |
| Position | **before** Grouped GEMM | **after** Grouped GEMM |
| Where weights apply | Nowhere (routing only picks addresses) | **Mandatory** (sum gate_i * y_i) |
| Hideable | Yes (send early, wait when needed) | **Mostly not**: must be complete before the next layer |
| Fusable | - | Can fuse with the next layer reduce-scatter |
| Fusable into one kernel | Yes (fuses with dispatch) | Yes, and this is exactly the premise of section 6 |

That last row is a practical optimization. If a TP all-reduce follows (the FFN output must be reduced across TP), the combine all-to-all runs in the same direction and the two can fuse into one collective, saving a round trip. Done separately, it is two full round trips.

## 6. Fusion: welding five steps into one kernel, MegaMoE and wave pipelining

The previous five sections split the pipeline apart because the components have different semantics. But **on a GPU, a semantic boundary is itself a source of waste**: every boundary costs a kernel launch, and every intermediate tensor gets written to HBM and read back. SwiGLU is the worst case, an elementwise multiply that leaves the compute units nearly idle while pushing the entire activation out to memory and pulling it back in.

**MegaMoE**, which DeepSeek open-sourced inside DeepGEMM in April 2026 (later revisions call it MegaMoE squared), collapses those five steps into a **single kernel invocation**: dispatch, Linear-1, SwiGLU, Linear-2 and combine all run without touching HBM, and intermediate tensors live only in a GPU scratch buffer.

Pin the scope first: **MegaMoE is an execution structure, not a model structure.** It does not change which tokens go to which expert, which is routing's job. It changes how those experts run on the GPU. So it does not replace MoE, it sits one layer below it.

### 6.1 Fusion removes three classes of cost, not one launch

| Cost | What the naive path pays | After fusion |
|---|---|---|
| CPU-side launches and syncs | E experts times 5 segments, one launch and one sync each | 1 |
| Intermediate tensors to HBM | Packed token buffer and the SwiGLU activation, written out and read back | Scratch buffer only, never HBM |
| GPU-to-CPU round trips | Every segment returns to the CPU before the next is issued | Driven entirely on the GPU |

The third row matters most. In a conventional implementation the CPU issues every kernel, so once the expert count grows, **launch overhead and synchronization latency alone eat a visible slice of the timeline**. That is why section 4.1's claim about turning E launches into 1 matters more at the systems layer than at the algorithms layer.

The site's `aihot-20260821` records the persistent form on SM100 and corroborates exactly this: the shared L1 segment, the routed dispatch plus MMA, the shared L2 segment and the FP32 accumulation are all scheduled **inside one persistent kernel**, **writing the BF16 output exactly once**. Note that shared experts and routed experts fuse together there too, so the fusion boundary is not one MoE layer but the entire FFN's work.

### 6.2 Wave scheduling rearranges time, it does not add bandwidth

Fusion only welds the seams shut. The actual speedup comes from the second move: **once dispatch, compute and combine live in the same kernel, the experts can be split into waves so different waves sit at different stages.**

The DeepSeek-V4 technical report puts it this way: split the experts into waves, each holding a small portion of them, and as soon as every expert within a wave has finished its communication, that wave starts computing immediately without waiting for the others. In steady state three things proceed concurrently: the current wave computes, the next wave transmits, and completed waves send results back.

<div class="fig">
<svg viewBox="0 0 680 322" width="100%" font-size="13" font-weight="400" role="img" aria-label="Execution timeline of plain MoE versus MegaMoE with wave pipelining">
<title>Figure 4: five serial operators versus wave pipelining after fusion into one kernel</title>

<text x="40" y="24" font-size="11" fill="#b91c1c" font-weight="600">Naive: five serial segments, red seams are kernel boundaries plus syncs</text>

<rect x="44" y="38" width="92" height="32" rx="4" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="90" y="58" font-size="10" fill="#1a1a1a" text-anchor="middle">dispatch</text>
<rect x="161" y="38" width="92" height="32" rx="4" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="207" y="58" font-size="10" fill="#1a1a1a" text-anchor="middle">Linear-1</text>
<rect x="278" y="38" width="92" height="32" rx="4" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="324" y="58" font-size="10" fill="#1a1a1a" text-anchor="middle">SwiGLU</text>
<rect x="395" y="38" width="92" height="32" rx="4" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="441" y="58" font-size="10" fill="#1a1a1a" text-anchor="middle">Linear-2</text>
<rect x="512" y="38" width="92" height="32" rx="4" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="558" y="58" font-size="10" fill="#1a1a1a" text-anchor="middle">combine</text>

<rect x="142" y="43" width="13" height="22" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>
<rect x="259" y="43" width="13" height="22" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>
<rect x="376" y="43" width="13" height="22" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>
<rect x="493" y="43" width="13" height="22" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>

<text x="40" y="88" font-size="10" fill="#6b7280">A red seam means compute cannot start until communication lands, and the result lands back on the CPU</text>

<text x="40" y="118" font-size="11" fill="#047857" font-weight="600">MegaMoE: one kernel plus wave pipelining, three streams in steady state</text>

<rect x="36" y="130" width="608" height="132" rx="6" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>

<line x1="310" y1="142" x2="310" y2="250" stroke="#6b7280" stroke-width="0.5" stroke-dasharray="3 3"/>
<line x1="384" y1="142" x2="384" y2="250" stroke="#6b7280" stroke-width="0.5" stroke-dasharray="3 3"/>

<text x="48" y="164" font-size="9" fill="#6b7280">wave 0</text>
<rect x="108" y="146" width="92" height="26" rx="3" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="154" y="163" font-size="10" fill="#1d4ed8" text-anchor="middle">send in</text>
<rect x="200" y="146" width="110" height="26" rx="3" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="255" y="163" font-size="10" fill="#047857" text-anchor="middle">compute</text>
<rect x="310" y="146" width="92" height="26" rx="3" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="356" y="163" font-size="10" fill="#1d4ed8" text-anchor="middle">send out</text>

<text x="48" y="200" font-size="9" fill="#6b7280">wave 1</text>
<rect x="200" y="182" width="92" height="26" rx="3" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="246" y="199" font-size="10" fill="#1d4ed8" text-anchor="middle">send in</text>
<rect x="292" y="182" width="110" height="26" rx="3" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="347" y="199" font-size="10" fill="#047857" text-anchor="middle">compute</text>
<rect x="402" y="182" width="92" height="26" rx="3" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="448" y="199" font-size="10" fill="#1d4ed8" text-anchor="middle">send out</text>

<text x="48" y="236" font-size="9" fill="#6b7280">wave 2</text>
<rect x="292" y="218" width="92" height="26" rx="3" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="338" y="235" font-size="10" fill="#1d4ed8" text-anchor="middle">send in</text>
<rect x="384" y="218" width="110" height="26" rx="3" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="439" y="235" font-size="10" fill="#047857" text-anchor="middle">compute</text>
<rect x="494" y="218" width="92" height="26" rx="3" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="540" y="235" font-size="10" fill="#1d4ed8" text-anchor="middle">send out</text>

<text x="347" y="280" font-size="10" fill="#1a1a1a" text-anchor="middle">Inside the dashed window: wave 0 sends out, wave 1 computes, wave 2 sends in</text>

<rect x="60" y="296" width="14" height="10" rx="2" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="82" y="305" font-size="10" fill="#6b7280">communication (send in and out)</text>
<rect x="290" y="296" width="14" height="10" rx="2" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="312" y="305" font-size="10" fill="#6b7280">expert compute</text>
<rect x="440" y="296" width="14" height="10" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>
<text x="462" y="305" font-size="10" fill="#6b7280">sync seam, welded shut</text>
</svg>
</div>

How to cut the waves is a real trade-off. **Too small** and the per-wave fixed overhead never amortizes, so it gets slower. **Too large** and it degenerates into a batch-wide sync, which is the same as not cutting at all. The tuning logic is therefore identical to the normal versus low-latency split in section 3.2: **is your bottleneck bandwidth or latency.**

One thing has to be said plainly here, or the conclusion comes out wrong: **wave scheduling rearranges time, it does not reduce traffic.** Not a single token goes unsent, and the `O(N·k·h)` volume is untouched. It converts waiting into overlap. It does not convert more into less.

### 6.3 Overlap has a hard precondition: the 6144 line

If communication is intrinsically more expensive than compute, no amount of wave cutting hides it. So whether overlap works is not an engineering question at all, it is **an inequality you can compute up front**.

For one token-expert pair, counting the three SwiGLU projections (gate, up, down) at `2hd` each:

- **Compute** `C = 6hd` FLOPs
- **Traffic** `B = h + 2h = 3h` bytes (dispatch sends `h` in FP8, combine returns `2h` in BF16)

Requiring compute time to be at least communication time, that is `C/B ≤ V_comp / V_comm`, the `h` cancels:

```text
C / B = 6hd / 3h = 2d        units: FLOPs / Byte
```

DeepSeek-V4-Pro uses `d = 3072`, so the threshold is **6144 FLOPs/Byte**.

What makes this line useful is that it is a **hardware co-design criterion rather than one vendor's trick**: if a chip's compute-to-bandwidth ratio sits below it, no scheduler saves you, and above it, overlap pays. That also explains why the same work validates on both NVIDIA GPUs and Huawei Ascend NPUs, because the criterion is portable.

| Case | Consequence |
|---|---|
| `d` larger (wider experts) | Threshold rises, communication is easier to hide |
| `d` smaller (narrow experts, such as a 3072-expert design) | Threshold falls, narrow experts are intrinsically harder to overlap, so **their bottleneck is per-GPU scheduling overhead, not cross-GPU bandwidth** |
| Bandwidth doubles | Effective threshold doubles, but the traffic does not change |
| Traffic itself grows (BF16 dispatch, say) | `B` rises to `4h`, the threshold drops to `1.5d`, a third instantly gone |

### 6.4 What it does not solve

Three limits worth remembering, or it gets treated as a cure-all:

1. **Forward pass only.** MegaMoE in DeepGEMM is inference-oriented; training needs an overlapping backward path written for it. Prime Intellect added one and reports 3.11x at the kernel level but 1.34x end to end, and that three-fold gap is the point: the end-to-end bottleneck is not this kernel.
2. **Power budget bound.** Pushing communication under compute means the compute units run close to continuously saturated, so **the power wall can arrive before the bandwidth wall**. Finer waves are therefore not automatically better.
3. **Neither routing skew nor traffic volume changes.** When one expert absorbs 40 percent of the tokens, fusion and waves have nothing to offer; that is section 7 territory. **It optimizes waiting, not work**, the same argument as misreading two in section 10.

### 6.5 Not to be confused with MagiMoE

The two names differ by one letter and solve different problems:

| | **MegaMoE** (DeepSeek / DeepGEMM) | **MagiMoE** (Sand.ai / MAGI-2) |
|---|---|---|
| Setting | **Cross-GPU EP**, experts spread over many GPUs | **Within one GPU**, a very large number of narrow experts (3072, say) |
| Dominant bottleneck | Overlapping cross-GPU communication with compute in time | Launch and padding waste from looping over experts |
| Mechanism | Five-step fusion into one kernel plus wave scheduling | Routing, Top-K, compact layout, grouped FFN, SwiGLU and combine fused |
| View of communication | **Rearrange time** to hide it | **Decouple structure** to shrink it (fixed-order exchange first, `O(N·H)` independent of top-k) |

That last row is the real conceptual divide: **MegaMoE accepts that the traffic exists and works out how to hide it, while MAGI-2's Multi-Head Latent MoE decouples traffic from routing so it no longer grows with top-k.** One is an engineering problem, the other a structural one. They stack, but they do not substitute for each other.

## 7. EPLB: training uses loss, inference moves weights

### 7.1 Why inference has no loss to lean on

Training MoE has auxiliary losses or bias terms that push routing toward balance through gradients. **At inference there is no gradient, no loss, and no permission to change the model.** Routing is a fixed function of the weights, and a hot expert stays hot.

That leaves two paths:

1. **Routing side**: add a bias that forcibly changes top-k outcomes. This is the DeepSeek aux-loss-free bias, but it is baked in during **training** and cannot be changed at deployment.
2. **Deployment side**: leave routing alone and change the **physical placement** of experts, giving hot experts redundant replicas and putting those replicas on idle GPUs.

EPLB takes path two. It complements aux-loss-free bias rather than replacing it: one governs training, the other governs deployment.

### 7.2 How to read moe_lb

Define the metric first:

```text
moe_lb = load on the slowest rank / mean load across ranks
```

- Perfectly balanced equals 1.0.
- 1.09 means **the busiest GPU does only 9 percent more work than the average**.

That looks small but matters enormously at EP64, because every step waits for the slowest GPU, so **effective throughput is roughly mean throughput divided by moe_lb**. A value of 1.09 already costs 9 percent of compute, and skipping rebalancing can push moe_lb past 2.0 and halve throughput.

`fw10` records DeepInfer measurements: 1.074 and 1.045 in prefill, 1.090 in decode.

### 7.3 Redundant replicas plus minimax greedy placement

<div class="fig">
<svg viewBox="0 0 680 330" width="100%" font-size="13" font-weight="400" role="img" aria-label="EPLB redundant expert replication and packing">
<title>Figure 5: hot experts are replicated, then packed onto different GPUs</title>

<text x="40" y="26" font-size="11" fill="#b91c1c" font-weight="600">Before: e7 takes 40 percent of tokens while other experts idle</text>

<g font-size="10">
<text x="20" y="56" fill="#6b7280">GPU 0</text>
<text x="20" y="86" fill="#6b7280">GPU 1</text>
<text x="20" y="116" fill="#6b7280">GPU 2</text>
<text x="20" y="146" fill="#6b7280">GPU 3</text>
</g>

<rect x="56" y="44" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="57" font-size="9" fill="#1d4ed8" text-anchor="middle">e0</text>
<rect x="132" y="44" width="70" height="18" rx="3" fill="#fca5a5" stroke="#ef4444" stroke-width="0.8"/>
<text x="167" y="57" font-size="9" fill="#7f1d1d" text-anchor="middle">e7 hot</text>

<rect x="56" y="74" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="87" font-size="9" fill="#1d4ed8" text-anchor="middle">e1</text>
<rect x="132" y="74" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="167" y="87" font-size="9" fill="#1d4ed8" text-anchor="middle">e2</text>

<rect x="56" y="104" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="117" font-size="9" fill="#1d4ed8" text-anchor="middle">e3</text>
<rect x="132" y="104" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="167" y="117" font-size="9" fill="#1d4ed8" text-anchor="middle">e4</text>

<rect x="56" y="134" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="147" font-size="9" fill="#1d4ed8" text-anchor="middle">e5</text>
<rect x="132" y="134" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="167" y="147" font-size="9" fill="#1d4ed8" text-anchor="middle">e6</text>

<text x="230" y="60" font-size="10" fill="#b91c1c">GPU 0 carries double load</text>
<text x="230" y="78" font-size="10" fill="#6b7280">the other 7 GPUs wait on it</text>

<line x1="30" y1="172" x2="650" y2="172" stroke="#e5e7eb" stroke-width="1"/>

<text x="40" y="196" font-size="11" fill="#047857" font-weight="600">After: e7 gets 3 replicas and splits the load</text>

<rect x="56" y="212" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="225" font-size="9" fill="#1d4ed8" text-anchor="middle">e0</text>
<rect x="132" y="212" width="70" height="18" rx="3" fill="#6ee7b7" stroke="#10b981" stroke-width="0.8"/>
<text x="167" y="225" font-size="9" fill="#064e3b" text-anchor="middle">e7 A</text>

<rect x="56" y="242" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="255" font-size="9" fill="#1d4ed8" text-anchor="middle">e1</text>
<rect x="132" y="242" width="70" height="18" rx="3" fill="#6ee7b7" stroke="#10b981" stroke-width="0.8"/>
<text x="167" y="255" font-size="9" fill="#064e3b" text-anchor="middle">e7 B</text>

<rect x="56" y="272" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="285" font-size="9" fill="#1d4ed8" text-anchor="middle">e3</text>
<rect x="132" y="272" width="70" height="18" rx="3" fill="#6ee7b7" stroke="#10b981" stroke-width="0.8"/>
<text x="167" y="285" font-size="9" fill="#064e3b" text-anchor="middle">e7 C</text>

<text x="230" y="228" font-size="10" fill="#047857">Three GPUs take a third each, the slowest is no longer the wall</text>
<text x="230" y="246" font-size="10" fill="#6b7280">Which to replicate: the one that lowers the max the most</text>
<text x="230" y="264" font-size="10" fill="#6b7280">Not simply the hottest one</text>
<text x="230" y="288" font-size="10" fill="#b91c1c">Cost: 3 copies of e7 weights occupy 3x the memory</text>
</svg>
</div>

The algorithm is not complicated. Score each expert from historical `tokens_per_expert`, greedily pick the replica that reduces the maximum load the most, and repeat until the redundancy budget is spent. The objective is **minimax**, not the average.

One easy mistake: **minimax greedy is not the same as replicating the hottest expert.** If the hottest expert is still the hottest after one replica, a second replica buys little, whereas replicating the second-hottest may reduce the new maximum more. Choose based on **the new max after replication**, not the current value.

### 7.4 Why the rebalance period is so long

Moving experts costs real money. One expert weight (say hidden 2048 by intermediate 512 at two bytes) is megabytes; rebalancing hundreds of them is gigabytes of GPU-to-GPU traffic. While the move is in flight those experts are unavailable.

So EPLB is **periodic**, not per-step. DeepInfer published this configuration:

| Stage | Redundant experts per GPU at start | Statistics window | Rebalance period |
|---|---|---|---|
| Prefill | 4 | 100 steps | 1000 steps |
| Decode | 1 | 1000 steps | 10000 steps |

Three different rationales:

- **Redundancy level**: prefill batches are large and must finish in one step, so hot-spotting is severe and a few replicas help. Decode batches are small and routing is smoother, so one is enough.
- **Statistics window**: long enough to filter single-step noise, short enough to track real distribution shifts. Prefill uses 100 steps (many tokens per step, plenty of samples); decode uses 1000 (few tokens per step, so accumulate more).
- **Rebalance period**: far longer than the window because moving is expensive. The 1000 to 10000 ratio encodes the principle: **statistics are cheap and moves are expensive, so observe often and act rarely**.

### 7.5 That 0.0001 nudge

A detail worth stealing: EPLB adds a tiny offset when scoring experts.

```text
score[e] = tokens_per_expert[e] + e * 1e-4
```

This makes all scores **strictly distinct**, so argmax never ties and different layers or processes cannot return different results.

What that 0.0001 buys is **reproducible scheduling**. In a cluster of 384 experts across 64 GPUs, two exactly equal scores can lead different argmax implementations (or even the same one at different moments) to pick different experts, so each rebalance produces a different layout. When you are chasing an intermittent slowdown, that nondeterminism multiplies the debugging cost by an order of magnitude.

**Trading a sliver of optimality for determinism is worth far more than one percent throughput in a production system.**

### 7.6 The costs: fault domain and memory

EPLB is not free.

- **Memory**: replicas are real weights. Six experts plus one replica per GPU in decode means roughly 17 percent more expert weights.
- **Fault domain**: EP64 means 64 GPUs share a **single communication domain**, so one failed GPU takes the whole instance down. Recovery is not a fast restart but a restart that **re-derives EP size from all currently healthy GPUs** and continues degraded on what remains.
- **So the safety net for Wide-EP is spare capacity, not recovery speed.** Capacity planning must account for this.

## 8. Putting it together: the latency ledger of one decode step

Only on a shared timeline do the relative weights become visible.

| Segment | Nature | Hideable | Relative weight in decode |
|---|---|---|---|
| Route (gate plus top-k) | Local compute | No | Small but nonzero (every step) |
| Dispatch | **Pure communication** | Partly (hook: send early, wait late) | **High**: paid every step, added straight to TPOT |
| Grouped GEMM | Compute | - | Medium (tiny M in decode, memory bound) |
| Combine | **Pure communication** | Mostly not (**hideable once fused and waved, see section 6**) | **High**: complete before the next layer |
| Weighted reduction | Local compute | No | Small |

Two conclusions:

1. **In decode, communication consumes a much larger share than in prefill.** Prefill processes thousands of tokens at once, so Grouped GEMM dominates; decode processes hundreds, so the fixed cost of two dispatch-combine exchanges becomes significant. **That is the entire reason DeepEP ships a low-latency mode.**
2. **EPLB does not sit on this timeline.** It operates across many steps, and its benefit shows up in **accumulated throughput**, not single-step latency. So: **tune dispatch and combine for TTFT and TPOT, tune EPLB and EP size for cluster throughput.** These are two different sets of knobs; mixing them makes the changes cancel out.

## 9. Parameter decision table: how to set the four knobs

| Knob | Small value | Large value | How to decide |
|---|---|---|---|
| **EP size** (GPUs per expert group) | More experts per GPU, smaller domain, smaller fault domain | Fewer experts per GPU, less memory pressure, better depth in the interactive regime | **Look at the SLA.** Low concurrency with long outputs means a large EP only burns communication; high interactivity (250+ tok/s per user) makes EP64 up to 10x better than EP8 |
| **Dispatch mode** | - | - | **Normal for prefill, low-latency for decode.** One model can use different backends per stage |
| **Redundant experts** | Saves memory | Survives hot spots | **Look at routing skew.** Measure `moe_lb` first; above 1.15 you should add |
| **Rebalance period** | Tracks fast, expensive moves | Stable, lags drift | **Look at traffic volatility.** Steady traffic allows longer periods; strong diurnal or event peaks argue for shorter |

On the first row, the numbers from `fw10` bear repeating (64xSM100, fixed three draft tokens, discounted at a 75 percent per-token acceptance rate):

- At 100 tok/s per user, EP64 delivers about 26.6k tok/s/GPU and EP8 about 25k: **nearly identical peaks**.
- At 250 tok/s per user, EP64 still holds about 15.5k while EP8 falls to about 1.7k.

**Wide-EP buys depth in the interactive regime, not peak throughput.** This is the easiest thing to miss in capacity planning: compute only "throughput at full batch" and you will conclude, wrongly, that a larger EP does not help.

## 10. Three common misreadings

**Misreading one: Grouped GEMM is only about saving launches.**
Saving launches is a side benefit. The real win is collapsing E independent tail effects into one, and in regimes with hundreds of experts holding few tokens each, that is an order of magnitude bigger. Conversely, **with few experts (say 8) that are each well filled, Grouped GEMM buys little**, so do not expect it to rescue you.

**Misreading two: EPLB fixes routing skew.**
EPLB changes the **physical placement** of experts, not the routing outcome. If one expert really takes 40 percent of tokens, three replicas let three GPUs take a third each, but total compute is unchanged. **It optimizes waiting, not work.** To reduce work you must rebalance on the routing side (aux-loss-free bias) or change the model architecture.

**Misreading three: dispatch is bandwidth bound.**
In prefill that is true, since the payload is large. **In decode the bottleneck is latency**: the per-step payload is too small to fill bandwidth, and what eats time is the fixed cost of the round trip and the synchronization points. So optimizing dispatch in decode means **lowering latency and footprint** (fixed buffers, fewer SMs, hook overlap), not raising bandwidth utilization.

## 11. Ties to other posts on this site

- **`sys8` (parallel strategy landscape)** gives EP a qualitative entry in the five-knife table: all-to-all, traffic scaling with top-k and batch, needs expert load balancing. This post expands each of those three clauses into a mechanism: why all-to-all must be variable length, how the traffic scales, and how balancing works.
- **`sys12` (Mooncake)** covers request-level routing and KV movement across instances. This post covers **token-level routing inside a layer**. Two different granularities of dispatch; do not conflate them.
- **`fw10` (DeepInfer)** section 3 describes Wide-EP per-stage ratios and section 3.1 the minimax greedy replication with the 1e-4 nudge. This post adds **the other two components** (variable-length dispatch semantics and the Grouped GEMM offsets contract) and threads all four into one pipeline. `fw10` answers what DeepSeek shipped; this post answers why each of the four pieces looks the way it does.
- **`fw7` (CUDA Graph)** uses dispatcher for the graph-mode selector, a different thing from the token dispatcher here. But the two genuinely intersect: **the low-latency kernel fixed-buffer design exists precisely so MoE decode can enter a graph**, which is what DeepEP v2 ElasticBuffer accomplished per `aihot-20260912`.
- **`op1` (MLA operator)** takes the single-GPU operator view while this post takes the cross-GPU data flow view. MLA compresses KV; Grouped GEMM optimizes expert compute. Both aim to read and write less data per step, one in attention and one in the FFN.
- **`fa4` (DeepSeek V4) and `fa10` (MiniMax M3.1 Flash)** describe expert counts and sparsity (configurations such as 384 experts with top-8). **The model expert count directly sets EP communication complexity:** more experts means fewer per GPU and more scattered dispatch destinations. Their model view and this systems view are two sides of one coin.
- **`aihot-20260925` (MegaBlocks and grouped GEMM)** is the only other place on this site mentioning grouped GEMM, and only in passing. This post covers why it matters, what the interface looks like, and how it relates to capacity factor.
- **`aihot-20260821` (the DeepGEMM SM100 MegaMoE kernel) and `fa4` (DeepSeek V4)** only report MegaMoE conclusions, one recording the persistent fused kernel shape and the other the 1.5 to 1.73x speedup. **Section 6 here is the first place on this site that opens it up**: which three classes of cost fusion removes, why waves can overlap, how the 6144 FLOPs/Byte threshold is derived, and why it is not the same thing as MagiMoE.

---

## Series navigation

**Inference Systems Infrastructure Notes**

- [(12) Mooncake: A KVCache-centric PD-Disaggregated Architecture](/en/blog/sys12-mooncake-kvcache-centric-serving)
- [(11) Tuning SGLang HiCache L2 on Kunlunxin P800/P900](/en/blog/sys11-hicache-l2-kunlunxin-tuning)
- [(10) KV Cache in Production](/en/blog/sys10-kv-cache-engineering-sglang-vllm-attentionstore)
- [(9) The KV Cache Landscape](/en/blog/sys9-kv-cache-fundamentals-optimization)
- [(8) Parallelism Strategies and PD Disaggregation](/en/blog/sys8-parallel-strategies-pd-disaggregation)

**Related**

- [(10) DeepInfer dissected](/en/blog/fw10-deepinfer): source of the Wide-EP ratios and EPLB measurements
- [(7) CUDA Graph](/en/blog/fw7-cuda-graph): where the other dispatcher lives
- [(1) MLA operator](/en/blog/op1-mla-operator): the single-GPU operator counterpoint
- [(4) DeepSeek V4 architecture](/en/blog/fa4-deepseek-v4): where MegaMoE first appears on the model side
