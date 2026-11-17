---
title: "Inference Systems Infrastructure Notes (13): MoE Expert Parallelism, Communication and Compute — How Dispatcher, DeepEP, Grouped GEMM and EPLB Form One Pipeline"
description: "One MoE forward pass in five steps: route, dispatch, Grouped GEMM, combine, weighted reduction. Why the hard part of dispatch is variable length rather than getting data out, why DeepEP ships normal and low-latency kernels for two different SLAs, why the real win of Grouped GEMM is collapsing E tails into one rather than saving kernel launches, and why EPLB can only rebalance by statistics plus moving weights at inference time. Includes a decode step latency ledger and a parameter decision table."
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
<text x="352" y="128" font-size="10" fill="#b91c1c" text-anchor="middle">Pure communication, hideable only with chunk-level pipelining</text>

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

The MoE FFN is fundamentally E independent small matmuls: expert e multiplies its own $M_e$ tokens by its own weight $W_e$. The most direct implementation is a loop:

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

That last row is a practical optimization. If a TP all-reduce follows (the FFN output must be reduced across TP), the combine all-to-all runs in the same direction and the two can fuse into one collective, saving a round trip. Done separately, it is two full round trips.

## 6. EPLB: training uses loss, inference moves weights

### 6.1 Why inference has no loss to lean on

Training MoE has auxiliary losses or bias terms that push routing toward balance through gradients. **At inference there is no gradient, no loss, and no permission to change the model.** Routing is a fixed function of the weights, and a hot expert stays hot.

That leaves two paths:

1. **Routing side**: add a bias that forcibly changes top-k outcomes. This is the DeepSeek aux-loss-free bias, but it is baked in during **training** and cannot be changed at deployment.
2. **Deployment side**: leave routing alone and change the **physical placement** of experts, giving hot experts redundant replicas and putting those replicas on idle GPUs.

EPLB takes path two. It complements aux-loss-free bias rather than replacing it: one governs training, the other governs deployment.

### 6.2 How to read moe_lb

Define the metric first:

```text
moe_lb = load on the slowest rank / mean load across ranks
```

- Perfectly balanced equals 1.0.
- 1.09 means **the busiest GPU does only 9 percent more work than the average**.

That looks small but matters enormously at EP64, because every step waits for the slowest GPU, so **effective throughput is roughly mean throughput divided by moe_lb**. A value of 1.09 already costs 9 percent of compute, and skipping rebalancing can push moe_lb past 2.0 and halve throughput.

`fw10` records DeepInfer measurements: 1.074 and 1.045 in prefill, 1.090 in decode.

### 6.3 Redundant replicas plus minimax greedy placement

<div class="fig">
<svg viewBox="0 0 680 330" width="100%" font-size="13" font-weight="400" role="img" aria-label="EPLB redundant expert replication and packing">
<title>Figure 4: hot experts are replicated, then packed onto different GPUs</title>

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

### 6.4 Why the rebalance period is so long

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

### 6.5 That 0.0001 nudge

A detail worth stealing: EPLB adds a tiny offset when scoring experts.

```text
score[e] = tokens_per_expert[e] + e * 1e-4
```

This makes all scores **strictly distinct**, so argmax never ties and different layers or processes cannot return different results.

What that 0.0001 buys is **reproducible scheduling**. In a cluster of 384 experts across 64 GPUs, two exactly equal scores can lead different argmax implementations (or even the same one at different moments) to pick different experts, so each rebalance produces a different layout. When you are chasing an intermittent slowdown, that nondeterminism multiplies the debugging cost by an order of magnitude.

**Trading a sliver of optimality for determinism is worth far more than one percent throughput in a production system.**

### 6.6 The costs: fault domain and memory

EPLB is not free.

- **Memory**: replicas are real weights. Six experts plus one replica per GPU in decode means roughly 17 percent more expert weights.
- **Fault domain**: EP64 means 64 GPUs share a **single communication domain**, so one failed GPU takes the whole instance down. Recovery is not a fast restart but a restart that **re-derives EP size from all currently healthy GPUs** and continues degraded on what remains.
- **So the safety net for Wide-EP is spare capacity, not recovery speed.** Capacity planning must account for this.

## 7. Putting it together: the latency ledger of one decode step

Only on a shared timeline do the relative weights become visible.

| Segment | Nature | Hideable | Relative weight in decode |
|---|---|---|---|
| Route (gate plus top-k) | Local compute | No | Small but nonzero (every step) |
| Dispatch | **Pure communication** | Partly (hook: send early, wait late) | **High**: paid every step, added straight to TPOT |
| Grouped GEMM | Compute | - | Medium (tiny M in decode, memory bound) |
| Combine | **Pure communication** | Mostly not | **High**: complete before the next layer |
| Weighted reduction | Local compute | No | Small |

Two conclusions:

1. **In decode, communication consumes a much larger share than in prefill.** Prefill processes thousands of tokens at once, so Grouped GEMM dominates; decode processes hundreds, so the fixed cost of two dispatch-combine exchanges becomes significant. **That is the entire reason DeepEP ships a low-latency mode.**
2. **EPLB does not sit on this timeline.** It operates across many steps, and its benefit shows up in **accumulated throughput**, not single-step latency. So: **tune dispatch and combine for TTFT and TPOT, tune EPLB and EP size for cluster throughput.** These are two different sets of knobs; mixing them makes the changes cancel out.

## 8. Parameter decision table: how to set the four knobs

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

## 9. Three common misreadings

**Misreading one: Grouped GEMM is only about saving launches.**
Saving launches is a side benefit. The real win is collapsing E independent tail effects into one, and in regimes with hundreds of experts holding few tokens each, that is an order of magnitude bigger. Conversely, **with few experts (say 8) that are each well filled, Grouped GEMM buys little**, so do not expect it to rescue you.

**Misreading two: EPLB fixes routing skew.**
EPLB changes the **physical placement** of experts, not the routing outcome. If one expert really takes 40 percent of tokens, three replicas let three GPUs take a third each, but total compute is unchanged. **It optimizes waiting, not work.** To reduce work you must rebalance on the routing side (aux-loss-free bias) or change the model architecture.

**Misreading three: dispatch is bandwidth bound.**
In prefill that is true, since the payload is large. **In decode the bottleneck is latency**: the per-step payload is too small to fill bandwidth, and what eats time is the fixed cost of the round trip and the synchronization points. So optimizing dispatch in decode means **lowering latency and footprint** (fixed buffers, fewer SMs, hook overlap), not raising bandwidth utilization.

## 10. Ties to other posts on this site

- **`sys8` (parallel strategy landscape)** gives EP a qualitative entry in the five-knife table: all-to-all, traffic scaling with top-k and batch, needs expert load balancing. This post expands each of those three clauses into a mechanism: why all-to-all must be variable length, how the traffic scales, and how balancing works.
- **`sys12` (Mooncake)** covers request-level routing and KV movement across instances. This post covers **token-level routing inside a layer**. Two different granularities of dispatch; do not conflate them.
- **`fw10` (DeepInfer)** section 3 describes Wide-EP per-stage ratios and section 3.1 the minimax greedy replication with the 1e-4 nudge. This post adds **the other two components** (variable-length dispatch semantics and the Grouped GEMM offsets contract) and threads all four into one pipeline. `fw10` answers what DeepSeek shipped; this post answers why each of the four pieces looks the way it does.
- **`fw7` (CUDA Graph)** uses dispatcher for the graph-mode selector, a different thing from the token dispatcher here. But the two genuinely intersect: **the low-latency kernel fixed-buffer design exists precisely so MoE decode can enter a graph**, which is what DeepEP v2 ElasticBuffer accomplished per `aihot-20260912`.
- **`op1` (MLA operator)** takes the single-GPU operator view while this post takes the cross-GPU data flow view. MLA compresses KV; Grouped GEMM optimizes expert compute. Both aim to read and write less data per step, one in attention and one in the FFN.
- **`fa4` (DeepSeek V4) and `fa10` (MiniMax M3.1 Flash)** describe expert counts and sparsity (configurations such as 384 experts with top-8). **The model expert count directly sets EP communication complexity:** more experts means fewer per GPU and more scattered dispatch destinations. Their model view and this systems view are two sides of one coin.
- **`aihot-20260925` (MegaBlocks and grouped GEMM)** is the only other place on this site mentioning grouped GEMM, and only in passing. This post covers why it matters, what the interface looks like, and how it relates to capacity factor.

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
