---
title: 'Inference Systems Infrastructure Notes (8): Parallelism Strategies and PD Disaggregation — how to slice TP/PP/EP/SP/CP, and how to size the prefill:decode ratio'
description: 'The five slicing axes (TP/PP/EP/SP/CP) in one place: what each cuts, which collective it uses, how much it costs, and how they combine; then PD disaggregation — why split, what it costs, and how to actually compute the prefill:decode instance ratio.'
pubDate: 2026-09-08
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys8-parallel-strategies-pd-disaggregation
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

A large model does not fit on one card, so we **slice** it; prefill and decode have opposite hardware appetites, so we **split** them. This post covers two things: the five slicing axes (TP / PP / EP / SP / CP) — what each one cuts, what it costs, and how they combine — and the number inside PD disaggregation that is most often guessed by gut feel but can actually be computed: **how many prefill instances per decode instance**.

## 1. Five Knives: the parallelism landscape

| Strategy | What it cuts | Collective | Comm volume | Typical constraint |
|---|---|---|---|---|
| **TP** tensor parallel | weight matrices (along hidden / head dim) | all-reduce / all-gather | **high** (twice per layer) | extremely chatty; effectively locked inside a node (NVLink), TP ≤ 8 |
| **PP** pipeline parallel | layers (stages) | point-to-point activation | **low** (only stage boundaries) | pipeline bubbles; needs many micro-batches |
| **EP** expert parallel | MoE experts | all-to-all | **medium, varies with top-k / batch** | only meaningful for MoE; needs expert load balance |
| **SP** sequence parallel | the sequence parts TP cannot reach (LayerNorm / Dropout / residual) | reduce-scatter / all-gather | medium | pairs with TP, see sys2 |
| **CP** context parallel | the sequence (attention sharded along sequence) | ring comm | low (amortized over sequence length) | very long context, see sys2 and sys7 |

One-line distinction: **TP cuts "wide" (inside matrices), PP cuts "deep" (layers), EP cuts "experts", SP/CP cut "long" (sequence)**.

## 2. The Three Main Knives

### TP (Tensor Parallel)

Split a large matmul across cards by column (or row); each card computes a slice, then results are merged.

- **Cost**: every layer needs an all-reduce before and after attention and FFN — **very high communication frequency**. That is why TP essentially only runs over intra-node NVLink; cross-node TP dies on PCIe/network latency.
- **Practice**: 8 cards per node → TP=8 is the common ceiling. Go bigger and you need PP or a different mix.
- **Common confusion**: TP slices *inside* the weight matrices, so when a model is small enough to fit on one card you should not use TP — it buys latency, not throughput.

### PP (Pipeline Parallel)

Cut layers into stages, one stage per card (or group of cards), and push micro-batches through like an assembly line.

- **Cost**: **pipeline bubbles**. GPUs idle during fill and drain; the fewer the micro-batches, the larger the bubble fraction.
- **Strength**: only stage-boundary activations cross the wire, so **communication volume is far lower than TP** — PP is the workhorse for cross-node scaling.
- **Practice**: use enough micro-batches to squeeze the bubbles out (micro-batches ≫ stages).

### EP (Expert Parallel)

MoE-only. Place different experts on different cards; tokens are routed to the card holding their expert, computed, then shipped back.

- **Cost**: **all-to-all**, with volume varying by top-k, batch size and expert count. Worse, **expert load imbalance** — the card holding a hot expert becomes the bottleneck first.
- **Practice**: EP is usually stacked with TP (EP×TP) and needs a capacity factor plus an auxiliary load-balancing loss to stop popular experts from being flooded.
- **Note**: EP only applies to MoE. Dense models have no EP.

## 3. How to Combine: a Common Chain

1. **Push TP up to the single-node limit** first (saturate NVLink, cut per-request latency).
2. **Add PP when a node is not enough** (low cross-node volume, the scaling workhorse).
3. **Stack EP for MoE models**, with load balancing.
4. **Add SP / CP for very long context** (see Ring Attention in sys2 and DCP in sys7).
5. When mixing, keep **communication directions orthogonal**: sys2 already warned that TP×SP can make all-gather and ring comm collide — stagger them.

## 4. PD Disaggregation: why split the two phases

Prefill and decode have **opposite** hardware appetites:

| | Character | Bottleneck | Wants what card |
|---|---|---|---|
| **Prefill** | processes the whole prompt in one shot, big matmuls | **compute-bound** (FLOPs) | strong Tensor Cores, high FLOPS |
| **Decode** | one new token per step, but re-reads weights and KV constantly | **memory-bandwidth-bound** | high HBM bandwidth, large memory |

Deploying them together (colocated) means: during prefill, compute is busy and bandwidth idles; during decode, bandwidth is busy and compute idles — each phase drags the other down, and they also contend for scheduling and interfere with each other.

PD disaggregation puts the two phases on separate GPU pools; prefill finishes and ships the KV Cache to a decode node (see section 3 of fw4 for a dedicated walkthrough and architecture diagram).

**Reported gains**: DistServe up to **4.48×** goodput and **10.2×** tighter SLOs on identical hardware; Mooncake **1.5–2.5×** throughput on Kimi production load; Splitwise **1.4–2.1×** throughput per dollar depending on prompt-length distribution.

## 5. What PD Disaggregation Costs, and When It Pays

**The cost is KV transfer.** Once prefill finishes, the KV for the whole sequence must move to the decode node:

- Order of magnitude: a 70B model with a 4000-token prompt at FP16 → about **600 MB** of KV; over 200 Gbps InfiniBand that is roughly **24 ms**.
- Those 24 ms land **directly on the request path**. The shorter the prompt, the more easily this fixed cost eats the disaggregation gain.

| Disaggregate when | Keep it colocated when |
|---|---|
| long-prompt RAG / Agent workloads | short prompt, long generation chat |
| code completion (large prompt, small output) | a single 8-card box running a small/medium model |
| high concurrency with visible two-phase imbalance | no fast interconnect (no NVLink / RDMA) |

> Rule of thumb: **large enough scale + high share of long prompts + fast interconnect** — all three, or stick with colocated serving.

One easily missed point: **the prefill pool and the decode pool can run different TP sizes** (Mooncake supports this), because the two phases bottleneck differently and forcing one parallel degree on both is unnecessary.

## 6. Sizing the PD Ratio: this number is computable (the core of this post)

The industry writes it as **XpYd**: X prefill nodes, Y decode nodes (Mooncake's terminology; early vLLM supported only 1P1D, and xPyD with differing TP is now supported).

### 6.1 The Formula

Let:

```
λ    = request arrival rate (QPS)
Tin  = average input length (ISL, tokens)
Tout = average output length (OSL, tokens)
P    = per-card prefill throughput (tokens/s)
D    = per-card decode throughput (tokens/s, at target batch and TPOT)
ρ    = target utilization (suggest ≤ 0.7, headroom for bursts)
```

Cards needed on each side:

```
N_p = λ × Tin  / (P × ρ)
N_d = λ × Tout / (D × ρ)
```

**Divide one by the other and both λ and ρ cancel:**

```
N_p : N_d  =  (Tin / P)  :  (Tout / D)
```

> **Key insight**: the ratio is **independent of QPS**. QPS decides how many cards you run in total, not how you split them between prefill and decode. What actually sets the ratio is **input/output length** and **per-card throughput on each side**. This is exactly why a gut-feel 1:1 is so often wrong.

### 6.2 Two Counter-Intuitive Examples

Assume measured `P = 8000 tok/s/card` and `D = 2000 tok/s/card`.

**Scenario A: RAG / Agent (long input, short output)** `Tin=4000, Tout=500`

```
N_p : N_d = (4000/8000) : (500/2000) = 0.5 : 0.25 = 2 : 1
```

→ **prefill is twice decode.** For long prompts, prefill is the bulk of the work.

**Scenario B: Chat (short input, long output)** `Tin=500, Tout=2000`

```
N_p : N_d = (500/8000) : (2000/2000) = 0.0625 : 1 ≈ 1 : 16
```

→ **decode dominates overwhelmingly.** With short prompts and long generations, prefill is nearly free.

(For reference: DistServe's classic chatbot configuration is 2 prefill : 6 decode, i.e. **1:3**, sitting in the middle — consistent with chat being decode-heavy.)

### 6.3 A Four-Step Procedure

1. **Collect histograms**: sample prompt / output length distributions from real traffic — **do not rely on averages alone**, the tail drives tail latency.
2. **Measure per-card throughput**: measure prefill throughput `P` and decode throughput `D` separately under continuous batching, with `D` measured at your TPOT target.
3. **Plug into the ratio**, then scale up to whole cards using `ρ ≤ 0.7`.
4. **Validate against tail TTFT / ITL**, then tune. DistServe frames the whole exercise as maximizing goodput under dual TTFT / TPOT SLOs.

### 6.4 Dynamic Adjustment and Engineering Notes

- **Role swapping**: idle prefill instances can be converted to decode instances (and vice versa) — a standard utilization lever.
- **Elastic scaling**: the ratio drifts with load, so the cluster must support changing it without a restart.
- **Routing affinity**: place prefill and decode close together to cut KV transfer hops; for cross-rack deployments confirm `T_transfer = KV_size / BW` does not dominate TTFT.
- **Opposite batching policies**: prefill prefers **small batches** (lower latency); decode prefers **large batches** (higher throughput).
- **KV transfer optimization**: quantization (FP8/INT8) and sparsity (transfer only important tokens) can cut transfer volume by more than half — the same idea as KV Cache quantization in op2.

## 7. Selection Checklist

1. **Pick parallelism before disaggregation**: TP to the node limit → PP across nodes → EP for MoE → SP/CP for long context.
2. **Never run TP across nodes** — too chatty.
3. **PP needs enough micro-batches** to suppress bubbles.
4. **Three conditions for PD split**: large scale + high long-prompt share + NVLink/RDMA available.
5. **Do not guess the ratio**: compute `(Tin/P) : (Tout/D)`, and remember it is QPS-independent.
6. **The two sides may use different TP sizes** — no need to force one.
7. **Leave burst headroom**: target utilization ≤ 0.7.
8. **Watch KV transfer volume**: it is both a latency and a cost source; compress it where you can.

**Tying it together**: parallelism solves "the model does not fit"; disaggregation solves "the two phases drag each other down"; and the ratio decides "how much of each pool before you waste silicon". Those three layers together are the full answer to turning a large model into an actual service.

*Note: this post consolidates the 2026-08-11 to 2026-09-08 daily AI hotspot tracking, community tracking (PD disaggregation wired up, unified-pool PD transfer), and published results (DistServe / Mooncake / Splitwise) into a systematic version.*
