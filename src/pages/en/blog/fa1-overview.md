---
title: 'Frontier Architecture Decoding Notes (1): The 2026 Frontier Overview — the Impossible Triangle of Long-Context × Efficiency × Scale'
description: 'This series follows the "Speculative Decoding" and "Multimodal" threads to the frontier model architectures themselves in 2026. The opener sets up a triangle: long-context (1M), inference/training efficiency, and parameter scale (trillion-MoE) pull against each other. Kimi K3, MiniMax M3, and DeepSeek V4 each grab a vertex and break the deadlock by reshaping attention differently. The full progressive roadmap is included.'
pubDate: 2026-08-06
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa1-overview
layout: ../../../layouts/BlogPost.astro
---

## 0. Where This Series Sits

"Speculative Decoding Notes" covered **how to run existing LLMs faster**; "Multimodal Decoding Notes" covered **how a model perceives and acts on the world**. Both lines converge on one question: **where did model architecture itself get to, in 2026?**

Three frontier models launched in 2026 — **Kimi K3, MiniMax M3, DeepSeek V4** — form a natural controlled experiment: they face the same engineering problem but give three different answers. This series takes them apart "step by step," then puts the trend back together at the end.

> Roadmap: **(1) Overview & roadmap** → (2) Kimi K3: dual evolution of attention and MoE → (3) MiniMax M2.7→M3: the leap of sparse attention → (4) DeepSeek V4: extreme compression and efficient training.

## 1. A Triangle: Long-Context × Efficiency × Scale

To understand the three, first set up an "impossible triangle":

- **Vertex A · Long context (1M tokens)**: let the model "read a whole book / a whole codebase / a long tool-call history."
- **Vertex B · Efficiency (low KV, low per-token compute)**: as context grows and models get bigger, attention and KV-cache cost explodes quadratically.
- **Vertex C · Scale (trillion-MoE)**: to raise capability you must stack 1T+ params, which is expensive to train and serve.

The naive approach (standard softmax attention + dense model) sits in the center and reaches none of the vertices — long context OOMs, big models burn money. All three break the deadlock by **reshaping attention + going MoE**, just with different levers:

```
            Long context (1M)
               /      \
          Redesign    Compress
          attention   attention
             |          |
   Kimi K3   |    DeepSeek V4
   (KDA+MLA) |    (CSA + HCA)
             \          /
          Sparse attention
             MiniMax M3
              (MSA)
               \      /
              Scale (MoE)
```

<div class="keybox">
<strong>In one line:</strong> the 2026 frontier race is not "who has more params" but "who can reshape attention to hold 1M context without blowing up KV and compute." The three picked <strong>redesign / sparsify / compress</strong> respectively.
</div>

## 2. The Three Routes at a Glance

| Model | Released | Core lever | Context | Scale | Open |
|---|---|---|---|---|---|
| **Kimi K3** | 2026 | Redesign attention (KDA + Gated MLA 3:1) + Stable LatentMoE + native vision | 1M | 2.8T MoE | weights open |
| **MiniMax M3** | 2026-06 | Sparse attention (MSA) + native multimodal + computer use | 1M (M2.7 only 200K) | undisclosed (MoE) | weights open |
| **DeepSeek V4** | 2026-04 | Compress attention (CSA + HCA) + Muon optimizer + OPD distillation | 1M (default) | Pro 1.6T / Flash 284B (MoE) | MIT open |

- **Kimi K3** — "redesign attention": Kimi Delta Attention (KDA) on top of Gated MLA (3:1 mix), plus Attention Residuals for stable training; MoE uses Stable LatentMoE (896 routed experts, 16 active), vision uses from-scratch MoonViT-V2 for native multimodality. Official: **~2.5× scale efficiency** vs K2.5.
- **MiniMax M3** — "sparsify attention": on top of M2.7 (standard attention, 200K, text-only, API-only), swaps in MiniMax Sparse Attention (MSA); at 1M context **per-token compute is ~1/20 of M2.7**, prefill ~9× and decode ~15× faster; adds native multimodal (text+image+video) and computer use, and open-weights.
- **DeepSeek V4** — "compress KV to the extreme": CSA (Compressed Sparse Attention, 4-token KV blocks + top-k) + HCA (Heavily Compressed Attention, 128×) hybrid, plus a 128-token uncompressed sliding-window branch; at 1M **per-token compute ~27% of V3.2, KV cache ~10%**; training swaps in the Muon optimizer for AdamW and uses OPD (On-Policy Distillation) of domain experts.

## 3. The Progressive Roadmap

The series is deliberately ordered by "strength of attention modification," light to heavy:

1. **(2) Kimi K3**: add a delta term (KDA) on MLA, mix MLA and a gate at 3:1 — "gentle attention change + redo MoE/vision," easiest to read, good entry point.
2. **(3) MiniMax M2.7→M3**: first see M2.7's standard-attention baseline, then how M3 sparsifies it — a live demo of dense→sparse, the most直观 illustration of sparsity's payoff.
3. **(4) DeepSeek V4**: compress KV itself (CSA/HCA), then stack Muon, OPD, FP8/FP4 — most aggressive, heaviest engineering, saved for the finale.

Read in order and a主线 emerges: **dense attention → sparse attention → compressed attention**, and all three converge on "MoE + 1M context" as the shared destination.

## 4. The Series Thread & Investment View

The commonalities matter more than the differences:

- **All go MoE + very long context**: the dense-LLM era is essentially over; 1M context is the new baseline.
- **All reshape attention, not just add hardware**: the breakthrough is at the "attention mechanism" layer, not more GPUs.
- **All open weights** (MIT / Apache): frontier capability is being commoditized; differentiation shifts from "model capability" to "engineering, data, ecosystem."

> Investment view: the "Moore's law of attention" (compute/KV per same context keeps dropping year over year) directly benefits **inference cost reduction** — every step down makes on-device / onboard real-time VLA and long-horizon Agents more feasible, exactly the cost inflection point for embodied AI and AI apps. Next: Kimi K3's "redesigned attention," taken apart model by model.
