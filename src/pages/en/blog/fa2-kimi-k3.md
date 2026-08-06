---
title: 'Frontier Architecture Decoding Notes (2): Kimi K3 — Dual Evolution of Attention and MoE'
description: 'Following the overview roadmap, the first model torn apart is Kimi K3. Instead of stacking compute, it redesigns the attention mechanism: Kimi Delta Attention (KDA) adds a delta term on MLA, Gated MLA mixes at 3:1, Attention Residuals stabilize training; the MoE is Stable LatentMoE (896 routed / 16 active); vision uses from-scratch MoonViT-V2 for native multimodality. Official scale efficiency is ~2.5× vs K2.5.'
pubDate: 2026-08-06
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa2-kimi-k3
layout: ../../../layouts/BlogPost.astro
---

## 0. Why Start With Kimi K3

As the overview said, Kimi K3 picked the gentlest and most readable path — "redesign attention." Compared with M2.7→M3's "sparsify" and DeepSeek V4's "compress," its changes sit closest to existing architectures, making it the natural **first stop** of the progressive arc.

## 1. Attention: a "Delta Term" on MLA

Kimi K3's attention core is **Kimi Delta Attention (KDA)**, built on MLA (Multi-head Latent Attention):

- **MLA baseline**: compress Key/Value into low-rank latent vectors, greatly saving KV cache — the approach DeepSeek popularized and many now adopt.
- **KDA's addition**: on top of MLA it stacks a **delta term** that makes attention more sensitive to "the relative change between the current query and historical positions," easing the standard-attention failure where distant info is drowned by nearby tokens in long context.
- **Gated MLA (3:1)**: mixes a "standard attention path" with the "MLA latent path" at a **3:1 gate** — keeps MLA's KV-saving advantage while using the standard path to recover some long-range modeling.

With **Attention Residuals**: the previous layer's attention output is added as a residual into the current layer, easing attention-signal decay in deep nets and stabilizing 1M-context training.

<div class="keybox">
<strong>Key:</strong> KDA does not overthrow MLA; it adds a "delta + gate(3:1) + residual" trio on top, aiming to <strong>make long-context modeling solid without much extra KV</strong>.
</div>

## 2. MoE: Stable LatentMoE

Kimi K3 is a 2.8T-param MoE; the point is "stable":

- **896 routed experts, 16 active per token** — many experts, few active, large capacity at controllable per-step cost.
- **SiTU-GLU activation**: a Situ-style GLU inside experts, balancing expressiveness and numerical stability.
- **Quantile Balancing**: uses quantile statistics for load balancing instead of an auxiliary loss, easing the "expert drought/flood" training jitter — this is where "Stable" comes from.

## 3. Vision: MoonViT-V2 Native Multimodality

Kimi K3's vision is not "bolt on a CLIP" but **MoonViT-V2** — a vision encoder **trained from scratch with next-token-prediction as its objective**:

- it shares the same generative objective as the language model, so image tokens and text tokens are predicted in one unified space;
- "seeing" and "speaking" are natively fused rather than spliced later, making interleaved image-text smoother in long context.

## 4. Infrastructure: MoonEP / FlashKDA / AgentEnv

- **MoonEP**: Expert-Parallelism communication optimization, supporting 896-expert large-scale MoE train/serve.
- **FlashKDA**: a Flash-style fused kernel for the KDA operator, lowering memory-bandwidth pressure.
- **AgentEnv**: workload optimization for Agent / tool-call scenarios, fitting long-horizon tasks under 1M context.

## 5. Results and Positioning

Official: **~2.5× scale efficiency** vs K2.5 — capability climbs faster per unit compute/data. 2.8T MoE + 1M context + native vision positions it as a general frontier model that "reads, sees, and reasons over long horizons."

> Sourcing note: the core architecture points come from the Moonshot AI official tech report (PDF) and the official WeChat release explainer; the tech-report PDF was not machine-parsed in this site's environment, so defer to the official release for details.

## 6. Investment View

- "Redesign attention + stable MoE + native vision" is an upgrade path **not dependent on extreme hardware**: same capability at lower compute, good for **inference cost reduction** and **domestic compute adaptation**.
- 2.8T MoE open-weights means frontier multimodal capability is further commoditized; the moat for the application layer (Agents, embodiment, long-doc/code understanding) **shifts from model to data and scenario**.
- Next: MiniMax, which takes a different road from KDA — sparsifying attention directly.
