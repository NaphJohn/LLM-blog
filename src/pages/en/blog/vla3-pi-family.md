---
title: 'VLA Decoding Notes (3): The π Family — Physical Intelligence’s VLA Lineage'
description: 'From π0 to π0.7, how Physical Intelligence pushed VLA toward "better than specialist policies". We walk the timeline: PaliGemma / Gemma3 backbones, Flow Matching 50Hz, cross-embodiment, RL specialization, and π0.7 memory + compositional generalization.'
pubDate: 2026-08-04
series: VLA Decoding Notes
lang: en
altLang: zh
altHref: /blog/vla3-pi-family
layout: ../../../layouts/BlogPost.astro
---

## 1. π0: A "Universal Interface" for VLA (2024-10)

Physical Intelligence's **π0** set the baseline:

- **Backbone**: PaliGemma (~3B VLM) + a separate **action expert** using **Flow Matching** for continuous actions at **50Hz**.
- **Data**: trained on OXE + proprietary demos, covering **7 robot platforms, 68 tasks**.
- **Unified interface**: language/image in, continuous action out. Proof that "one general VLA can work across embodiments".

## 2. π0-Fast: Discrete Speedup

Adds **action discretization + autoregressive** on top of π0 for inference speed — an engineering trade-off between the continuous vs discrete routes, for latency-sensitive settings.

## 3. π0.5: Cross-Embodiment + Task Decomposition (2025-04)

- **Cross-embodiment co-training**: multiple different robots share one "general brain", learning hardware-agnostic manipulation common sense.
- **Task decomposition**: a high level splits complex instructions into sub-goals; low-level policies execute them in turn — making long-horizon tasks tractable.

## 4. π0.6*: RL Specialization (RECAP)

Uses **reinforcement learning** to specialize the trained VLA (RECAP-style), pushing specific-task success rates further. The typical second half of "general pre-train + specialist post-train".

## 5. π0.7: Memory + Compositional Generalization (2026-04)

π0.7 is the current apex:

- **Backbone upgrade**: Gemma3 4B + 860M action expert.
- **MEM memory**: learns from past experience, carries contextual memory for steadier long multi-step tasks.
- **Multimodal prompts**: image / language / demonstration mixed prompts — "do it like the demo".
- **Emergent compositional generalization**: generalizes to unseen instruction combinations — a signal of moving toward "truly general".
- **Result**: beats **specialist policies** on several benchmarks, not just matching them.

<div class="fig">
  <svg viewBox="0 0 680 150" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="20" y="30" font-size="13" fill="#1a1a1a" font-weight="bold">π family evolution</text>
    <line x1="40" y1="80" x2="640" y2="80" stroke="#ccc" stroke-width="2" marker-end="url(#c)"/>
    <g font-size="11" text-anchor="middle">
      <circle cx="90" cy="80" r="7" fill="#4285F4"/><text x="90" y="110">π0</text><text x="90" y="125" font-size="9" fill="#555">FlowMatch 50Hz</text>
      <circle cx="200" cy="80" r="7" fill="#4285F4"/><text x="200" y="110">π0-Fast</text><text x="200" y="125" font-size="9" fill="#555">discretize</text>
      <circle cx="320" cy="80" r="7" fill="#34A853"/><text x="320" y="110">π0.5</text><text x="320" y="125" font-size="9" fill="#555">cross-embod.</text>
      <circle cx="440" cy="80" r="7" fill="#FBBC04"/><text x="440" y="110">π0.6*</text><text x="440" y="125" font-size="9" fill="#555">RL spec.</text>
      <circle cx="560" cy="80" r="7" fill="#EA4335"/><text x="560" y="110">π0.7</text><text x="560" y="125" font-size="9" fill="#555">memory+comp.</text>
    </g>
    <defs><marker id="c" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#ccc"/></marker></defs>
  </svg>
  <p class="fig-cap">Fig: The π lineage — backbone upgrade, cross-embodiment, RL specialization, memory + compositional generalization</p>
</div>

## 6. The Through-Line (One Line)

> VLM backbone upgrade → cross-embodiment reuse → RL specialization → memory + compositional generalization. **General VLA is moving from "can work" to "stronger than specialists".**

## 7. Bridge

The π family is the overseas benchmark. Next: domestic players — Ant Lingbo's LingBot-VLA 2.0 "one brain, many machines", Xiaomi's open real-time VLA, Tencent HyVLA's FlowPRO — plus a clarification of a common mix-up.
