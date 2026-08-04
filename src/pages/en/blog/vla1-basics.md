---
title: 'VLA Decoding Notes (1): From "See" to "Act" — What Is a Vision-Language-Action Model'
description: 'The core of embodied AI is extending "understanding" into "action". This piece defines VLA (Vision-Language-Action) models, why they emerged, the paradigm shift from task-specific policies to native VLA, and the two routes for representing actions (discrete tokens vs continuous generation).'
pubDate: 2026-08-04
series: VLA Decoding Notes
lang: en
altLang: zh
altHref: /blog/vla1-basics
layout: ../../../layouts/BlogPost.astro
---

## 1. Why VLA

The "Multimodal Decoding Notes" covered VLM — models that can **see** an image and **say** what is in it. But many real tasks demand not just "say" but **act**: put the bowl on the rack, fold the towel, slot the part into place.

A **VLA (Vision-Language-Action) model** wires those three things into one chain:

> Input = image / video (what is seen) + language instruction (what to do)
> Output = **continuous robot actions** (where to go, how to move, how much force)

It is the operational core of **Embodied AI** — letting a model generate not just text, but control commands that change the physical world.

<div class="fig">
  <svg viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <rect x="10" y="70" width="120" height="60" rx="10" fill="#E8F0FE" stroke="#4285F4"/>
    <text x="70" y="105" text-anchor="middle" font-size="14" fill="#1a1a1a">Image / Video</text>
    <rect x="160" y="70" width="120" height="60" rx="10" fill="#E6F4EA" stroke="#34A853"/>
    <text x="220" y="105" text-anchor="middle" font-size="14" fill="#1a1a1a">Language</text>
    <rect x="320" y="60" width="140" height="80" rx="10" fill="#FEF7E0" stroke="#FBBC04"/>
    <text x="390" y="100" text-anchor="middle" font-size="14" fill="#1a1a1a">VLA model</text>
    <text x="390" y="120" text-anchor="middle" font-size="11" fill="#555">VLM + action head</text>
    <rect x="510" y="70" width="150" height="60" rx="10" fill="#FCE8E6" stroke="#EA4335"/>
    <text x="585" y="100" text-anchor="middle" font-size="14" fill="#1a1a1a">Robot action</text>
    <text x="585" y="120" text-anchor="middle" font-size="11" fill="#555">continuous ctrl</text>
    <line x1="130" y1="100" x2="158" y2="100" stroke="#888" stroke-width="2" marker-end="url(#a)"/>
    <line x1="280" y1="100" x2="318" y2="100" stroke="#888" stroke-width="2" marker-end="url(#a)"/>
    <line x1="460" y1="100" x2="508" y2="100" stroke="#888" stroke-width="2" marker-end="url(#a)"/>
    <defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#888"/></marker></defs>
  </svg>
  <p class="fig-cap">Fig: VLA upgrades "see + say" into "see + say + do"</p>
</div>

## 2. The Paradigm Shift

Robot control has walked several paths:

1. **Task-specific policy**: hand-written planning + perception per task. Poor generalization.
2. **Imitation learning (BC / DAgger)**: learn "state → action" from demos. Breaks across embodiments.
3. **VLM retrofit**: bolt an action head onto a VLM. Cheap, but understanding and action are disjointed.
4. **Native VLA (joint V+L+A pre-training)**: vision, language, **action** pre-trained in one representation. The current mainstream (π0, OpenVLA, Xiaomi-Robotics-0).

> Key point: VLA is not "train yet another big model" — it extends the common sense and generalization that LLM/VLM already have, via the alignment paradigm from earlier notes, into the physical action space.

## 3. How to Represent Actions: Two Routes

Continuous actions (6–7 DoF arm angles, base velocity) are not text tokens. Two encodings dominate:

- **Discrete tokenization**: quantize actions into tokens, autoregressively generated. Simple and reuses the LLM stack, but quantization **truncates precision** and the trajectory stutters — at high frequency the robot acts like a "slow wooden dummy".
- **Continuous generation (Diffusion / Flow Matching)**: regress the **continuous action distribution** directly, producing smooth, high-frame-rate vectors. The choice of high-performance models like π0 and Xiaomi.

## 4. The Data Bottleneck: OXE and Cross-Embodiment

VLA's "ingredients" are robot demonstration data. The key open corpus is **Open X-Embodiment (OXE)** — demos from dozens of labs and many robot embodiments, merged into a cross-embodiment dataset so the model learns hardware-agnostic manipulation common sense.

But the real bottleneck: language models ate trillions of tokens, video models ate billions of clips, while top companies have only **hundreds of thousands of hours** of high-quality physical interaction — at least an order of magnitude short of validating a scaling law (WAIC 2026 consensus).

> This echoes the earlier note that "data construction is often the ceiling": VLA competition is short-term about architecture, long-term about **who runs the data flywheel first**.

## 5. Bridge

This piece gave VLA its skeleton. Next: the "engine" of action generation — Diffusion Policy, Flow Matching, Action Chunking, and how Xiaomi crushed latency to 80ms. Then the full π-family evolution, and domestic players (Ant Lingbo, Xiaomi, Tencent).

> Investment angle (echoing your portfolio framing): embodied AI is shifting from "showing off" to "competing on brains". The real moat is **model system + real-scene data loop**. Back companies with an engineering loop, not single-parameter hype.
