---
title: 'VLA Decoding Notes (2): The Action Engine — Diffusion Policy, Flow Matching, Low Latency'
description: 'Why high-performance VLAs use continuous generation instead of discrete tokens. We dissect Diffusion Policy, Flow Matching, Action Chunking, and how Xiaomi-Robotics-0 uses MoT + async inference to hit 80ms latency and run real-time on a 4090.'
pubDate: 2026-08-04
series: VLA Decoding Notes
lang: en
altLang: zh
altHref: /blog/vla2-action-generation
layout: ../../../layouts/BlogPost.astro
---

## 1. Why Discrete Tokens Fall Short

As noted, quantizing continuous actions into autoregressive tokens **truncates precision** and stutters. Robots need high-frequency, smooth, perturbation-sensitive continuous control — exactly what diffusion / flow matching deliver.

## 2. Diffusion Policy: Denoise Actions Like an Image

Diffusion Policy (Chi et al., 2023) borrows image-generation diffusion:

- Training adds noise to action sequences; the network learns "recover action from noise";
- Inference starts from random noise and denoises over steps into a **multimodal, robust** trajectory.

It naturally supports multi-modal distributions and resists noise in demonstration data.

## 3. Flow Matching: Faster and More Stable Than Diffusion

Flow Matching does not model a complex probability path; it learns the mapping of a **probability flow**, pushing a simple distribution straight toward the target action distribution.

- Simpler training objective, steadier gradients;
- Far fewer inference sampling steps than DDPM (tens to hundreds) — down to **~5 steps**;
- π0 and Xiaomi both use Flow Matching as the action expert's core.

> The "flow matching" we covered in the speculative-decoding notes here lands as **physical action**, not text tokens — the same math tool, reused across modalities.

## 4. Action Chunking: Predict a Whole Block at Once

Deciding every millisecond is slow and jittery. Action Chunking (from ACT / π0) lets the model **predict a short block of actions** at once:

- Lower decision frequency, smoother motion;
- Intra-block autoregressive/diffusion generation, inter-block prefix from history;
- π0 uses Flow Matching + 50Hz chunks — the "smooth + high-frequency" recipe.

## 5. Xiaomi's Play: MoT Loosely-Coupled + Async Inference

Xiaomi-Robotics-0 (open-sourced 2026-02, 4.7B) is a low-latency VLA exemplar:

- **MoT**: VLM "brain" understands; 16-layer **DiT (Diffusion Transformer) "cerebellum"** generates action blocks. Loosely coupled via **KV Cache** — brain output feeds the cerebellum, no recompute.
- **Flow Matching training**: sampling steps cut from DDPM's tens to 5.
- **Async inference**: model inference and robot execution are **decoupled** — latency no longer stalls real-machine continuity, killing "action断层" at the mechanism level.
- **Clean Action Prefix + Λ-shape attention**: prior-step actions as input keep temporal continuity; a special mask makes the model watch current visual feedback, reacting sharply to sudden changes.

Result: **80ms latency, 30Hz control, real-time on RTX 4090**, SOTA on LIBERO / CALVIN / SimplerEnv.

<div class="fig">
  <svg viewBox="0 0 680 190" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <rect x="10" y="60" width="150" height="70" rx="10" fill="#E8F0FE" stroke="#4285F4"/>
    <text x="85" y="95" text-anchor="middle" font-size="13" fill="#1a1a1a">VLM brain</text>
    <text x="85" y="113" text-anchor="middle" font-size="11" fill="#555">understand</text>
    <rect x="200" y="60" width="150" height="70" rx="10" fill="#FEF7E0" stroke="#FBBC04"/>
    <text x="275" y="95" text-anchor="middle" font-size="13" fill="#1a1a1a">KV Cache</text>
    <text x="275" y="113" text-anchor="middle" font-size="11" fill="#555">loose coupling</text>
    <rect x="390" y="60" width="150" height="70" rx="10" fill="#E6F4EA" stroke="#34A853"/>
    <text x="465" y="95" text-anchor="middle" font-size="13" fill="#1a1a1a">DiT cerebellum</text>
    <text x="465" y="113" text-anchor="middle" font-size="11" fill="#555">action block</text>
    <rect x="560" y="60" width="110" height="70" rx="10" fill="#FCE8E6" stroke="#EA4335"/>
    <text x="615" y="95" text-anchor="middle" font-size="13" fill="#1a1a1a">robot</text>
    <text x="615" y="113" text-anchor="middle" font-size="11" fill="#555">30Hz async</text>
    <line x1="160" y1="95" x2="198" y2="95" stroke="#888" stroke-width="2" marker-end="url(#b)"/>
    <line x1="350" y1="95" x2="388" y2="95" stroke="#888" stroke-width="2" marker-end="url(#b)"/>
    <line x1="540" y1="95" x2="558" y2="95" stroke="#888" stroke-width="2" marker-end="url(#b)"/>
    <defs><marker id="b" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#888"/></marker></defs>
  </svg>
  <p class="fig-cap">Fig: Xiaomi MoT — brain and cerebellum loosely coupled via KV Cache, driving the robot asynchronously</p>
</div>

## 6. Bridge

The action-generation mechanism is clear. Next: how the π family evolved with it — from π0's Flow Matching 50Hz to π0.7's memory and compositional generalization.
