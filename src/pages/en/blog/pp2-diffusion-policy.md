---
title: Diffusion Policy Deep Read — Porting Stable Diffusion's Denoising to Robot Action Generation
description: A close read of Diffusion Policy (Chi et al., 2023, MIT/Columbia) — replacing autoregressive/GMM action heads with a conditional diffusion model for multimodal, high-dimensional, stable 6-DoF action generation; SOTA on 14 of 15 benchmarks.
pubDate: 2026-08-12
series: Paper Primer Notes
lang: en
altLang: zh
altHref: /blog/pp2-diffusion-policy
layout: ../../../layouts/BlogPost.astro
---

## 1. The One-Line Takeaway

**The core idea of Diffusion Policy: stop predicting the robot's next action with "1-D autoregressive + Gaussian mixture"; instead treat the action sequence itself as an "image" and use the exact same diffusion-denoising paradigm as Stable Diffusion to "imagine" and generate a trajectory.** The result: SOTA on 14 of 15 manipulation benchmarks, with a particular edge on scenes where "one observation admits many valid actions".

> This is the same topic as, but a different solution from, this blog's "VLA Notes (2) The Action Engine" — there we placed Diffusion Policy in the landscape of action generation; here we go back to the paper itself.

## 2. How Was Action Generated Before? The Pain

A robot policy `π(aₜ | oₜ)` maps observation `oₜ` (image/state) to action `aₜ`. Mainstream approaches:

- **Autoregressive (GPT-style)**: actions as tokens, generated one by one. Problem: error compounds, slow, poor at multimodality.
- **Gaussian Mixture Model (GMM)**: one observation yields a weighted sum of a few Gaussians. Problem: limited freedom; high-dimensional 6-DoF actions (pose+orientation) are hard to fit, and it tends to "average" into a blurred action that matches nothing.

The deeper trouble is **multimodality**: seeing a cup, the robot may "grasp from the left" or "from the right" — both valid. GMM tends to average the two into one odd intermediate motion.

## 3. What Diffusion Policy Does

It borrows a **conditional diffusion model (DDPM family)**:

- Treat the future action chunk `A = [aₜ, aₜ₊₁, …, aₜ+H]` as a high-dimensional vector (analogous to an image).
- Training: progressively add noise to the real action `A₀` to get `Aₖ` (pure noise at step `k`), teaching the network to "given observation `oₜ`, predict the added noise `ε` from `Aₖ`".
- Inference: **start from pure random noise `A_K`, iteratively denoise for `K` steps, obtain a clean trajectory, and execute it immediately** (usually the first few steps, then re-observe and re-diffuse — receding-horizon / action chunking).

The network is typically a **CNN (1D/2D U-Net) or DiT (Diffusion Transformer)**; the condition `oₜ` is injected at every denoising step via FiLM / Cross-Attention.

## 4. Why Diffusion Fits Actions Especially Well

| Dimension | GMM / Autoregressive | Diffusion Policy |
|-----------|----------------------|------------------|
| Multimodal | Finite peaks only | Latent space expresses complex multimodal naturally |
| High-dim | 6-DoF blurs/averages | Denoise whole trajectory as image, preserves detail |
| Training | Mode collapse / averaging | Simple (predict noise) target, stable |
| Uncertainty | Implicit | Explicit: resampling yields varied valid actions |

Analogy: GMM is "forcing a cat out of a few bell curves"; diffusion is "start from noise, erase into a cat step by step" — far friendlier to complex shapes.

## 5. Key Design Choices (the paper's tricks)

1. **Action Chunking**: generate `H` future steps (e.g. 8–16) at once, not single steps — smooth, jitter-resistant, hides execution latency.
2. **Observation conditioning**: visual encoding injected via FiLM/Cross-Attention into every U-Net layer, ensuring "see before act".
3. **Time-axis handling**: the action sequence is laid along the time axis and fed to a 1D-conv U-Net; denoising is "along time".
4. **Few-step sampling**: later work (DDPO, consistency) compresses `K` from dozens to a few steps for real-time control.

## 6. Why the Results Shine

The paper compares across **4 families / 15 tasks** (sim + real, rigid/cloth/liquid):

- Versus prior SOTA (incl. GPT-style, and other diffusion conditionings): **leading on 14 of 15**, with clear average success gains.
- The edge is largest exactly where GMM/autoregressive are weak: multimodal demos, high-D pose, temporally coordinated tasks.

## 7. Relation to Transformer / VLA

- Diffusion Policy solves the **action head**; the "seeing" and "understanding" can still be done by ViT/LLM (the V+L in VLA).
- Hence it commonly serves as the **action engine of VLA** (RT-2, π0, HiF-VLA), alongside the autoregressive action-head route.
- "Denoising" and Transformer's "attention" are the two great foundations of modern generative AI — one governs "parallel computation of relations", the other "carving structure out of noise".

## 8. Summary

Diffusion Policy = treat action sequence as "image" + conditional diffusion denoising + Action Chunking. It uses generative modeling's multimodal/high-dimensional expressiveness to fix the traditional policy head's weakness on "complex, multi-solution, high-dimensional actions", becoming one of the most practical action-generation paradigms in robot learning since behavioral cloning.

> To go deeper on engineering trade-offs (diffusion vs autoregressive for actions), return to "VLA Notes (2) The Action Engine".
