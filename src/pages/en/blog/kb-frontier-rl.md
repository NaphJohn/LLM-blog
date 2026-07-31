---
title: "Frontier Inference & RL: OAT / Ring-Zero / SPS / HiLS-Attention / Exploration-Paradox RL / AMVL"
description: A batch of "frontier small papers" from summer 2026—OAT traces Agent failures backward from success trajectories; Ring-Zero pushes Zero-RL to 1T params; SPS separates state-prediction for efficiency; HiLS-Attention extrapolates 8K training to 4M; plus exploration-paradox RL and AMVL latent reasoning. A quick tour of the core ideas.
pubDate: 2026-07-31
series: AI Knowledge Base
lang: en
altLang: zh
altHref: /blog/kb-frontier-rl
layout: ../../../layouts/BlogPost.astro
---

## 1. OAT: tracing Agent failures from "streams of success" (arXiv:2607.12747)

- **Problem**: when a long-horizon Agent errs, you often only know "the final result is wrong", not "which step went off".
- **Approach**: unsupervised, lightweight (Neural CDE) method that learns "where failure happens" **backward from success trajectories**, no human-labeled failure samples needed.
- **Advantage**: detects trajectory drift **earlier** than behavior monitoring.
- **Embodied mapping**: transferable to "robot manipulation failure localization"—improving debuggability of autonomous systems, a key piece of the "robot autonomous decision" reliability in the embodied investment thesis.

## 2. Ring-Zero: Zero-RL pushed to 1T params (Renmin Univ × Ant, 2607.12395)

- **Finding**: pushing Zero-RL (no SFT cold-start, learn reasoning directly from RL) to **1T params** makes **advanced reasoning strategies emerge spontaneously**.
- **Significance**: further validates that "scale + rule-based rewards" can elicit complex reasoning, weakening reliance on massive SFT data.

## 3. SPS: state-prediction separation (Cornell × Harvard, 2607.01218)

- **Core**: decouple "state representation" from "future prediction" in training, **2.6× training efficiency**.
- **Significance**: decoupled representation and prediction are easier to reuse and scale—an engineering idea for efficient world models / oracles.

## 4. HiLS-Attention: 8K training extrapolated to 4M (Tencent Hunyuan, 2607.02980)

- **Core**: an attention improvement letting a model trained on **8K context** extrapolate to **4M**, **13.5× faster prefill**; open-sourced.
- **Significance**: a "train short, use long" cost-effective route, directly helping ultra-long document/code/multimodal-sequence scenarios.

## 5. Exploration-Paradox RL (ByteDance Seed × MSU, 2607.06987)

- **Core**: corrects **importance-sampling bias** in exploration, making RL exploration more efficient and stable.
- **Significance**: a detail but important—entropy collapse / insufficient exploration in long-horizon reasoning RL is a common pitfall; this gives a systematic fix.

## 6. AMVL: latent-space continuous reasoning (SJTU × Ant, 2607.00461)

- **Core**: do continuous reasoning in **latent space**, BLINK benchmark **+10.83**.
- **Significance**: echoes the "diffusion / continuous representation for reasoning" route (also see block-diffusion drafting like DFlash), complementary to pure discrete-token reasoning.

## 7. Practical takeaways

- Together these works point to: **reasoning capability is shifting from "scaling params" to "better training signals + better long-context / latent-space representation"**.
- For investing: progress in inference efficiency (OAT debuggability, SPS/HiLS efficiency, AMVL latent) is the underlying variable deciding LLM service cost and on-device embodied feasibility.
