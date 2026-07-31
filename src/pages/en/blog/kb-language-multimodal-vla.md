---
title: "Generative Language & Multimodal Paradigms: from GPT-1 to VLA and World Models"
description: One thread ties together the evolution of the model "brain"—GPT-1 establishes generative pretraining, LLaVA brings multimodal instruction tuning, RT-2 turns vision-language into robot actions, and π0.7 / HiF-VLA fuse VLA with world models (WAM).
pubDate: 2026-07-31
series: AI Knowledge Base
lang: en
altLang: zh
altHref: /blog/kb-language-multimodal-vla
layout: ../../../layouts/BlogPost.astro
---

## 1. The thread: from "language generation" to "embodied action"

This post strings several foundational works into one clear line:

> Generative pretraining (can "write" language) → Multimodal instruction tuning (can "see + talk") → Vision-Language-Action models / VLA (can "act") → VLA + world model (can "think before acting").

For investors: the "brain" of embodied AI did not appear from nowhere—it reuses the already-validated LLM/VLM pretrain+finetune paradigm; the only change is that the output space expands from "text tokens" to "action tokens".

## 2. GPT-1: the seed paradigm of generative pretrain+finetune (Radford et al., 2018)

- **One-liner**: first showed that "generative pretraining + downstream finetuning" can sweep NLU tasks, writing the entire DNA of later ChatGPT.
- **Core formula**: causal LM on BookCorpus `L₁ = Σ log P(uᵢ | u<i; Θ)`, then migrate `Θ` and add a linear head `W_y`, with `λ·L₁` to prevent forgetting.
- **Architecture**: Decoder-only Transformer, 12 layers, d=768, h=12, ctx=512, 117M params (small, but the route is complete).
- **Impact**: established the GPT line → GPT-2/3/4/ChatGPT; rivaled BERT (bidirectional mask); generative capability ultimately won.

## 3. LLaVA: bringing instruction tuning to multimodality (NeurIPS 2023 Oral)

- **Core innovation**: used GPT-4 to "imagine" 158K multimodal instruction samples from COCO captions (dialog/description/reasoning, zero human annotation).
- **Minimal architecture**: frozen CLIP ViT + one trainable linear projection `W` + Vicuna LLM, two-stage training (align features, then instruction tune).
- **Results**: GPT-4-as-Judge relative score 85.1%, ScienceQA 92.53% SOTA.
- **Paradigm impact**: established the "frozen vision encoder + projection bridge + LLM finetune" standard VLM recipe, spawning LLaVA-1.5/NeXT, CogVLM, InternVL, Qwen-VL.

## 4. RT-2: the first large-scale VLA, VLM outputs robot actions (Google DeepMind, 2023)

- **Core idea**: discretize 7-DoF continuous actions into **text tokens** (256-bin), sharing the output space with language; autoregressively generate an action string then decode (co-fine-tuning).
- **Result**: unseen-task success rate RT-1 32% → RT-2 62%.
- **Significance**: founded the OpenVLA / π0 / GR00T VLA line—the essence of the "robot brain" is "generate actions like language".

## 5. π0.7 and HiF-VLA: VLA fused with world models (WAM)

- **π0.7 (Physical Intelligence)**: first integrates a **world model as a Subgoal Image Provider** into VLA—predict a subgoal image "in mind" before acting, then execute toward it, fixing pure-VLA "hallucinated actions". ARX arm's folding skill transferred **zero-shot** to UR5 dual-arm (cross-embodiment).
- **HiF-VLA (Westlake MiLAB, CVPR 2026)**: motion-centric bidirectional spatiotemporal reasoning; a joint expert predicts "future motion + action" together. Memory 31.4GB (baseline 63.6), latency 117.7ms (baseline 229.5), LIBERO-LONG SOTA.
- **Endgame (WRAM)**: VLA and world model "fuse and co-exist" is the embodied endgame; data remains the biggest bottleneck (off by 1~2 orders of magnitude).

## 6. Practical takeaways

- When evaluating embodied companies/models, watch three things: ① do they reuse mature LLM/VLM pretraining (not reinvent); ② does the VLA use a world model to "think before acting"; ③ cross-embodiment / zero-shot transfer (decides mass-production replicability).
- This thread also explains why "LLM cost reduction" directly helps embodied AI: every drop in inference cost makes on-device / on-robot real-time VLA more feasible.
