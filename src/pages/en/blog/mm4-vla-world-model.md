---
title: 'Multimodal Decoding Notes (4): From VLA to World Models — RT-2 → π0.7 → HiF-VLA'
description: 'Turning "see + speak" into "act": the Vision-Language-Action model (VLA) discretizes actions into tokens so a VLM directly outputs robot actions. But pure VLA "hallucinates actions"; π0.7 uses a world model as a subgoal-image provider, and HiF-VLA does motion-centric bidirectional spatiotemporal reasoning, letting the model "think then act". This is the endgame direction for embodied intelligence.'
pubDate: 2026-07-31
series: Multimodal Decoding Notes
lang: en
altLang: zh
altHref: /blog/mm4-vla-world-model
layout: ../../../layouts/BlogPost.astro
---

## 1. From "Understanding" to "Acting": What Is VLA

Parts 1-3: (1) foundations & alignment → (2) VLM (see + speak) → (3) generation (draw). This chapter connects all that to **physical action**.

**VLA (Vision-Language-Action)**: input "vision + language instruction", output **robot action**. Its core idea is the landing point of Part (1)'s main thread — **treat action itself as a kind of token to be generated.**

## 2. RT-2: The First Large-Scale VLA, Letting a VLM Output Actions Directly (Google DeepMind, 2023)

- **Core idea**: discretize 7-DoF continuous actions into text tokens at 256-bin, sharing the output space with language; autoregressively generate the action string then decode and execute (co-fine-tuning).
- **Result**: unseen-task success rate went from RT-1's 32% to RT-2's 62%.
- **Significance**: established the subsequent VLA lineage of OpenVLA / π0 / GR00T — **the essence of a "robot brain" is "generating action as if it were language"**. This closes the loop on Part (1)'s claim: multimodality = extrapolating the generation space from text to action.

## 3. The Fatal Flaw of Pure VLA: It "Hallucinates Actions"

After RT-2, the field found: on long-horizon, fine-grained tasks, pure VLA generates **actions that look plausible but are physically unexecutable** (object doesn't exist, trajectory clips through geometry). Root cause — VLA only learns the "action distribution" without an **internal model of how the physical world evolves**.

## 4. π0.7: Using a World Model as a "Subgoal-Image Provider" (Physical Intelligence)

- **Core做法 (core approach)**: for the first time integrates a **world model (World Model) as a Subgoal Image Provider** into VLA — before acting, first predict a **subgoal image** ("what the world should look like next") inside its "mind", then use it to guide action execution.
- **What it solves**: "think through the next world state, then act" replaces "blindly generate actions", significantly overcoming action hallucination.
- **Zero-shot cross-embodiment**: the folding skill learned on an ARX arm transferred **zero-shot to a UR5 dual-arm** — evidence that "world-model-guided" learning captures transferable general skills rather than overfitting to one robot.

## 5. HiF-VLA: Motion-Centric Bidirectional Spatiotemporal Reasoning (Westlake MiLAB, CVPR 2026)

- **Core**: Motion-centric bidirectional spatiotemporal reasoning; a joint expert simultaneously predicts "future Motion + action".
- **Efficiency**: memory 31.4GB (baseline 63.6), latency 117.7ms (baseline 229.5), LIBERO-LONG SOTA.
- **Significance**: explicitly co-modeling action and world evolution is more stable and faster than pure action generation.

## 6. The Endgame Consensus (WRAM): VLA and World Model "Fuse and Coexist"

- The field is converging: **pure VLA is not enough; VLA + world model (WAM, World-Action Model) is the endgame for embodied intelligence.**
- The world model's role is not only "subgoal image": it also enables **predictive planning / failure anticipation / simulation-style data augmentation**, easing the data bottleneck (below).
- **Data remains the biggest bottleneck**: high-quality "image-text-action" pairs are 1~2 orders of magnitude short, and the world model can precisely generate large amounts of synthetic training signal by "rolling out in its internal world" — another motive for the VLA + world-model fusion.

## 7. Practical Takeaways

When evaluating an embodied company / model, focus on three points (echoing Part 1's framework):

1. **Does it reuse a mature LLM / VLM pre-training paradigm** (rather than reinventing one) — decides engineering maturity;
2. **Does the VLA introduce a world model for "think then act"** — decides long-horizon reliability;
3. **Cross-embodiment / zero-shot transfer ability** — decides mass-production replicability (π0.7's ARX→UR5 is key evidence).

> This thread also explains why "LLM cost reduction + efficient inference" (next chapter) directly benefits embodiment: every drop in inference cost makes on-device / onboard real-time VLA + world model one notch more feasible.

Next chapter we pull the view to the "substrate": the attention mechanisms, efficient training / inference systems, on-device intelligence, and frontier RL that support all the multimodal models above — i.e. the optimization techniques that make these models "run, run fast, run on-device".
