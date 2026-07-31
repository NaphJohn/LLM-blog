---
title: 'Multimodal Decoding Notes (1): Foundations & Alignment — from "Writing" to "Seeing & Acting"'
description: This series uses autoregressive generation as the single key that unlocks multimodality. The opener explains what a modality is, why different modalities must be squeezed into one shared representation space (the alignment problem), and how GPT-1's generative pre-training became the seed paradigm for the multimodal brain. A full roadmap closes the post.
pubDate: 2026-07-31
series: Multimodal Decoding Notes
lang: en
altLang: zh
altHref: /blog/mm1-foundations
layout: ../../../layouts/BlogPost.astro
---

## 0. Where This Series Sits

"Speculative Decoding Notes" covered **how to run LLMs faster**; this series covers **where a model's ability to perceive and act on the world comes from** — i.e. multimodality.

The thread running through the whole series: **autoregressive generation (autoregressive decoding) is the key that unifies every modality.** Text is a sequence of tokens; images can be discretized into token sequences (or generated continuously via diffusion); actions can be discretized into action tokens; even a "world model's" predictions can be viewed as a kind of generation. Once you see "generation = decoding", multimodality stops being a scattering of sub-fields and becomes one coherent evolution.

> Roadmap: **(1) Foundations & Alignment** → (2) VLM evolution → (3) Generative models (diffusion / GAN) → (4) VLA & world models → (5) Efficient systems & frontier inference.

## 1. What Is a "Modality", and Where's the Hard Part

- **Modality**: a natural form of information — text, image, audio, video, action (robot joint angles / end-effector pose), even touch, IMU signals.
- **The core difficulty = alignment**: raw data of different modalities have wildly different distributions (text = discrete symbols, image = continuous pixel tensors, action = low-frequency continuous vectors). To let one model "understand" all of them, they must be **projected into one computable representation space**, where "text ↔ image ↔ action" correspondences can be learned.
- **Two levels of alignment**:
  - *Representation alignment*: encode different modalities into nearby vectors (e.g. CLIP pulls "cat image" and "cat" close);
  - *Instruction alignment*: make the model reason / generate across modalities given a natural-language instruction (e.g. "move the cup left" → a sequence of actions).

## 2. Generative Pre-training: the "Seed Paradigm" of the Multimodal Brain

The origin is **GPT-1 (Radford et al., 2018)** — the first proof that "generative pre-training + downstream fine-tuning" can dominate NLU, writing the entire genome later inherited by ChatGPT.

- **Core formula**: causal language modeling on BookCorpus `L₁ = Σ log P(uᵢ | u<i; Θ)`, then transfer `Θ`, add a linear head `W_y`, fine-tune, with `λ·L₁` to prevent forgetting.
- **Architecture**: Decoder-only Transformer, 12 layers, d=768, h=12, ctx=512, 117M params (small, but the roadmap is complete).
- **Why it is the multimodal bedrock**: GPT established a highly scalable paradigm — **first "learn to generate" on massive data, then replace / extend the generation space to new modalities.** Later multimodal work merely pushes this "generation space" from text tokens outward to image tokens, then action tokens.

> In one line: **multimodality is not built from scratch; it is GPT's "generative pre-training + fine-tuning" paradigm extrapolated along the output space all the way to vision and action.**

## 3. The Main Thread of Multimodal Evolution (Preview)

```
Generative pre-training (can "write" language)
  └─ Multimodal instruction fine-tuning (can "see + speak")   → Series (2) VLM
       └─ Vision-Language-Action VLA (can "act")               → Series (4) VLA + world model
            └─ VLA + world model (can "think then act")         → Series (4)
A separate generative branch: diffusion / GAN (can "draw")      → Series (3)
The efficiency floor that supports all of it: attention / KV Cache / spec decoding / on-device → Series (5)
```

Investment angle: an embodied "brain" did not appear from nowhere — it reuses the pre-training + fine-tuning paradigm already validated by language / multimodal LLMs; the only difference is the **output space extends from "text tokens" to "action tokens"**. This also explains why "LLM cost reduction" directly benefits embodiment — every drop in inference cost makes on-device / onboard real-time VLA one notch more feasible.

## 4. Practical Takeaways

- When evaluating a multimodal / embodied company, first check **whether it reuses a mature LLM / VLM pre-training paradigm** (rather than reinventing one) — the first signal of engineering maturity.
- Alignment quality (not parameter count) often sets the usability floor; data (especially "image-text" and "image-text-action" pairs) is the biggest bottleneck, often 1~2 orders of magnitude short.
- Next chapter we get concrete: how ViT, CLIP, and LLaVA turn "see + speak" into a reproducible standard paradigm.
