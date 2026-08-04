---
title: 'Multimodal Decoding Notes (2): VLM Evolution — ViT → CLIP → LLaVA'
description: Following the "can see and speak" thread, we dissect the standard Vision-Language Model (VLM) paradigm. ViT turns an image into a token sequence; CLIP uses contrastive learning to squeeze image and text into one shared space; LLaVA uses GPT-4 to synthesize data plus a frozen vision encoder + projection bridge + LLM fine-tuning, settling the whole VLM methodology.
pubDate: 2026-07-31
series: Multimodal Decoding Notes
lang: en
altLang: zh
altHref: /blog/mm2-vlm-evolution
layout: ../../../layouts/BlogPost.astro
---

## 1. Why VLM Is the Pivot Chapter

Part (1) said: the essence of multimodality is **alignment**. The text side already has mature language models; what about the image side? The evolution of VLM is exactly the step-by-step process of making "image" digestible for a language model.

## 2. ViT: Slicing an Image into a Token Sequence (Dosovitskiy et al., 2020)

- **Core insight**: a Transformer does not care about input modality. Slice an image into 16×16 patches, treat each patch as a "visual token", and feed it into a Transformer like text tokens.
- **Recipe**: `image → patch embedding → +position → Transformer encoder → global representation`.
- **Significance**: for the first time image and text are **architecturally isomorphic** (both token sequences), paving the way for unified representation. Virtually every multimodal vision encoder today builds on ViT (or a variant).

## 3. CLIP: Squeezing Image & Text into One Space via Contrastive Learning (Radford et al., 2021)

- **Core mechanism**: contrastive learning on 400M image-text pairs — pull matching pairs close, push mismatched pairs apart. The training objective is not a classification head but a "image-text match" similarity function.
- **Key output**: a **shared embedding space** where "cat image" and the text "a cat" land in nearby vectors. This space enables zero-shot: use class names as prompts and compute image-text similarity to classify.
- **Significance**: CLIP is the first industrial-grade "alignment" solution and became the vision backbone for LLaVA / DALL·E / every image-text model.

## 4. LLaVA: the Standard VLM Paradigm Is Set (NeurIPS 2023 Oral)

LLaVA proved one thing: **you do not need to train a multimodal LLM from scratch — freeze the vision encoder + a light projection + fine-tune the LLM is enough.**

- **Data magic**: use GPT-4 to "imagine" **158K multimodal instruction examples** from COCO captions (dialogue / description / reasoning, zero human labels) — cracking the deadlock of scarce multimodal instruction data.
- **Minimal architecture**: `frozen CLIP ViT + trainable linear projection W + Vicuna LLM`.
- **Two-stage training**:
  1. *Feature alignment*: use only image-text pairs, train `W` to align ViT output to LLM word-embedding space;
  2. *Instruction fine-tuning*: attach the 158K instructions, unfreeze the LLM and train together.
- **Results**: GPT-4-as-Judge relative score 85.1%, ScienceQA 92.53% SOTA.

<div class="fig">
  <img src="/llava_arch.svg" alt="LLaVA architecture: input image encoded by frozen CLIP ViT-L/14, bridged via a trainable projection to Vicuna's text-embedding space, concatenated with text tokens and autoregressively decoded by the LLM" />
  <p class="fig-cap">Fig: The LLaVA standard paradigm — frozen vision encoder + light projection bridge + LLM fine-tuning (redrawn)</p>
</div>

> This paradigm (frozen vision encoder + projection bridge + LLM fine-tuning) spawned LLaVA-1.5 / NeXT, CogVLM, InternVL, Qwen-VL — almost every "chat about an image" model today descends from this skeleton.

## 5. What Determines a VLM's Quality

- **Alignment-data quality**: LLaVA's 158K shows data construction (not model structure) is often the ceiling. Diversity and reasoning-chain depth directly decide "understand → explain" ability.
- **Projection design**: linear projection is lightest, but MLP / attention-style projections are more stable on complex tasks.
- **Vision-encoder resolution**: higher resolution (more patches) is crucial for OCR / fine-grained understanding; Qwen-VL / InternVL keep investing here.
- **Coupling depth with the LLM**: from "train projection only" → "joint fine-tuning" → "native multimodal pre-training (GPT-4V / Gemini)" — capability ceiling rises, and so does cost.

## 6. Bridge to the Next Chapter

VLM solved "see + speak". But many real tasks demand "act" — turning visual understanding into **actions**. Next we return to the other generative branch: how images / video are generated (diffusion and GAN); then we connect all this to robot actions, entering VLA and world models.

> Practical tip: when looking at a multimodal model, ask three questions first — "whose vision encoder (CLIP/ViT family?), how was the alignment data made, how is the projection connected?" — these three quickly reveal engineering maturity, matching the evaluation framework from Part (1).
