---
title: "Parameter-Efficient Fine-Tuning & On-Device Intelligence: LoRA / QLoRA / On-Device Deep Research / EcoSpec"
description: Three routes to "small resources, big models"—LoRA/QLoRA freeze the backbone and train only low-rank deltas; On-Device Deep Research shows a 4B model can do retrieval-augmented deep research; EcoSpec folds MoE's hidden activation cost into speculative decoding. Plus the paradigm significance of "on-device inference + test-time compute".
pubDate: 2026-07-31
series: AI Knowledge Base
lang: en
altLang: zh
altHref: /blog/kb-peft-ondevice
layout: ../../../layouts/BlogPost.astro
---

## 1. LoRA / QLoRA (Parameter-Efficient Fine-Tuning, PEFT)

- **Mechanism**: freeze pretrained `W₀`, train only the low-rank increment `ΔW = B × A` (r ≪ d), cutting finetune cost 50~100×.
- **QLoRA**: adds 4-bit NF4 + double quantization, letting a **65B model finetune on a single card**; community QLoRA finetune of MiniMax-8B costs < $20.
- **In use**: LLaVA itself uses LoRA (only 0.5% params); Qwen3.5 recommends `r=64 + DoRA`.
- **Significance**: turns "model customization" from a rich-only game into a平民 skill, the engineering base for today's vertical/industry model boom.

## 2. On-Device Deep Research at 4B (arXiv:2607.12257)

- **Core claim**: a **4B model can do "deep research"**—using "exposure boundary" (faithfulness) to curb hallucination and "retrieval coverage boundary" to curb omission.
- **Significance**: future **on-device / local "retrieval-augmented research assistant"** without cloud LLMs.
- **Investment mapping**: directly bullish for the on-device chip / edge-compute narrative; also matches your "localized, low-cost" approach to gathering industry intelligence.

## 3. EcoSpec: cost-aware speculative decoding for MoE (arXiv:2607.12696)

- **Problem**: speculative decoding usually assumes "the draft model is fast enough", but in MoE, **expert scattering** adds hidden activation cost (moving expert weights).
- **Approach**: factor activation cost into the drafting stage, reducing expert movement → faster decoding without quality loss.
- **Benefits**: directly helps inference of sparse MoE like **DeepSeek-V3/V4** in cost-sensitive scenarios.

## 4. On-device inference & test-time compute

- **Role**: run "good-enough models" locally / on-device (4B research, on-device VLA), and use **test-time compute** (multi-step verification, self-evolving verifiers) to compensate for insufficient parameters.
- **Gains**: saves cloud cost, preserves privacy, lowers latency—**the route on-device real-time robotic inference depends on**.
- **Representative**: On-Device Deep Research, various on-device VLAs.

## 5. Practical takeaways

- Cost reduction has two complementary paths: ① training-side PEFT (LoRA/QLoRA) to lower customization cost; ② inference-side "small model + test-time compute + retrieval" replacing blind scaling-up.
- For embodied / on-device investing: watch who makes "4B-class model + retrieval/verification" work—this is the cost inflection point for on-device real-time intelligence.
