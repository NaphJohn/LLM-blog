---
title: "DSpark Deep Dive: Semi-Autoregressive Drafting with a Confidence Scheduler"
description: A breakdown of DSpark (DeepSeek + PKU, open-source) — its semi-autoregressive drafter, Markov-head ordering, and confidence scheduler, plus where it lands in vLLM / SGLang.
pubDate: 2026-07-31
series: Speculative Decoding Notes
lang: en
altLang: zh
altHref: /blog/ep4-dspark
layout: ../../../layouts/BlogPost.astro
---

## 1. Background and origin

**DSpark** was open-sourced by **DeepSeek + Peking University** on 2026-06-27 (MIT license, repo `deepseek-ai/DeepSpec`). It takes a **different route** from DFlash: instead of diffusion denoising, it uses **semi-autoregressive drafting + a Markov head** for ordering and a **confidence scheduler** to dynamically control the verification length.

## 2. Core 1: Semi-Autoregressive Drafter + Markov Head

- **Semi-autoregressive**: emits tokens in **blocks** — parallel across blocks, ordered within a block;
- **Markov head**: injects token-to-token sequential dependency **within** a block, keeping the in-block generation order consistent (no scrambling).

This contrasts with DFlash's diffusion paradigm: DSpark sits between "pure autoregressive" and "pure parallel" — more parallel than autoregressive, more controllable than pure diffusion.

## 3. Core 2: Confidence Scheduler

This is DSpark's cleverest move:

- Instead of a fixed block size, it **dynamically decides how many tokens to verify this round based on draft confidence**;
- **High confidence** → verify more (swap several tokens at once); **low confidence** → verify fewer (avoid the waste of feeding a whole block that gets largely rejected).

Versus DFlash's fixed block size = 16, DSpark's verification length adapts to "should we trust the draft right now", pushing closer to α's theoretical ceiling.

## 4. Core 3: Parallel target verification, lossless

Like DFlash, DSpark is **lossless speculative decoding**: the draft block goes to the target model for one parallel forward pass, and rejection sampling guarantees the output distribution is exactly the target model's.

## 5. Results and ecosystem

| Metric | Value / Status |
|---|---|
| V4 speedup | **57–85%** (per DeepSeek / PKU release) |
| Throughput | **+400%** |
| vLLM landing | PR **#46995** in progress (reuses sparse-MLA non-causal indexing, `DSparkSpeculator` extends `DFlash`, Triton non-causal SWA kernel) |
| Multi-backend | Supports SGLang / OpenInfer simultaneously |

## 6. Relationship to DFlash (preview)

DSpark changes "draft ordering + verification-length scheduling"; DFlash changes "draft parallelism paradigm + execution engine" — they are **on different layers and orthogonal, hence composable**. That is exactly why OpenInfer can run **both** paths on Qwen3-4B. Full comparison: [DFlash vs DSpark](../ep5-dflash-vs-dspark).

> This is Ep4 of the "Speculative Decoding Notes" series. Prior: [DFlash Deep Dive](../ep3-dflash); head-to-head: [DFlash vs DSpark](../ep5-dflash-vs-dspark).
