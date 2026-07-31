---
title: "DFlash Deep Dive: Block-Diffusion Drafting, KV Injection, and a Bubble-Free Pipeline"
description: A breakdown of DFlash's three mechanisms — block-diffusion drafting, target hidden-state conditioning (KV injection), and the Spec V2 overlap scheduler — plus its measured speedups on Qwen3.
pubDate: 2026-07-31
series: Speculative Decoding Notes
lang: en
altLang: zh
altHref: /blog/ep3-dflash
layout: ../../../layouts/BlogPost.astro
---

## 1. Background and origin

**DFlash** was systematically introduced by **Z Lab + SGLang + Modal** in a blog post on 2026-06-15, with the goal of pushing speculative decoding on the Qwen3 family up to **6×**. It breaks the "serial drafting" ceiling from Ep2 because it reworks both the drafting paradigm and the execution engine.

## 2. Core 1: Block Diffusion Drafter

Classic drafters are **autoregressive and serial** — they guess one token at a time. DFlash switches to a **diffusion** paradigm:

- A single forward pass proposes an entire **block** of masked future tokens in parallel (default block size = 16, `--speculative-dflash-block-size`);
- Intuitively, instead of "guessing the next token one by one", it "fills in a whole block of possibilities at once, then confirms them together";
- The cost is that diffusion denoising needs multiple iterations, but versus K serial autoregressive forward passes this is still substantially cheaper.

This single move sidesteps Ep2's bottlenecks (a) serial drafting and (c) in-block sequential dependency — drafting goes from "serial token chaining" to "parallel block emission".

## 3. Core 2: Target hidden-state conditioning + KV injection

Parallelism alone is not enough; the acceptance length α is what drives speedup. DFlash's key trick is to inject the **target model's intermediate hidden states** into the drafter's KV projections (across layers):

- The drafter no longer "guesses blind" — it is **conditioned on the target model's features**;
- KV injection aligns the drafter's proposals tightly with the target context → acceptance rate goes up significantly.

This is where DFlash's high α comes from, and a major reason it is ~2.5× faster than EAGLE-3.

## 4. Core 3: Spec V2 engine + overlap scheduler (execution layer)

Most SD work only changes the algorithm; DFlash also reworks the execution engine:

- On a standard pipeline, host-side cleanup / KV allocation and GPU compute leave **synchronization bubbles**;
- The **overlap scheduler** runs host tasks **concurrently** with GPU compute → an additional **+33%** throughput.

This is what separates DFlash from "pure-algorithm" approaches: **it changes both the drafting paradigm and the execution engine**.

## 5. Measured results

| Metric | Value |
|---|---|
| Qwen3-8B peak speedup | **6×** |
| vs EAGLE-3 | ~**2.5×** faster |
| On Blackwell | up to **15×** |
| overlap scheduler gain | **+33%** throughput |

## 6. Summary and next

DFlash's triple move — **block-diffusion parallel drafting + KV injection for higher acceptance + bubble-free pipeline** — lifts SD from 2–3× to 6×. Next we look at the alternative route **DSpark**: instead of diffusion, it uses semi-autoregressive drafting + a Markov head + a confidence scheduler to take a different path.

> This is Ep3 of the "Speculative Decoding Notes" series. Prior: [The Autoregressive Drafter's Ceiling](../ep2-arg-ceiling); head-to-head: [DFlash vs DSpark](../ep5-dflash-vs-dspark); next: [DSpark Deep Dive](../ep4-dspark).
