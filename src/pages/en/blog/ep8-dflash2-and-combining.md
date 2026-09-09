---
title: 'DFlash 2 and the Combine Question: Block-Diffusion Upgrade, but Why Don''t Frameworks Just Stack DSpark Scheduling On Top?'
description: 'DFlash 2 patches block-internal coherence and tail decay with a path selector and local conv; it is orthogonal, not a replacement for DSpark. This post explains why vLLM/SGLang do not simply bolt DSpark-style confidence scheduling onto a DFlash 2 drafter, and what the actually-shippable combination looks like today.'
pubDate: 2026-09-09
series: Speculative Decoding Notes
lang: en
altLang: zh
altHref: /blog/ep8-dflash2-and-combining
layout: ../../../layouts/BlogPost.astro
---

> The one-line thread: **DFlash 2 changes *how the draft is produced* (post-hoc path selection + physical information flow); DSpark changes *how much to verify* (confidence scheduling). The drafter is a mutually-exclusive config knob in the frameworks; only the scheduler is orthogonally composable — and DSpark's scheduler depends on its own confidence-head signal, so reusing it on a DFlash drafter still needs a confidence proxy plus a benchmark. That is why it is not shipped as a named combo yet.**

This post follows ep3 (DFlash) / ep4 (DSpark) / ep5 (head-to-head) / ep7 (MTP & confidence heads). Quick DFlash 2 recap, then the meta-question.

## 1. What DFlash 2 actually changes (recap)

DFlash 1 had two known flaws: block-internal incoherence (per-position independent Top-1 → "我 我 想") and tail decay (block-internal attention down to 8% in shallow layers). DFlash 2 patches both:

| Patch | What it does | Cost |
|---|---|---|
| Path selector (2M params) | LM Head already emits the full vocab → keep **Top-16 candidates** for free; a light scorer computes "adjacency" in parallel; **greedy backtracking** picks one coherent path (lookup-only, 0.6% latency); rejection sampling preserved → provably lossless | +2M |
| Local conv (16.5M / +3%) | Insert **two-tap dynamic depthwise conv** (current + left token) around each Attn/FFN sublayer; forces left→right flow; block-internal attention 9.4%→**0.5%**, equivalent to deepening to 15 layers | +16.5M |

Measured (MindStudio, single H200, Qwen3.8-27B): acceptance length **5.46** on GSM8K vs 5.02 (built-in MTP head) and 4.36 (community DSpark drafter); 3.1–3.4× at concurrency 1, still >1.0× at concurrency 32, and ahead of MTP and the DSpark drafter at every concurrency and task.

## 2. The meta-question: why not just combine them?

The intuition is fair — "DFlash 2 drafter + DSpark-style confidence-scheduled verification, one owns the draft, the other owns the verify, different layers, why not stack?" Three layers of the answer:

### 2.1 The drafter is a mutually-exclusive config, not a stackable module

A single SD request runs **one** drafter. Frameworks pick it with one `method` field:

- SGLang: `--speculative-algorithm DFLASH` vs `DSPARK` (exclusive)
- vLLM: `speculative-config '{ "method": "dflash" }'` vs `"dspark"` (exclusive)

DFlash is **block diffusion** (one forward, whole block); DSpark is **semi-AR + Markov head** (block-parallel, intra-block serial). Different checkpoints and forward logic — you cannot be "block-diffusion" and "semi-AR" in the same forward. So "combining drafters" is physically impossible; you pick DFlash 2 *or* DSpark.

### 2.2 What is truly composable is the verify/scheduling policy — and it is already partly unified

DSpark's confidence scheduler (verify more when confident, less when unsure) lives in the **verify stage**, decoupled from the drafter. In principle it is **model-agnostic** and can sit on top of any drafter. In practice:

- vLLM has `#48692` adaptive speculative decoding in progress;
- SGLang `#30261` already merged full **confidence scheduling + ragged verify + CUDA graph**;
- aihot logs show adaptive token budgets already lifted DSpark TTFT by 55–65%.

So "DFlash 2 drafter + adaptive verify budget" is already the closest real shape — it just usually goes by "DFlash + adaptive K", not "DFlash × DSpark".

### 2.3 So why doesn't anyone bolt DSpark's scheduler onto DFlash directly? The confidence signal source

DSpark's scheduler consumes the **per-token confidence its own Markov / Confidence Head produced at draft time**. Swap in a DFlash 2 drafter and that signal source changes:

- DFlash 2's path selector emits a **coherence / accept score**, not a per-token accept probability;
- only the target model's verification yields the real per-token accept probability.

To reuse DSpark's scheduler verbatim on a DFlash drafter you first need a **confidence proxy** (mapping path-selector score / target accept probability into what the scheduler expects), then **publish a benchmark proving the combo wins**. That glue + eval is exactly what is "not yet packaged as a named combo" — not impossible, just missing a contributor who PRs it and benchmarks it.

### 2.4 Two more real constraints

- **Maturity**: as of now, DFlash 2 integration is not in stable releases — vLLM is an unmerged PR (`#52816`, branch install), SGLang needs source build. DSpark is merged but "actively expanding". Combining two not-yet-stable methods is lower priority.
- **High-concurrency backfire**: at concurrency 32 some DSpark tasks drop below 1.0× (slower than vanilla); DFlash 2 stays >1.0×. Adaptive scheduling that picks the wrong K under heavy load directly loses throughput. Frameworks default to the safest **fixed K** to avoid regressions; adaptive is opt-in.

## 3. Conclusion: what the "best combo" actually looks like in today's frameworks

| Combo | Feasible? | Status |
|---|---|---|
| DFlash 2 drafter + fixed-K verify | ✅ shipped | SGLang (source) / vLLM (PR #52816) |
| DSpark drafter + confidence scheduler | ✅ shipped | vLLM #46995 / SGLang #30261 |
| **DFlash 2 drafter + adaptive verify budget** | ✅ effectively feasible | vLLM #48692 / SGLang adaptive budget, just not called a "combo" |
| DFlash 2 drafter + DSpark scheduler (zero-change) | ❌ impossible | confidence signal source mismatch |
| DFlash 2 + DSpark dual drafters in parallel | ❌ impossible | drafters are mutually exclusive |

So your line "not a replacement, an overlay" is right — with one precise footnote: **the overlay is at the scheduler layer, not the drafter layer**; and moving DSpark's scheduler onto DFlash "as-is" is blocked by a missing confidence proxy plus a missing benchmark. For OpenInfer the pragmatic landing spot is the first row: DFlash 2 drafter + the frameworks' existing adaptive verify budget, letting "how much to verify" ride on the already-shipped confidence scheduler rather than rewiring DSpark's Markov head.

## 4. Thread back to earlier posts

- ep5 said the two are "orthogonal, composable" — this post pins "composable" to "scheduler layer", "exclusive" to "drafter layer".
- ep7 covered DSpark's Confidence Head and load-aware scheduling — this post notes that scheduler depends on that head's signal, hence not directly portable to a DFlash drafter.
- Possible next: a per-operator data-flow diagram for the path selector or the two-tap conv, or an ep9 on how vLLM #48692 / SGLang #30261 implement the adaptive budget.
