---
title: "The Autoregressive Drafter's Ceiling: Why It Stalls at 2–3×"
description: Starting from the acceptance length α, this post breaks down why autoregressive drafters cap out at 2–3× speedup, and where DFlash / DSpark find their opening.
pubDate: 2026-07-31
series: Speculative Decoding Notes
lang: en
altLang: zh
altHref: /blog/ep2-arg-ceiling
layout: ../../../layouts/BlogPost.astro
---

## 1. Recap: where does the speedup actually come from?

In the previous post we described a speculative-decoding (SD) step as two phases:

1. **Draft**: a small model proposes K candidate tokens in parallel;
2. **Verify**: the large target model runs **one parallel** forward pass and independently accepts / rejects each candidate (rejection sampling).

Let the average number of accepted candidates be **α**. Then each "one large-model forward" yields roughly **α + 1** tokens (α accepted + 1 the target model writes itself). The ideal speedup is approximately:

```
speedup ≈ (α + 1) / (1 + cost_draft / cost_target)
```

When the drafter is cheap enough (cost_draft ≪ cost_target), the denominator approaches 1, so speedup ≈ **α + 1**.

**So the SD ceiling is α — the acceptance length.** To go faster, ask: how large can α be?

## 2. The physical upper bound on α

Rejection sampling guarantees a key property: **the distribution of accepted tokens is exactly the target model's distribution** (this is what makes SD "lossless"). The probability a single candidate is accepted is

```
P(accept) = min(1, p_target(token) / p_draft(token))
```

Hence α depends on the **distance (KL divergence) between the draft and target distributions**. The smaller and cheaper the drafter, the larger its deviation from the target → lower acceptance → smaller α.

Empirically, when using a "same-architecture smaller model" as the drafter, α typically lands in the **2–4** range. That is the root cause of why classic SD stalls at **2–3×**: it is not an engineering shortfall, it is a statistical lock imposed by distributional mismatch.

## 3. The hidden cost of autoregressive drafting

The formula above silently assumes one thing: **drafting K tokens costs ≈ 0**. But for an **autoregressive drafter** this is false —

An autoregressive drafter must also run its own forward pass per token, and it must do so **serially** (candidate i can only be drafted after candidate i−1). So:

- Drafting K candidates = **K serial forward passes**;
- As K grows, drafting latency grows **linearly**, eating the savings from parallel verification;
- Meanwhile α does not grow linearly with K — the "tail" candidates are mostly rejected (acceptance has a statistical ceiling).

Result: blindly increasing K does not speed things up linearly, and can even slow down. **There is an optimal K, and at that optimum the overall speedup sits right around 2–3×.**

## 4. Three structural bottlenecks (summary)

| Bottleneck | Explanation |
|---|---|
| (a) Serial drafting | The autoregressive drafter is itself serial, canceling the gain from "parallel target verification" |
| (b) α has a statistical ceiling | Set by draft/target distribution mismatch; can't be beaten by cranking K |
| (c) In-block sequential dependency | Token order within a block must be modeled serially; no natural parallelism |

Classic approaches (EAGLE family, etc.) have squeezed (b) to its limit but remain dragged down by (a) and (c) — that is the real story behind "stuck at 2–3×".

## 5. The opening (preview)

Since the bottleneck is "serial drafting" and "in-block ordering", the path forward is clear:

- **DFlash**: replace "serial autoregressive drafting" with **block-diffusion one-shot parallel drafting**, use KV injection to lift acceptance, and an overlap scheduler to kill execution bubbles (evades a, c);
- **DSpark**: use **semi-autoregressive + Markov head** to build in-block ordering, then a **confidence scheduler** to dynamically set the verification length (eases a, b).

Next, we dive into **DFlash** internals.

> This is Ep2 of the "Speculative Decoding Notes" series. Next: [DFlash Deep Dive](../ep3-dflash). For the head-to-head, see [DFlash vs DSpark](../ep5-dflash-vs-dspark).
