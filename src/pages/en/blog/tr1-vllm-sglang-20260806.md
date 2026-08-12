---
title: vLLM & SGLang Community Tracker · 2026-08-06 — Both Frameworks Land Ling-3.0-flash Same Day; Step Stalls
description: LLM Infra Daily #6. Window 2026-08-05→08-06 — vLLM 52 commits / SGLang 61 commits; same day, same model (Ling-3.0-flash), both frameworks, both with speculative decoding; StepFun shows zero pushes for a third straight day and is reported to have split its strategy.
pubDate: 2026-08-12
series: Community Tracker Notes
lang: en
altLang: zh
altHref: /blog/tr1-vllm-sglang-20260806
layout: ../../../layouts/BlogPost.astro
---

## ★ Most Worth Watching Today

**On the same day, both frameworks landed the domestic model Ling-3.0-flash; meanwhile StepFun (Step) shows zero pushes for a third straight day and is reported to have split its internal strategy into two lines.**

- **vLLM** merged **#51045**: Bailing V3 MLA/KDA hybrid MoE + MTP drafter + ling3 parser, 15 files +2070 lines.
- **SGLang** merged **#33556**: Ling-3.0-flash cookbook, with GB300 measured GSM8K 96.66% / 96.97%.
- Same day, same model, both frameworks, both with speculative decoding — domestic model enablement has moved from "can it run" to "how fast can it run".

The contrast: **StepFun-ai org's main repos are still frozen at 06-01 / 04-03, and its official vllm fork stopped at 05-28**; while on 08-05 media reported Step internally established two independent strategy lines — "large model" and "agent terminal" — with overseas commercialization led by API (especially voice models).

**Conclusion**: Step's stagnation in the open-source inference ecosystem is not an ad-hoc scheduling issue; it aligns with where the org is moving resources. Treat Step as an observation benchmark for domestic inference enablement, but **down-weight the expectation**.

## 1. Baseline (no new tag from any of the three)

| Object | Latest stable | Released | This window |
|--------|--------------|----------|-------------|
| vLLM | v0.26.0 | 2026-07-27 | main: 52 commits |
| SGLang | v0.5.16 | 2026-07-25 | main: 61 commits (sgl-kernel → 0.4.6) |
| StepFun | Step-3.7-Flash / 3.5-Flash | push frozen 06-01 / 04-03 | zero GitHub progress; strategy news only |

> Note: sgl-kernel bumped to 0.4.6 via #33678 — a sub-package version; the SGLang main tag is unchanged.

## 2. The "Same-Frame" Signal of Domestic Model Enablement

Ling-3.0-flash entering both vLLM and SGLang trunks the same day, both explicitly with **speculative decoding (MTP drafter / cookbook throughput)**, tells us:

1. "Day-one dual-framework" support for new domestic models is now normal; the ecosystem moved from "asking frameworks to adopt" to "frameworks competing to adopt".
2. Speculative decoding (see this blog's "Speculative Decoding Notes" ep1–ep6) is now a **standard launch feature**, not optional.
3. GB300 measured GSM8K 96%+ confirms domestic models "can run" on inference cards; competition shifts to "run cheaper, run faster".

## 3. How to Read Step's "Absence"

Three days of zero commits + main repos frozen for months + strategy-split news all point the same way:

- Step's open-source inference investment has **yielded, temporarily**, to a two-front war ("large model vs agent terminal") and overseas API/voice commercialization.
- For engineering decisions like "should we prioritize Step in vLLM/SGLang", down-weight for now; if support is mandatory, prefer its official fork over waiting for mainline merge.

## 4. Positioning of This "Tracker" Series

This series (tr) turns daily vLLM / SGLang upstream commits, model cookbooks, and domestic-model enablement progress into readable notes with a "same-day same-frame / who is falling behind" comparative lens — filling the timeliness gap left by the principle-heavy "vLLM & SGLang Framework Notes (fw)".

- For the principle主线: back to **fw1–fw9** (why an inference engine → vLLM/SGLang internals → release waves → Wan2.2 dual-expert MoE).
- For operator detail: back to **op1 (MLA operator)**.
- For model-side comparison: back to **fa1–fa4 (Kimi K3 / MiniMax M3 / DeepSeek V4)**.

## 5. Today's Watchlist (to verify)

- [ ] Does vLLM's MTP drafter for Ling-3.0-flash land in the 0.26.x stable line, or main only.
- [ ] Does SGLang #33556's GB300 result set a new throughput baseline (vs vLLM on the same model).
- [ ] After Step's strategy split, any new open-source move (e.g. an agent-terminal inference framework).

> Sources: GitHub vLLM / SGLang commit streams (2026-08-05 01:00Z → 08-06 01:00Z); Sina/NetEase/Toutiao 08-05 strategy-line reporting.
