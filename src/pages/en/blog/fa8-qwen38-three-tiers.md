---
title: 'Frontier Architecture Decoding Notes (8): The Qwen3.8 Family in One Frame — the 2.4T Flagship, Flash the Serving Build, and Flash-Next the Architecture Preview'
description: 'The three Qwen3.8 product lines in one place: Flash-Next is the Qwen4 architecture preview (125B MoE plus a 51B N-gram table, about 6B active per token, 3x GDN alternating with 1x QSA, trained with Muon at roughly one ninth the cost of Qwen3.7-Plus); Flash is the serving-tuned build of the same architecture (weights closed, $0.16/$0.47 per million tokens); the 2.4T-A95B flagship is the current-generation line (512 experts in a 10+1 split, about 95B active per token, 1M context, $2/$6 with no long-prompt surcharge). Includes a full comparison table and a structure diagram.'
pubDate: 2026-09-09
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa8-qwen38-three-tiers
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

The Qwen3.8 family is not "three models of different sizes" — it is **three product lines doing three different jobs**:

| Version | What it is | One line |
|---------|------------|----------|
| **Flash-Next** | **Architecture preview** of the next-generation Qwen4 (open-sourced 2026-08-26) | Uses 6B activations to probe where the architecture goes next |
| **Flash** | **Serving-tuned build** of that same architecture (weights closed) | The production model behind the API pricing |
| **2.4T-A95B / Max** | **Current-generation flagship** (open weights released 8/12) | Qwen-Max-class capability open-weighted for the first time |

One line to remember: **Next tells you what the next generation looks like, Flash is what you can cheaply use today, and 2.4T is the strongest model you can also download yourself.**

## 1. The Three Lines in One Table

| Dimension | Flash-Next | Flash | 2.4T-A95B (Max) |
|-----------|------------|-------|------------------|
| **Positioning** | Architecture preview (Qwen4Exp) | Serving-tuned production build | Current-generation flagship |
| **Total params** | 125B MoE + 51B N-gram (about 180B stored) | Same architecture as Flash-Next | **2.4T** sparse MoE |
| **Active per token** | **about 6B (<5%)** | Same | **about 95B (about 4%)** |
| **Attention** | **QSA sparse attention** (alternating 3:1 with GDN) | Same | Gated attention |
| **Experts** | 512-expert MoE | Same | 512 experts (**10 routed + 1 shared**) |
| **Context** | Native 262,144, YaRN to 1M | Same | 1M |
| **Modalities** | Text / image / video | Same | Multimodal |
| **Weights** | ✅ Open (qwen-community-1.0, BF16 + FP8) | ❌ Closed | ✅ Open (2.4T-A95B, custom license) |
| **API pricing** | 1 CNY in / 3 CNY out per million tokens | **$0.16 / $0.47** per million tokens | **$2 / $6**, no long-prompt surcharge |
| **Independent eval** | Beats DeepSeek-V4-Flash on most benchmarks | — | Artificial Analysis intelligence index **58**, coding **71.8** |

> Training cost: Flash-Next came in at about **one ninth of Qwen3.7-Plus** — the most direct evidence that the small-activation-plus-new-architecture route pays off.

## 2. Seven Shared Components: Aligned at the Source Level

Put the three repositories side by side and **seven component families keep the same names and organization**: GDN, QSA, MoE, RMSNorm, Residual, KV Cache, MTP. The difference is never "present or absent" — it is "what parameters and what combination":

<div class="fig">
<svg viewBox="0 0 660 300" role="img" aria-label="Flash-Next versus the 2.4T flagship">
  <title>Figure 1: Flash-Next (Qwen4 preview) versus 2.4T-A95B (current-generation flagship)</title>
  <text x="20" y="26" font-size="13" fill="#6b7280">Flash-Next / Flash — Qwen4 preview architecture (about 6B active per token)</text>
  <rect x="20" y="40" width="300" height="34" rx="6" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="170" y="62" font-size="12" fill="#534AB7" text-anchor="middle">N-gram Embedding 51B (host memory, not HBM)</text>
  <rect x="20" y="81" width="300" height="34" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="170" y="103" font-size="12" fill="#185FA5" text-anchor="middle">GDN x3 per group (Gated DeltaNet linear attention)</text>
  <rect x="20" y="122" width="300" height="34" rx="6" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="170" y="144" font-size="12" fill="#534AB7" text-anchor="middle">QSA sparse attention x1 per group (3:1 with GDN)</text>
  <rect x="20" y="163" width="300" height="34" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="170" y="185" font-size="12" fill="#185FA5" text-anchor="middle">MoE 512 experts, about 6B active per token</text>
  <rect x="20" y="204" width="300" height="34" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="170" y="226" font-size="12" fill="#185FA5" text-anchor="middle">MTP multi-token prediction head</text>
  <text x="360" y="26" font-size="13" fill="#6b7280">2.4T-A95B / Max — current-generation flagship (about 95B active)</text>
  <rect x="360" y="52" width="280" height="44" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="500" y="72" font-size="12" fill="#185FA5" text-anchor="middle">GDN (Gated DeltaNet)</text>
  <text x="500" y="88" font-size="11" fill="#6b7280" text-anchor="middle">linear attention for local and global state</text>
  <rect x="360" y="108" width="280" height="44" rx="6" fill="#faeeda" stroke="#BA7517"/>
  <text x="500" y="128" font-size="12" fill="#854F0B" text-anchor="middle">Gated attention (current-generation mainline)</text>
  <text x="500" y="144" font-size="11" fill="#6b7280" text-anchor="middle">not QSA — still the proven route of this generation</text>
  <rect x="360" y="164" width="280" height="44" rx="6" fill="#faeeda" stroke="#BA7517"/>
  <text x="500" y="184" font-size="12" fill="#854F0B" text-anchor="middle">MoE 512 experts (10 routed + 1 shared)</text>
  <text x="500" y="200" font-size="11" fill="#6b7280" text-anchor="middle">about 95B active per token (about 4%)</text>
  <rect x="360" y="220" width="280" height="44" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="500" y="240" font-size="12" fill="#185FA5" text-anchor="middle">1M context, multimodal, flat $2/$6 pricing</text>
  <text x="500" y="256" font-size="11" fill="#6b7280" text-anchor="middle">no long-prompt surcharge</text>
  <text x="20" y="288" font-size="11" fill="#6b7280">Blue = shared across both generations | Purple = added or replaced in Flash-Next (Qwen4 preview) | Orange = current-generation flagship route</text>
</svg>
<figcaption>Figure 1: both generations share the GDN / MoE / MTP skeleton; Flash-Next swaps in QSA sparse attention plus a 51B host-resident N-gram table for 6B activations, while the flagship stays on gated attention with 95B activations.</figcaption>
</div>

**The real generational divide is attention**: the flagship (2.4T) uses the proven **gated attention** of the current generation, while Flash-Next switches to **QSA sparse attention**, alternating 3:1 with GDN. That is what "preview" means — **Qwen4 will most likely follow the QSA line**.

## 3. Flash-Next: Trading a 51B Host Table for 6B Activations

fa7 already covered its weight organization, so here is only **why the design holds up**:

- **125B main model plus a 51B N-gram Embedding**, roughly 180B parameters on disk (about 360 GB in BF16)
- The key trade: **the 51B N-gram table does not go into HBM — it lives in host memory**, trading PCIe bandwidth for HBM capacity
- Only about **6B activates per token** (roughly 95% sparsity), which is what buys the order-of-magnitude drop in inference cost
- Trained with the **Muon optimizer** at about **one ninth** the cost of Qwen3.7-Plus, yet stronger on coding and office tasks
- Native context of 262,144, extendable to 1M with YaRN

**The hidden precondition**: you must accept part of the model being resident in host memory. That is unfriendly to single-GPU setups and a good deal for host-memory-rich serving fleets — which is exactly why the API price lands at 1 CNY in / 3 CNY out per million tokens, about a third cheaper than DeepSeek-V4-Flash.

## 4. Flash: The Serving-Tuned Build

Flash is the easiest to misread — **it is not a smaller Flash-Next, it is the same architecture tuned for production**:

- Weights are **closed** (the single most important difference from Flash-Next)
- API pricing of **$0.16 per million input** and **$0.47 per million output** tokens
- It carries the production traffic of the QwenCloud API

**Why two names**: the preview (Next) exists to publish the architecture direction and gather community feedback; the serving build (Flash) exists to provide stable supply. Same architecture, but service level, quantization choices, and rollout cadence can be managed separately. This is also why a model with "Next" in its name **should not go straight to production** — the model card itself marks it as an architecture preview, and tooling support may lag.

## 5. The 2.4T-A95B: Qwen-Max-Class Capability, Open-Weighted for the First Time

The historical weight of this release exceeds its parameter count: **the Max tier has always been the closed, hosted, most capable option — this is the first time it ships as open weights**.

- **2.4T total parameters in a sparse MoE**, about **95B active per token** (about 4% — a sparsity ratio close to Flash-Next, but with over 15x the absolute size)
- Of the 512 experts, **10 are routed and 1 is shared** — different from the plain 512-expert structure of Flash-Next
- API side: **1M context**, multimodal, **flat $2/$6 with no long-prompt surcharge** (this pricing strategy matters a lot for long-context workloads)
- Open weights as **Qwen3.8-2.4T-A95B** (released 8/12) plus an **FP8 variant**, under a custom non-Apache license
- Independent evaluation: Artificial Analysis **intelligence index 58, coding index 71.8** — the strongest in the Qwen3.8 family

## 6. Inference Control: Two Underrated Parameters

Two API parameters introduced with Qwen3.8 matter more for engineering than any benchmark:

| Parameter | What it does | Engineering value |
|-----------|--------------|-------------------|
| **`reasoning_effort`** | Tune reasoning depth per request | Classification over short text gains nothing from deep thinking and pays for it in latency and cost; multi-step debugging does. **You no longer need two models for cheap work and hard work** |
| **`preserve_thinking`** | Retain thinking context across turns | In iterative sessions the model stops re-deriving the same conclusions every turn — fewer tokens, and no drift when the second derivation lands somewhere slightly different. Long agentic sessions get cheaper and more stable |

Both point at the same design goal: **optimize for tasks that take many steps, not for one impressive answer**.

## 7. Which One Should You Pick

| Your situation | Pick |
|-----------------|------|
| Self-hosting, want to try the next-gen architecture, can accept 360 GB of weights plus a host-memory N-gram table | **Flash-Next** (BF16 or FP8) |
| Lowest API cost for everyday coding and office tasks | **Flash** ($0.16/$0.47) |
| Current best capability plus 1M context and multimodal, self-hosted | **2.4T-A95B** (budget memory by the 95B activation, not the 2.4T total) |
| Limited local hardware, just want it to run | **27B** (dense, 55.6 GB in BF16, the most downloaded of the family) |

**One reminder**: do not size the 2.4T flagship by its total parameter count — budget KV Cache and activations by the 95B figure (see sys9 for the formula), and only the weights by the total.

## 8. Links to the Rest of the Series

- **fa6 (Gated DeltaNet)**: the linear attention layer shared by all three versions — the mechanics live there.
- **fa7 (dual checkpoints)**: the weight organization, config differences, and storage breakdown of Flash-Next versus 27B; not repeated here.
- **sys9 / sys10**: use those formulas and tools to size KV Cache and serving cost for any of the three.
- **ag1**: `preserve_thinking` is designed exactly for agentic multi-turn workloads — another instance of KV Cache turning from a memory problem into a scheduling problem.
