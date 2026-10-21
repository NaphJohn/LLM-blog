---
title: "Frontier Architecture Decoding Notes (9): A Decade of Residual Connections — ResNet → Hyper-Connections → mHC → Attention Residuals"
description: "Residual connections went unchanged for a decade after 2015, but LLMs turned them into a bottleneck: equal-weight addition dilutes shallow features and blows up deep hidden states. Of the three improvement routes, Hyper-Connections widens the stream but trains unstably, mHC restores stability with a doubly stochastic constraint and is already in DeepSeek-V4, and Attention Residuals treats depth as sequence with the most upside and the highest engineering cost. Includes a decade roadmap, a Post/Pre-norm comparison, three-route structure diagrams, an mHC doubly stochastic matrix diagram, and the Kimi Linear 48B-A3B experiment table."
pubDate: 2026-09-20
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa9-residual-connections
layout: ../../../layouts/BlogPost.astro
---

> Organized from Qiaonan "A Survey of LLM Structures: Three Improvements to the Residual Connection", 2026-09-20, with diagrams.

## 0. The One-Line Thread

Residual connections were fixed in shape in 2015 and stayed unchanged for a decade — but LLMs turned them into a bottleneck: **equal-weight addition dilutes shallow features and blows up deep hidden states**. Three improvement routes:

| Route | Approach | Stability | Adoption |
|-------|----------|-----------|----------|
| **Hyper-Connections** (ByteDance, 2024) | Residual stream C → nC, learnable depth + width connections | Unstable training (unconstrained mixing matrices explode when chained) | ICLR 2025 Borderline Accept |
| **mHC** (DeepSeek, 2026) | Same widening, but the mixing matrix is constrained to be **doubly stochastic** | Stable: mean preserved, spectral norm ≤ 1, products do not blow up | **Already in DeepSeek-V4** |
| **Attention Residuals** (Kimi, 2026) | Treat depth as sequence; each layer attends to all previous layers | Best experimental gain (Full version) | Block version works at N = 8 |

One line to remember: **HC and mHC are the same route — widening the residual stream — and the doubly stochastic constraint is the step that makes it actually usable; AttnRes treats the depth dimension as a sequence dimension, a bolder leap whose engineering adaptation period will probably be as long as the Transformer's was.**

## 1. Why the Residual Connection Became the Bottleneck

The residual connection aggregates information crudely — **it sums the outputs of all previous layers with equal weight** — which creates three problems:

- **Information dilution**: the relative contribution of shallow features decays linearly with depth;
- **Hidden-state blowup**: deep hidden states aggregate more and more information, and a single C-dimensional residual stream can no longer hold it;
- **Representation collapse**: as layers stack, representations of adjacent layers grow more similar and each new layer adds less, so the model's **effective depth is smaller than its actual depth**. Note this is not training failure — the model can be stacked deep, it just does not gain proportionally from going deeper.

<div class="fig">
<svg viewBox="0 0 660 210" role="img" aria-label="A decade of residual-connection evolution">
  <title>Figure 1: A decade of residual-connection evolution (2015 → 2026)</title>
  <text x="20" y="26" font-size="13" fill="#6b7280">A decade of residual-connection evolution (2015 → 2026): one bottleneck, three routes</text>
  <rect x="20" y="48" width="140" height="34" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="90" y="69" font-size="12" fill="#185FA5" text-anchor="middle">2015 · ResNet</text>
  <rect x="20" y="82" width="140" height="96" rx="6" fill="#f9fafb" stroke="#d1d5db"/>
  <text x="90" y="102" font-size="11" fill="#374151" text-anchor="middle">Identity maps beat</text>
  <text x="90" y="120" font-size="11" fill="#374151" text-anchor="middle">vanishing gradients; 152</text>
  <text x="90" y="138" font-size="11" fill="#374151" text-anchor="middle">layers became possible —</text>
  <text x="90" y="156" font-size="11" fill="#374151" text-anchor="middle">but every front layer is</text>
  <text x="90" y="174" font-size="11" fill="#374151" text-anchor="middle">summed with equal weight</text>
  <rect x="180" y="48" width="140" height="34" rx="6" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="250" y="69" font-size="12" fill="#534AB7" text-anchor="middle">2024 · Hyper-Conns</text>
  <rect x="180" y="82" width="140" height="96" rx="6" fill="#f9fafb" stroke="#d1d5db"/>
  <text x="250" y="102" font-size="11" fill="#374151" text-anchor="middle">Residual stream C → nC,</text>
  <text x="250" y="120" font-size="11" fill="#374151" text-anchor="middle">split into learnable depth</text>
  <text x="250" y="138" font-size="11" fill="#374151" text-anchor="middle">connections + width</text>
  <text x="250" y="156" font-size="11" fill="#374151" text-anchor="middle">connections (cross-stream</text>
  <text x="250" y="174" font-size="11" fill="#374151" text-anchor="middle">information exchange)</text>
  <rect x="340" y="48" width="140" height="34" rx="6" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="410" y="69" font-size="12" fill="#534AB7" text-anchor="middle">2026 · mHC</text>
  <rect x="340" y="82" width="140" height="96" rx="6" fill="#f9fafb" stroke="#d1d5db"/>
  <text x="410" y="102" font-size="11" fill="#374151" text-anchor="middle">Doubly stochastic constraint</text>
  <text x="410" y="120" font-size="11" fill="#374151" text-anchor="middle">on the mixing matrix makes</text>
  <text x="410" y="138" font-size="11" fill="#374151" text-anchor="middle">multi-stream residuals</text>
  <text x="410" y="156" font-size="11" fill="#374151" text-anchor="middle">stable; already used in</text>
  <text x="410" y="174" font-size="11" fill="#374151" text-anchor="middle">DeepSeek-V4</text>
  <rect x="500" y="48" width="140" height="34" rx="6" fill="#faeeda" stroke="#BA7517"/>
  <text x="570" y="69" font-size="12" fill="#854F0B" text-anchor="middle">2026 · AttnRes</text>
  <rect x="500" y="82" width="140" height="96" rx="6" fill="#f9fafb" stroke="#d1d5db"/>
  <text x="570" y="102" font-size="11" fill="#374151" text-anchor="middle">Depth treated as sequence:</text>
  <text x="570" y="120" font-size="11" fill="#374151" text-anchor="middle">each layer attends to all</text>
  <text x="570" y="138" font-size="11" fill="#374151" text-anchor="middle">previous layers; Block</text>
  <text x="570" y="156" font-size="11" fill="#374151" text-anchor="middle">version needs only N=8</text>
  <text x="570" y="174" font-size="11" fill="#374151" text-anchor="middle">for most of the gain</text>
  <g stroke="#9ca3af" stroke-width="1.5" fill="none">
    <path d="M163 65 L173 65"/><path d="M168 61 L174 65 L168 69"/>
    <path d="M323 65 L333 65"/><path d="M328 61 L334 65 L328 69"/>
    <path d="M483 65 L493 65"/><path d="M488 61 L494 65 L488 69"/>
  </g>
  <text x="20" y="196" font-size="11" fill="#6b7280">Blue = identity residual ｜ Purple = widened stream (HC / mHC) ｜ Orange = cross-layer attention</text>
</svg>
<figcaption>Figure 1: A decade of residual-connection evolution. ResNet fixed the aggregation scheme in 2015; ByteDance Hyper-Connections first widened the residual stream in 2024; in 2026 DeepSeek mHC restored stability with doubly stochastic matrices and entered a real model, while Kimi Attention Residuals treats the depth dimension as a sequence dimension.</figcaption>
</div>

## 2. The Post-norm vs Pre-norm Trade-off

To understand the three routes, first look at how the modern LLM default came to be:

- **Post-norm (original Transformer)**: `x_{l+1} = LayerNorm(x_l + F(x_l))`. Upside: every layer forcibly rewrites the representation, so collapse is rare. Downside: gradients must chain through the **Jacobian product** of every LayerNorm, so early training needs carefully designed warmup.
- **Pre-norm (modern LLM default)**: `x_{l+1} = x_l + F(LayerNorm(x_l))`. Upside: the identity path is direct, so gradients have a floor. Downside: **a big trunk plus a tiny delta** — the trunk grows layer by layer while each correction term stays constant, so going deeper looks more and more like fine-tuning on a huge trunk, and deep layers tend to collapse.

<div class="fig">
<svg viewBox="0 0 660 290" role="img" aria-label="Post-norm and Pre-norm signal and gradient paths">
  <title>Figure 2: Post-norm and Pre-norm signal and gradient paths</title>
  <text x="175" y="26" font-size="13" fill="#6b7280" text-anchor="middle">Post-norm · original Transformer</text>
  <text x="475" y="26" font-size="13" fill="#6b7280" text-anchor="middle">Pre-norm · modern LLM default</text>
  <rect x="90" y="46" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="175" y="65" font-size="12" fill="#185FA5" text-anchor="middle">x_l</text>
  <rect x="90" y="98" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="175" y="117" font-size="12" fill="#185FA5" text-anchor="middle">F(x_l)</text>
  <rect x="90" y="150" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="175" y="169" font-size="12" fill="#185FA5" text-anchor="middle">x_l + F(x_l)</text>
  <rect x="90" y="202" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="175" y="221" font-size="12" fill="#185FA5" text-anchor="middle">LayerNorm(·) = x_l+1</text>
  <rect x="390" y="46" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="475" y="65" font-size="12" fill="#185FA5" text-anchor="middle">x_l</text>
  <rect x="390" y="98" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="475" y="117" font-size="12" fill="#185FA5" text-anchor="middle">LayerNorm(x_l)</text>
  <rect x="390" y="150" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="475" y="169" font-size="12" fill="#185FA5" text-anchor="middle">F(LayerNorm(x_l))</text>
  <rect x="390" y="202" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="475" y="221" font-size="12" fill="#185FA5" text-anchor="middle">x_l + F(·) = x_l+1</text>
  <g stroke="#378ADD" stroke-width="1.5" fill="none">
    <path d="M175 76 L175 94"/><path d="M171 90 L175 97 L179 90"/>
    <path d="M175 128 L175 146"/><path d="M171 142 L175 149 L179 142"/>
    <path d="M175 180 L175 198"/><path d="M171 194 L175 201 L179 194"/>
    <path d="M475 76 L475 94"/><path d="M471 90 L475 97 L479 90"/>
    <path d="M475 128 L475 146"/><path d="M471 142 L475 149 L479 142"/>
    <path d="M475 180 L475 198"/><path d="M471 194 L475 201 L479 194"/>
  </g>
  <g stroke="#7f77dd" stroke-width="1.5" fill="none" stroke-dasharray="4 3">
    <path d="M90 61 C 56 61, 56 165, 84 165"/><path d="M80 161 L87 165 L80 169"/>
    <path d="M560 61 C 594 61, 594 217, 566 217"/><path d="M572 213 L565 217 L572 221"/>
  </g>
  <text x="84" y="118" font-size="10" fill="#6b7280" text-anchor="end">identity path</text>
  <text x="566" y="140" font-size="10" fill="#6b7280" text-anchor="start">identity path</text>
  <text x="175" y="252" font-size="11" fill="#6b7280" text-anchor="middle">Pro: every layer rewrites the representation</text>
  <text x="175" y="270" font-size="11" fill="#6b7280" text-anchor="middle">Con: gradients chain through LN Jacobians</text>
  <text x="475" y="252" font-size="11" fill="#6b7280" text-anchor="middle">Pro: identity path is direct; gradient floor</text>
  <text x="475" y="270" font-size="11" fill="#6b7280" text-anchor="middle">Con: big trunk + tiny delta; deep collapse</text>
</svg>
<figcaption>Figure 2: Post-norm puts LayerNorm after the residual addition — every layer forcibly rewrites the representation, but gradients chain through LN Jacobians. Pre-norm wires the identity path straight to the next layer, giving gradients a floor at the cost of an ever-growing trunk and ever-smaller per-layer corrections.</figcaption>
</div>

The division of labor is clear: **Post-norm preserves depth expression but has unstable gradients; Pre-norm has stable gradients but tends to collapse**. Modern LLMs overwhelmingly pick Pre-norm for a pragmatic reason: engineering-wise, "can it train stably" matters more than "is depth fully used".

## 3. Three Improvement Routes

<div class="fig">
<svg viewBox="0 0 660 280" role="img" aria-label="Residual structures of the three improvement routes">
  <title>Figure 3: Residual structures of the three improvement routes</title>
  <text x="115" y="26" font-size="12" fill="#6b7280" text-anchor="middle">Standard residual · 1 stream · C dims</text>
  <text x="325" y="26" font-size="12" fill="#6b7280" text-anchor="middle">mHC multi-stream · n streams · nC dims</text>
  <text x="535" y="26" font-size="12" fill="#6b7280" text-anchor="middle">AttnRes cross-layer attention · wired to all previous</text>
  <path d="M220 44 L220 272" stroke="#d1d5db" stroke-dasharray="3 3"/>
  <path d="M430 44 L430 272" stroke="#d1d5db" stroke-dasharray="3 3"/>
  <rect x="40" y="64" width="62" height="22" rx="4" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="71" y="79" font-size="11" fill="#185FA5" text-anchor="middle">F1</text>
  <rect x="40" y="106" width="62" height="22" rx="4" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="71" y="121" font-size="11" fill="#185FA5" text-anchor="middle">F2</text>
  <rect x="40" y="148" width="62" height="22" rx="4" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="71" y="163" font-size="11" fill="#185FA5" text-anchor="middle">F3</text>
  <rect x="152" y="56" width="10" height="126" rx="3" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="170" y="122" font-size="10" fill="#6b7280" text-anchor="start">C-dim stream</text>
  <g stroke="#378ADD" stroke-width="1.2" fill="none">
    <path d="M71 86 L71 104"/><path d="M67 100 L71 106 L75 100"/>
    <path d="M71 128 L71 146"/><path d="M67 142 L71 148 L75 142"/>
    <path d="M102 75 L146 75"/><path d="M142 71 L149 75 L142 79"/>
    <path d="M102 117 L146 117"/><path d="M142 113 L149 117 L142 121"/>
    <path d="M102 159 L146 159"/><path d="M142 155 L149 159 L142 163"/>
  </g>
  <text x="124" y="104" font-size="10" fill="#6b7280" text-anchor="middle">equal +1</text>
  <text x="115" y="246" font-size="11" fill="#6b7280" text-anchor="middle">fixed weights, plain summation</text>
  <text x="115" y="264" font-size="11" fill="#6b7280" text-anchor="middle">dilution + hidden-state blowup</text>
  <rect x="262" y="60" width="26" height="140" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <rect x="306" y="60" width="26" height="140" rx="4" fill="#e9e4fb" stroke="#7f77dd"/>
  <rect x="350" y="60" width="26" height="140" rx="4" fill="#e2dbf6" stroke="#7f77dd"/>
  <text x="325" y="48" font-size="10" fill="#6b7280" text-anchor="middle">n streams · nC dims</text>
  <g stroke="#7f77dd" stroke-width="1.2" fill="none">
    <path d="M290 112 L302 112"/><path d="M296 108 L289 112 L296 116"/><path d="M298 108 L305 112 L298 116"/>
    <path d="M334 112 L346 112"/><path d="M340 108 L333 112 L340 116"/><path d="M342 108 L349 112 L342 116"/>
    <path d="M290 152 L302 152"/><path d="M296 148 L289 152 L296 156"/><path d="M298 148 L305 152 L298 156"/>
    <path d="M334 152 L346 152"/><path d="M340 148 L333 152 L340 156"/><path d="M342 148 L349 152 L342 156"/>
  </g>
  <text x="325" y="98" font-size="10" fill="#6b7280" text-anchor="middle">H_res mixing (doubly stochastic)</text>
  <text x="325" y="246" font-size="11" fill="#6b7280" text-anchor="middle">widen the stream, learnable mixing</text>
  <text x="325" y="264" font-size="11" fill="#6b7280" text-anchor="middle">cross-stream exchange, numerically stable</text>
  <rect x="462" y="64" width="58" height="22" rx="4" fill="#faeeda" stroke="#BA7517"/>
  <text x="491" y="79" font-size="11" fill="#854F0B" text-anchor="middle">F4</text>
  <rect x="462" y="110" width="58" height="22" rx="4" fill="#faeeda" stroke="#BA7517"/>
  <text x="491" y="125" font-size="11" fill="#854F0B" text-anchor="middle">F3</text>
  <rect x="462" y="156" width="58" height="22" rx="4" fill="#faeeda" stroke="#BA7517"/>
  <text x="491" y="171" font-size="11" fill="#854F0B" text-anchor="middle">F2</text>
  <rect x="462" y="202" width="58" height="22" rx="4" fill="#faeeda" stroke="#BA7517"/>
  <text x="491" y="217" font-size="11" fill="#854F0B" text-anchor="middle">F1</text>
  <g stroke="#BA7517" stroke-width="1.2" fill="none">
    <path d="M520 121 C 556 121, 556 72, 542 72"/><path d="M548 68 L541 72 L548 76"/>
    <path d="M520 167 C 578 167, 578 75, 542 75"/><path d="M548 71 L541 75 L548 79"/>
    <path d="M520 213 C 598 213, 598 78, 542 78"/><path d="M548 74 L541 78 L548 82"/>
  </g>
  <text x="560" y="102" font-size="10" fill="#6b7280" text-anchor="start">α3</text>
  <text x="578" y="146" font-size="10" fill="#6b7280" text-anchor="start">α2</text>
  <text x="594" y="196" font-size="10" fill="#6b7280" text-anchor="start">α1</text>
  <text x="535" y="246" font-size="11" fill="#6b7280" text-anchor="middle">weighted sum replaces equal weight</text>
  <text x="535" y="264" font-size="11" fill="#6b7280" text-anchor="middle">Block N=8 captures most gains</text>
</svg>
<figcaption>Figure 3: The three routes compared. The standard residual has a single C-dimensional stream and sums all previous layers with equal weight; mHC widens the stream into n nC-dimensional streams with learnable mixing (the mixing matrix constrained to be doubly stochastic); AttnRes wires each layer directly to the outputs of all previous layers, replacing the fixed weight of 1 with attention weights.</figcaption>
</div>

### 3.1 Hyper-Connections (ByteDance, 2024) — Widen the Residual Stream

The core move expands the residual stream from C dimensions to **nC dimensions** (n is the expansion rate) and introduces learnable **depth connections** (inter-layer weighting) and **width connections** (information exchange across streams). One HC layer decouples into these two parts, with output written as `Ĥ = Bᵀ·T(HᵀA_m)ᵀ + A_rᵀ·H`.

The shortcoming is blunt: **the memory wall**. Widening the stream by n times (commonly n = 4) multiplies memory footprint and I/O, making large-scale pre-training extremely expensive. ICLR 2025 reviewers gave it a Borderline Accept — acknowledging the performance but questioning the theoretical novelty and engineering cost.

### 3.2 mHC (DeepSeek, 2026) — Buy Stability with Doubly Stochastic Matrices

The single-layer form of HC is `x_{l+1} = H_res·x_l + H_postᵀ·F(H_pre·x_l)`. **The expressive power comes from H_res, and so does the instability** — in the original HC, H_res is an unconstrained matrix; after chaining through layers as ∏H_i, it explodes when amplified and vanishes when decayed.

mHC does not degenerate into the identity matrix (that would lose the cross-stream exchange capability); instead it constrains H_res to the **doubly stochastic matrix set (Birkhoff polytope)**: row sums = 1, column sums = 1, all entries non-negative. H_pre and H_post also get non-negativity constraints (mixing positive and negative entries causes information cancellation).

<div class="fig">
<svg viewBox="0 0 660 284" role="img" aria-label="The doubly stochastic constraint of mHC">
  <title>Figure 4: The doubly stochastic constraint of mHC</title>
  <text x="20" y="24" font-size="12" fill="#6b7280">H_res constrained to doubly stochastic matrices (Birkhoff polytope)</text>
  <text x="20" y="42" font-size="12" fill="#6b7280">row sums = column sums = 1, all entries non-negative</text>
  <text x="170" y="62" font-size="11" fill="#534AB7">H_res (3 streams)</text>
  <rect x="170" y="68" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="197" y="92" font-size="12" fill="#534AB7" text-anchor="middle">0.3</text>
  <rect x="228" y="68" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="255" y="92" font-size="12" fill="#534AB7" text-anchor="middle">0.5</text>
  <rect x="286" y="68" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="313" y="92" font-size="12" fill="#534AB7" text-anchor="middle">0.2</text>
  <rect x="170" y="110" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="197" y="134" font-size="12" fill="#534AB7" text-anchor="middle">0.1</text>
  <rect x="228" y="110" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="255" y="134" font-size="12" fill="#534AB7" text-anchor="middle">0.3</text>
  <rect x="286" y="110" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="313" y="134" font-size="12" fill="#534AB7" text-anchor="middle">0.6</text>
  <rect x="170" y="152" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="197" y="176" font-size="12" fill="#534AB7" text-anchor="middle">0.6</text>
  <rect x="228" y="152" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="255" y="176" font-size="12" fill="#534AB7" text-anchor="middle">0.2</text>
  <rect x="286" y="152" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="313" y="176" font-size="12" fill="#534AB7" text-anchor="middle">0.2</text>
  <text x="346" y="92" font-size="12" fill="#374151" text-anchor="start">= 1</text>
  <text x="346" y="134" font-size="12" fill="#374151" text-anchor="start">= 1</text>
  <text x="346" y="176" font-size="12" fill="#374151" text-anchor="start">= 1</text>
  <text x="197" y="208" font-size="12" fill="#374151" text-anchor="middle">= 1</text>
  <text x="255" y="208" font-size="12" fill="#374151" text-anchor="middle">= 1</text>
  <text x="313" y="208" font-size="12" fill="#374151" text-anchor="middle">= 1</text>
  <text x="170" y="226" font-size="10" fill="#6b7280" text-anchor="start">row sums = 1 ｜ column sums = 1 ｜ entries ≥ 0</text>
  <path d="M396 52 L396 240" stroke="#d1d5db" stroke-dasharray="3 3"/>
  <text x="412" y="80" font-size="12" fill="#185FA5" text-anchor="start">Mean preserved</text>
  <text x="412" y="98" font-size="10.5" fill="#6b7280" text-anchor="start">1ᵀH = 1ᵀ — mean signal is layer-invariant</text>
  <text x="412" y="128" font-size="12" fill="#185FA5" text-anchor="start">No amplification</text>
  <text x="412" y="146" font-size="10.5" fill="#6b7280" text-anchor="start">‖H‖₂ ≤ 1 ⇒ ‖Hx‖₂ ≤ ‖x‖₂ (norm capped)</text>
  <text x="412" y="176" font-size="12" fill="#185FA5" text-anchor="start">Products stay stable</text>
  <text x="412" y="194" font-size="10.5" fill="#6b7280" text-anchor="start">closed under multiplication ⇒ ∏H_i stays in the set</text>
  <text x="20" y="256" font-size="11" fill="#6b7280">HC (unconstrained): 10·x₁ − 8·x₂ + 5·x₃ → can blow up</text>
  <text x="20" y="274" font-size="11" fill="#534AB7">mHC (doubly stochastic): 0.3·x₁ + 0.5·x₂ + 0.2·x₃ → stable weighted average</text>
</svg>
<figcaption>Figure 4: mHC constrains the mixing matrix H_res to the doubly stochastic set. Row and column sums of 1 with non-negative entries give three properties: the mean signal across streams is layer-invariant, the spectral norm does not exceed 1, and chained products stay in the set — the unconstrained HC combination 10·x₁ − 8·x₂ + 5·x₃ is replaced by a stable weighted average.</figcaption>
</div>

The three properties in one sentence: **mean preserved, no amplification, products still stable**.

**How it is generated**: flatten the n×C multi-stream state and apply RMSNorm, produce three unconstrained logits, then project them separately — `H_pre = σ(·)`, `H_post = 2σ(·)`, `H_res = Sinkhorn-Knopp(·)` (alternating row-column normalization, about 20 iterations).

The engineering companions (what actually makes mHC deployable):

| Optimization | Approach | Saves |
|--------------|----------|-------|
| **Kernel Fusion** | Fuse the three projections into one matmul (nC → n²+2n); multiply-then-divide RMSNorm; put the whole Sinkhorn-Knopp into a single custom kernel | One nC-dimensional intermediate read/write plus repeated kernel launches |
| **Selective recomputation** | Pack L_r consecutive layers into a block; forward stores only the block start input; backward recomputes the lightweight part; optimal block size `L_r* ≈ √(nL/(n+2))` | Activation memory |
| **DualPipe comm overlap** | mHC must pass nC dimensions instead of C between PP stages; use three stream classes (regular compute / communication / high-priority small compute) to overlap communication with computation | Pipeline bubbles |

DeepSeek has already put mHC into **DeepSeek-V4** — the only one of the three routes inside a real model.

### 3.3 Attention Residuals (Kimi, 2026) — Rotate Attention by 90 Degrees

The idea steps entirely outside the "widen the stream" frame: the Kimi team argues that **the depth dimension of a network and the time dimension of a sequence are essentially isomorphic**. If every token on the sequence can attend to all previous tokens, why not let every layer attend to the outputs of all previous layers?

So equal-weight summation becomes weighted summation: `h_l = Σ α_{i→l} · v_i`, where `α_{i→l} = softmax(q_lᵀ·RMSNorm(k_i))`, `q_l = w_l` (learnable), and k_i = v_i is the output of layer i.

Full AttnRes is theoretically perfect, but every layer must access the outputs of all previous layers: compute O(L²d), memory and communication O(Ld). Engineering-wise, take **Block AttnRes**: split L layers into N blocks, keep traditional residuals inside a block and attention residuals between blocks. The rule of thumb **N = 8** captures most of the Full version's gain, with complexity down to Memory O(N) and Computation O(N²).

There is also a two-phase inference strategy: since queries are static (w_l), the queries of S layers inside a block can be packed into one MatMul — **Phase 1** computes inter-block attention in one shot and returns the online-softmax state, **Phase 2** serially computes the local sums inside the block and updates the attention weights, reducing memory visits from S times to 1.

## 4. Experiments (Kimi Linear 48B-A3B Structure)

| Active params | Layers | batch | Baseline | Block AttnRes | Full AttnRes | mHC(-lite) |
|---------------|--------|-------|----------|---------------|--------------|------------|
| 194M | 24 | 192 | 1.931 | 1.909 | **1.899** | 1.906 |
| 241M | 26 | 256 | 1.895 | 1.875 | **1.874** | 1.869 |
| 296M | 28 | 320 | 1.829 | 1.809 | **1.804** | 1.807 |
| 436M | 32 | 384 | 1.766 | 1.746 | **1.737** | 1.747 |
| 528M | 34 | 432 | 1.719 | 1.693 | **1.692** | 1.694 |

Overall ordering: **Full AttnRes > Block AttnRes ≈ mHC > Baseline**. Notably Block AttnRes (N = 8) is already very close to the Full version, and mHC sits in the same band — the two routes gain comparably; the difference is engineering cost.

## 5. My Take

- **HC and mHC are the same route**: both widen the inter-layer residual stream to hold more information, without changing hidden_size or adding Attention/FFN compute. The doubly stochastic constraint is the step that makes this route actually usable.
- **mHC is the strongest candidate for de facto standard**: the only one inside a real model (DeepSeek-V4), and paired with a full engineering stack of kernel fusion / recomputation / DualPipe — the performance is not stuck in theory.
- **AttnRes has more imagination but lands further away**. The authors offer a good analogy: going from the standard residual to HC/mHC is like RNN to LSTM; attention residuals are more like RNN to Transformer. Attention has been proven to solve long-range dependencies at the cost of enormous compute — the Transformer took five years from 2017 to truly dominate in 2022. AttnRes will probably need an equally long engineering adaptation period.
- **Engineering takeaway**: if you work on inference adaptation (say, large-model inference optimization on domestic accelerators), the mHC residual stream expanding from C to nC means extra memory and cross-stage communication beyond the KV cache, so kernel fusion and recomputation must be reimplemented yourself; the AttnRes in-block query packing plus two-phase online-softmax is a memory-access optimization pattern well worth borrowing.

## 6. Links to the Rest of the Series

- **fa2 (Kimi K3)**: AttnRes comes from the Kimi team — pushing the idea of "redesigning architecture with attention" from within the sequence to between layers, in the same lineage as fa2's attention redesign.
- **fa4 (DeepSeek V4)**: mHC has entered DeepSeek-V4 and is one of the underlying components of fa4's extreme compression and efficient training route.
- **fa6 (Gated DeltaNet)**: linear attention changes "how to compute inside the sequence"; this post changes "how to pass information between layers" — the two dimensions do not replace each other.
- **sys9 / sys12**: the mHC nC residual stream means extra memory and cross-stage communication beyond the KV cache; use those two posts' formulas and tools to estimate.
- **op1 / op2**: RMSNorm, Sinkhorn-Knopp normalization, and kernel fusion at the operator level — compare with the operator-series notes.
