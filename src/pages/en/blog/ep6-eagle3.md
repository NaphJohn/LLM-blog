---
title: "EAGLE-3 Deep Dive: Drop the Feature Constraint, Let the Drafter Finally Scale with Data"
description: Starting from why feature prediction is a scaling lock, this post breaks down EAGLE-3's Training-Time Test and multi-layer feature fusion, with a three-generation comparison table and a scaling curve.
pubDate: 2026-08-04
series: Speculative Decoding Notes
lang: en
altLang: zh
altHref: /blog/ep6-eagle3
layout: ../../../layouts/BlogPost.astro
---

## 1. Why a dedicated look at EAGLE-3

Ep1–Ep5 covered the shape of speculative decoding (SD), the ceiling of autoregressive drafters, and the two new routes DFlash / DSpark. But one name keeps showing up — **EAGLE** — the originator of the whole "feature-level drafting" school. DFlash and DSpark both use it as the baseline ("about 2.5× faster than EAGLE-3").

The EAGLE lineage has three generations:

- **EAGLE-1** (ICML 2024): autoregresses at the feature level, borrowing the target model's top-layer features to guess more accurately;
- **EAGLE-2** (EMNLP 2024): dynamically schedules the draft tree by confidence, allocating compute more cleverly;
- **EAGLE-3** (2025): **drops the feature-prediction constraint**, giving the draft model its first scaling law.

What makes EAGLE-3 remarkable is not "a bit faster" — it **proved that a draft model can keep benefiting from more data, exactly like a large model**. That is something the speculative-decoding field had never seen before.

## 2. Recap: how EAGLE guesses better

Vanilla SD uses a small LLM as the drafter; the small model's distribution diverges from the large model's, so acceptance rate α is low (~1.6–1.9×).

EAGLE-1's insight: **instead of a separate small model guessing tokens, reuse the target model's intermediate features**. It autoregresses at the feature level — concatenating the target's top-layer feature with a "one-step-ahead token" and feeding it to a 1-layer draft Decoder that predicts the next feature, then uses the target's LM head to turn it into a token. This reached ~3–4×.

EAGLE-2 found that the drafter's confidence is highly calibrated with the actual acceptance rate, so it **dynamically shapes the draft tree** (fewer branches for easy positions, more for hard ones), pushing speedup to ~4.2×.

## 3. The feature-prediction constraint: why it is a scaling lock

EAGLE-1/2's drafter is trained under **two losses**:

- **feature loss** (SmoothL1): make the draft output approximate the target model's top-layer feature;
- **token loss** (CrossEntropy): get the token right.

When the authors tried scaling training data 8×, speedup **barely moved**. The culprit is the feature loss:

> The drafter has only 1 Decoder layer and tiny capacity. The feature loss forces it to spend capacity on "geometric fitting" — placing the output vector close to the target's feature in space — instead of on the real goal: predicting the token as accurately as possible.

In other words, feature prediction is an **over-strong regularizer**: it helps generalization when data is scarce, but becomes a lock when data is abundant. Worse, the feature constraint also **locks the input** to "top-layer features only" (since top-layer features and next-token logits are one-to-one).

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#374151">Figure 1: Draft training data volume vs speedup (LLaMA-Instruct 3.1 8B, MT-bench, schematic)</text>
  <line x1="100" y1="40" x2="100" y2="250" stroke="#9ca3af" stroke-width="1"/>
  <line x1="100" y1="250" x2="540" y2="250" stroke="#9ca3af" stroke-width="1"/>
  <text x="92" y="254" font-size="10" fill="#6b7280" text-anchor="end">3.5</text>
  <text x="92" y="191" font-size="10" fill="#6b7280" text-anchor="end">4.5</text>
  <text x="92" y="127" font-size="10" fill="#6b7280" text-anchor="end">5.5</text>
  <text x="92" y="64" font-size="10" fill="#6b7280" text-anchor="end">6.5</text>
  <polyline points="120,218 220,212 330,212 450,212" fill="none" stroke="#3b82f6" stroke-width="2"/>
  <polyline points="120,142 220,117 330,98 450,85" fill="none" stroke="#eab308" stroke-width="2.5"/>
  <circle cx="120" cy="218" r="3" fill="#3b82f6"/><circle cx="220" cy="212" r="3" fill="#3b82f6"/><circle cx="330" cy="212" r="3" fill="#3b82f6"/><circle cx="450" cy="212" r="3" fill="#3b82f6"/>
  <circle cx="120" cy="142" r="3" fill="#eab308"/><circle cx="220" cy="117" r="3" fill="#eab308"/><circle cx="330" cy="98" r="3" fill="#eab308"/><circle cx="450" cy="85" r="3" fill="#eab308"/>
  <text x="120" y="268" font-size="10" fill="#6b7280" text-anchor="middle">1×</text>
  <text x="220" y="268" font-size="10" fill="#6b7280" text-anchor="middle">2×</text>
  <text x="330" y="268" font-size="10" fill="#6b7280" text-anchor="middle">4×</text>
  <text x="450" y="268" font-size="10" fill="#6b7280" text-anchor="middle">8×</text>
  <text x="320" y="286" font-size="10" fill="#6b7280" text-anchor="middle">Training data volume (× relative to ShareGPT)</text>
  <rect x="498" y="58" width="14" height="10" fill="#3b82f6"/><text x="518" y="67" font-size="10" fill="#374151">EAGLE / EAGLE-2 (flat)</text>
  <rect x="498" y="78" width="14" height="10" fill="#eab308"/><text x="518" y="87" font-size="10" fill="#374151">EAGLE-3 (rises w/ data)</text>
  <text x="16" y="290" font-size="10" fill="#6b7280">After dropping feature prediction, the drafter shows its first scaling law: more data, higher speedup, up to 6.5×.</text>
</svg>
</div>

## 4. EAGLE-3's two changes

Two designs that work together — neither alone is enough.

### 4.1 Drop feature prediction, predict tokens directly via Training-Time Test

Core insight: **feature prediction is a means, not an end** — its only job was letting single-step training generalize to multi-step inference, at the cost of over-constraining the output. EAGLE-3 makes the drafter predict tokens directly, and **simulates the multi-step generation process during training** (wiring the target's LM head and sampling into the drafter's training loop). This is called **Training-Time Test**: the drafter practices, at training time, exactly the multi-step generation it will do at test time, eliminating the train–test distribution gap that plagued EAGLE.

<div class="fig">
<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#374151">Figure 2: EAGLE-2 (feature-constrained) vs EAGLE-3 (Training-Time Test)</text>
  <text x="175" y="42" font-size="12" font-weight="700" fill="#1e3a8a" text-anchor="middle">EAGLE-2</text>
  <text x="505" y="42" font-size="12" font-weight="700" fill="#854d0e" text-anchor="middle">EAGLE-3</text>

  <rect x="60" y="58" width="230" height="38" rx="5" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="175" y="81" font-size="11" fill="#1e3a8a" text-anchor="middle">Input: feature f_t + ahead token t₍ₜ₊₁₎</text>
  <line x1="175" y1="96" x2="175" y2="116" stroke="#9ca3af" stroke-width="1"/>
  <rect x="60" y="116" width="230" height="38" rx="5" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="175" y="139" font-size="11" fill="#1e3a8a" text-anchor="middle">Draft model (1-layer Decoder)</text>
  <line x1="175" y1="154" x2="175" y2="174" stroke="#9ca3af" stroke-width="1"/>
  <rect x="60" y="174" width="230" height="42" rx="5" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="175" y="195" font-size="11" fill="#1e3a8a" text-anchor="middle">Predict f₍ₜ₊₁₎</text>
  <text x="175" y="210" font-size="10" fill="#1e3a8a" text-anchor="middle">(SmoothL1 fits target feature)</text>
  <line x1="175" y1="216" x2="175" y2="236" stroke="#9ca3af" stroke-width="1"/>
  <rect x="60" y="236" width="230" height="34" rx="5" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="175" y="257" font-size="11" fill="#1e3a8a" text-anchor="middle">LM head → token</text>
  <text x="175" y="298" font-size="10" fill="#6b7280" text-anchor="middle">Single-step train→multi-step; feature lock caps capacity & input</text>

  <rect x="390" y="58" width="230" height="38" rx="5" fill="#fef9c3" stroke="#eab308"/>
  <text x="505" y="81" font-size="11" fill="#854d0e" text-anchor="middle">Input: multi-layer fused feature + prev token</text>
  <line x1="505" y1="96" x2="505" y2="116" stroke="#9ca3af" stroke-width="1"/>
  <rect x="390" y="116" width="230" height="38" rx="5" fill="#fef9c3" stroke="#eab308"/>
  <text x="505" y="139" font-size="11" fill="#854d0e" text-anchor="middle">Draft model (capacity scalable)</text>
  <line x1="505" y1="154" x2="505" y2="174" stroke="#9ca3af" stroke-width="1"/>
  <rect x="390" y="174" width="230" height="42" rx="5" fill="#fef9c3" stroke="#eab308"/>
  <text x="505" y="195" font-size="11" fill="#854d0e" text-anchor="middle">Predict token directly</text>
  <text x="505" y="210" font-size="10" fill="#854d0e" text-anchor="middle">(CrossEntropy loss)</text>
  <line x1="505" y1="216" x2="505" y2="236" stroke="#9ca3af" stroke-width="1"/>
  <rect x="390" y="236" width="230" height="46" rx="5" fill="#fef9c3" stroke="#eab308"/>
  <text x="505" y="256" font-size="10" fill="#854d0e" text-anchor="middle">Training simulates multi-step generation:</text>
  <text x="505" y="271" font-size="10" fill="#854d0e" text-anchor="middle">wire LM head + sampling into training loop</text>
  <text x="505" y="298" font-size="10" fill="#6b7280" text-anchor="middle">Train = "rehearse" test-time multi-step, kills distribution gap</text>
</svg>
</div>

### 4.2 Multi-layer feature fusion

With the feature constraint gone, the input is free too — instead of top-layer features only, EAGLE-3 **fuses low / mid / high-level features** from the target model, giving the drafter richer semantic context. This and "direct token prediction" are mutually prerequisite: without the feature constraint, you would not dare to change the input.

## 5. Three-generation comparison

| Dimension | EAGLE-1 | EAGLE-2 | EAGLE-3 |
|---|---|---|---|
| Draft style | feature-level AR | dynamic draft tree | direct token + multi-layer fusion |
| Feature constraint | yes | yes | **no** |
| Input | top-layer feature | top-layer feature | low/mid/high fusion |
| Scaling law | no | no | **yes** |
| Typical speedup | ~3–4× | ~4.2× | **up to 6.5×** |

## 6. Results

| Metric | Value |
|---|---|
| Max speedup | **6.5×** (Vicuna 13B, HumanEval, T=0) |
| vs EAGLE-2 | ~**1.4×** lower latency |
| SGLang throughput (batch=64, H100) | **+1.38×** |
| Compatibility | fully compatible with EAGLE-2's draft tree |

## 7. Takeaway

The EAGLE lineage in one line:

> EAGLE-1 solved "how to borrow the large model's info to guess better" → EAGLE-2 solved "how to spend the compute budget smarter" → EAGLE-3 solved "how to keep benefiting from more data."

It sends a signal to the whole SD field: **a drafter need not be a tiny toy — drop the wrong constraint, give it richer input and more capacity, and it can scale too**. That is why DFlash / DSpark both benchmark against EAGLE-3: it defines the ceiling of the "feature-level drafting" route.

> This post belongs to the "Speculative Decoding Notes" series, Ep6. See also: [DFlash Deep Dive](../ep3-dflash), [DSpark Deep Dive](../ep4-dspark), [DFlash vs DSpark](../ep5-dflash-vs-dspark); back to the series index at [Speculative Decoding Notes](../).
