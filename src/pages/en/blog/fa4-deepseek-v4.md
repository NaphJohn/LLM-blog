---
title: 'Frontier Architecture Decoding Notes (4): DeepSeek V4 — Extreme Compression and Efficient Training'
description: 'The finale. DeepSeek V4 takes the most aggressive road: compress KV itself — CSA (Compressed Sparse Attention, 4-token KV blocks + top-k) + HCA (Heavily Compressed Attention, 128×) hybrid, plus a 128-token uncompressed sliding-window branch; at 1M, per-token compute ~27% of V3.2, KV ~10%. Training swaps in the Muon optimizer, OPD distillation, FP8 training / FP4 expert params. Pro 1.6T / Flash 284B (MoE), MIT open.'
pubDate: 2026-08-06
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa4-deepseek-v4
layout: ../../../layouts/BlogPost.astro
---

## 0. Finale: the Most Aggressive "Compression" Road

The first two: Kimi K3 changed attention **internals** (delta), MiniMax M3 changed attention's **scope** (sparse). DeepSeek V4 goes further — **compresses the KV cache itself** and rebuilds training too. This is the heaviest modification stop on the roadmap.

## 1. Attention: CSA + HCA Hybrid

- **CSA (Compressed Sparse Attention)**: compress KV into **4-token blocks**, then top-k over blocks — both compressed and sparsely selected.
- **HCA (Heavily Compressed Attention)**: compression ratio up to **128×**, for the "most distant, most compressible" history.
- **Uncompressed sliding-window branch**: the most recent **128 tokens** use plain attention, keeping local fine modeling intact.

The three mix into a layered "precise near, compressed far" structure.

<div class="fig">
<svg viewBox="0 0 680 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="CSA+HCA hybrid attention structure">
  <rect x="0" y="0" width="680" height="240" fill="none"/>
  <text x="20" y="28" font-size="14" font-weight="700" fill="#1a1a1a">Query position → layered handling of historical KV</text>
  <rect x="20" y="60" width="160" height="50" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="100" y="90" font-size="13" fill="#047857" text-anchor="middle">recent 128 tokens</text>
  <rect x="200" y="60" width="160" height="50" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="280" y="90" font-size="13" fill="#1d4ed8" text-anchor="middle">CSA (4-token blocks + top-k)</text>
  <rect x="380" y="60" width="200" height="50" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="480" y="82" font-size="13" fill="#b45309" text-anchor="middle">HCA (heavy compress 128×)</text>
  <text x="480" y="100" font-size="11" fill="#b45309" text-anchor="middle">most distant history</text>
  <line x1="180" y1="85" x2="198" y2="85" stroke="#6b7280"/>
  <line x1="360" y1="85" x2="378" y2="85" stroke="#6b7280"/>
  <rect x="20" y="140" width="640" height="70" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="40" y="166" font-size="12.5" fill="#1a1a1a">Hybrid effect (official, vs V3.2 @ 1M):</text>
  <text x="40" y="188" font-size="12.5" fill="#1a1a1a">per-token compute ≈ 27% · KV cache ≈ 10% · default 1M context</text>
  <text x="40" y="205" font-size="11" fill="#6b7280">precise near (window) + mid compress (CSA) + far extreme compress (HCA)</text>
</svg>
  <p class="cap">Figure: DeepSeek V4's layered attention — uncompressed near, harder-compressed the farther away.</p>
</div>

## 2. Training Side: Muon + OPD + Low-Bit

- **Muon optimizer** replaces AdamW: for orthogonal / moment matrices Muon (momentum + orthogonalization) saves memory and stabilizes convergence — the key to V4's cheaper training.
- **OPD (On-Policy Distillation)**: the main model distills "domain-expert" small models online, compressing specialized capability into the unified model and avoiding offline-distillation distribution shift.
- **FP8 training / FP4 expert params**: MoE expert weights in FP4, further cutting training/storage cost.
- **MegaMoE EP communication**: expert-parallel comm optimization, 1.5–1.73× faster; plus **disk KV cache**, TileLang DSL, Ascend NPU adaptation.

## 3. Two Variants and Pricing

| Variant | Params (MoE) | Positioning | Price (per M tokens) |
|---|---|---|---|
| **Flash** | 284B / 13B active | light, low-latency | input 1 ¥ / output 2 ¥ (cache-hit input 0.2 ¥) |
| **Pro** | 1.6T / 49B active | strong reasoning | input 12 ¥ / output 24 ¥ |

Both default to 1M context, max output ~384K; **MIT-licensed and open**.

## 4. Benchmarks (official)

- Codeforces rating **3206** (V4-Pro-Max);
- SWE-Verified **80.6%**;
- Toolathlon **51.8**;
- Putnam 2025 **full marks**.

Code, agentic tool-use, and competition math — consistent with "extreme compression buys long context + strong reasoning."

## 5. The Three Converge: a Shared Destination

Stringing the four posts together:

| Route | Representative | Attention change | 1M-context cost |
|---|---|---|---|
| Redesign | Kimi K3 | KDA delta + gate(3:1) + residual | saves KV via MLA |
| Sparsify | MiniMax M3 | MSA sparse selection | per-token compute ~1/20 of M2.7 |
| Compress | DeepSeek V4 | CSA+HCA compress KV | per-token compute ~27% of V3.2, KV ~10% |

<div class="keybox">
<strong>Conclusion:</strong> the three differ in "where they cut" (internals / scope / KV itself), but the <strong>destination is identical — MoE + 1M context + attention redesign</strong>. After 2026, dense LLMs essentially retire; "long context + low KV" becomes the frontier model's passing bar.
</div>

## 6. Investment View

- **The inference-cost curve keeps dropping**: KV and per-token compute compressed to 1/10–1/20, directly benefiting **on-device / onboard real-time models** and **long-horizon Agents** — the cost inflection for embodied AI is closer.
- **Capability commoditization accelerates**: all three open-weight (MIT/Apache); "frontier model capability" is no longer the moat — the **moat shifts to data, engineering, ecosystem, and scenario**.
- **Training cost-down (Muon / FP4 / OPD)** means small teams can also train strong models, benefiting **domestic compute and vertical-model** ecosystems.
