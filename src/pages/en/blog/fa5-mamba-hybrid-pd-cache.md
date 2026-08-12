---
title: 'Frontier Architecture Decoding Notes (5): Mamba Hybrid State + PD Disaggregation — Why Cache Correctness Must Avoid "Silent Hit Misplacement"'
description: 'The first four posts covered how frontier models "reshape attention"; this one turns to "how to serve the resulting hybrid architectures." Using Jamba-style Mamba+Attention hybrid models as the example, it breaks down the "hybrid state" (KV cache + Mamba SSM recurrent state) that must be carried across the Prefill-Decode boundary in PD-disaggregated deployment, explains why the Mamba state has no token-id key and thus silently mis-hits under KV offload plus multi-connector setups, and shows how fingerprint / version / connector-level state-id keep cache correctness.'
pubDate: 2026-08-10
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa5-mamba-hybrid-pd-cache
layout: ../../../layouts/BlogPost.astro
---

## 0. Why a Standalone Post

The first four posts (overview → Kimi K3 → MiniMax M3 → DeepSeek V4) covered **how to reshape attention so it holds 1M context without blowing up KV and compute**. But from fa3 and fa4 a second thread is visible: **attention is being partly replaced by "non-attention"** — MiniMax's MSA sparsifies it, DeepSeek's CSA/HCA compress it, and the more radical step is to swap some layers for **Mamba-style state-space models (SSM)**.

That yields a class of "hybrid architectures": Attention layers interleaved with Mamba layers (classically Jamba, one Attention layer per eight Mamba layers). When the architecture changes, **the "state" you must maintain at serving / inference time changes too** — and that is exactly where engineering footguns hide and get ignored. This post takes it apart.

<figure class="arch-fig">
<svg viewBox="0 0 680 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="State transfer and silent mis-hit risk for hybrid models under PD disaggregation">
  <defs>
    <marker id="arrowEn" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#8B4513"/>
    </marker>
    <style>
      .box{fill:#fff7ed;stroke:#c2410c;stroke-width:1.5}
      .box2{fill:#eff6ff;stroke:#1d4ed8;stroke-width:1.5}
      .gate{fill:#fef9c3;stroke:#a16207;stroke-width:1.5}
      .lab{font:600 13px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .sub{font:11px -apple-system,'PingFang SC',sans-serif;fill:#475569}
      .warn{font:600 11px -apple-system,'PingFang SC',sans-serif;fill:#b91c1c}
      .conn{font:10px -apple-system,'PingFang SC',sans-serif;fill:#6b7280}
    </style>
  </defs>

  <rect class="box" x="20" y="50" width="190" height="150" rx="8"/>
  <text class="lab" x="115" y="74" text-anchor="middle">Prefill node</text>
  <text class="sub" x="115" y="98" text-anchor="middle">produces "hybrid state"</text>
  <line x1="40" y1="112" x2="190" y2="112" stroke="#c2410c" stroke-dasharray="3 3"/>
  <text class="sub" x="115" y="132" text-anchor="middle">① KV cache</text>
  <text class="sub" x="115" y="148" text-anchor="middle">(token-id keyed)</text>
  <text class="sub" x="115" y="172" text-anchor="middle">② Mamba SSM state</text>
  <text class="sub" x="115" y="188" text-anchor="middle">(fixed-size, no token-id key)</text>

  <rect class="gate" x="250" y="95" width="180" height="60" rx="8"/>
  <text class="lab" x="340" y="118" text-anchor="middle">RDMA / Mooncake</text>
  <text class="sub" x="340" y="138" text-anchor="middle">raw bytes: ptr+index×item_len</text>
  <line x1="210" y1="125" x2="248" y2="125" stroke="#8B4513" stroke-width="2" marker-end="url(#arrowEn)"/>
  <line x1="432" y1="125" x2="470" y2="125" stroke="#8B4513" stroke-width="2" marker-end="url(#arrowEn)"/>

  <rect class="box2" x="470" y="50" width="190" height="150" rx="8"/>
  <text class="lab" x="565" y="74" text-anchor="middle">Decode node</text>
  <text class="sub" x="565" y="98" text-anchor="middle">consumes hybrid state</text>
  <text class="sub" x="565" y="132" text-anchor="middle">autoregressive by token</text>
  <text class="sub" x="565" y="148" text-anchor="middle">generate next segment</text>
  <text class="sub" x="565" y="172" text-anchor="middle">if state misplaced →</text>
  <text class="warn" x="565" y="188" text-anchor="middle">silent wrong output</text>

  <rect class="gate" x="250" y="252" width="380" height="70" rx="8"/>
  <text class="lab" x="440" y="276" text-anchor="middle">Cache-correctness guard (absence ⇒ silent mis-hit)</text>
  <text class="sub" x="440" y="298" text-anchor="middle">fingerprint (layer+req id+step+prefix hash)</text>
  <text class="sub" x="440" y="314" text-anchor="middle">version / sequence no · connector-level state-id (route by id, not position)</text>

  <text class="conn" x="340" y="240" text-anchor="middle">↓ must do virtual→physical id translation at handoff; forbid compaction while transfer in flight</text>
  <path d="M340,155 C340,200 340,212 340,252" fill="none" stroke="#a16207" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrowEn)"/>
</svg>
<figcaption>Fig: Under PD disaggregation, a hybrid model must carry the bundle "KV cache + Mamba SSM state" across nodes. The SSM state has no token-id key; once misplaced / aliased / read stale in routing, the Decode side "thinks it hit" but gets wrong history, and the error silently accumulates along the sequence — no error thrown. The guard = give both state types a fingerprint / version / state-id so mis-hits become observable.</figcaption>
</figure>

## 1. What "Mamba Hybrid State" Means

In a hybrid model, the "memory" needed to advance the sequence comes from two streams:

- **Attention layers → KV cache**: grows with sequence length, stored in token-sized blocks, with the token sequence as a natural "addressing key."
- **Mamba layers → SSM recurrent state (hidden state h)**: fixed-size, independent of sequence length, updated recursively as `hₜ = Ā·hₜ₋₁ + B·xₜ`, `yₜ = C·hₜ`.

The "**hybrid state**" is the whole bundle — **KV cache + Mamba SSM state** — that PD (Prefill-Decode) disaggregation must carry across nodes from Prefill to Decode. It is called "hybrid," not "Mamba state," because both kinds of state must be managed correctly; neither alone is enough.

## 2. PD Disaggregation: Pure-Attention vs Hybrid Carry Different Things

```
Pure-attention model (e.g. Llama)
  Prefill output ──cross-node──▶ Decode carries only: KV cache (token-id hashed)
                              wrong blocks surface as obvious degradation / miss → not "silent"

Mamba hybrid model (e.g. Jamba)
  Prefill output ──cross-node──▶ Decode must carry BOTH:
     A. KV cache (Attention layers, grows with seq, token-id addressed)
     B. Mamba SSM state (Mamba layers, fixed-size, position-dependent, no token-id key)
```

A pure-attention model carries only KV across the boundary; a hybrid model **must carry both**, and stream B has no "natural key" like Attention does.

## 3. Why Hybrids Are More Prone to "Silent Hit Misplacement"

The key difference is the **addressing key**:

- **KV cache has a natural key**: the token sequence itself is the key, and prefix caches like vLLM's use block hashes to verify. A wrong block usually shows up as obvious degradation or a miss — **easy to catch**.
- **Mamba SSM state has no token-id key**: it is a fixed-size contiguous tensor, addressable only by "position / request context." Once it is misrouted, mis-numbered, or read stale during transport, the Decode side "thinks it hit" the correct state but actually got another prefix's / another request's state — the model **keeps generating, but the content is already wrong, and throws no error**.

That is "**silent hit misplacement**": unlike a wrong KV block that is "loud," it **emits errors silently**. Even a pure-attention model's wrong KV tends to surface; a Mamba state misplacement is stealthier, which is why "cache correctness" targets it specifically.

<div class="warnbox">
<strong>⚠️ Where the stealth comes from:</strong> an SSM state misplacement triggers no cache miss and no explicit error; it merely makes every subsequent hₜ recurse on the wrong history — the error accumulates quietly along the sequence, yet the first-token distribution often "looks normal," so a manual spot check rarely catches it at a glance.
</div>

## 4. KV Offload + Multi-Connector Amplify the Risk

The phrase "KV offload + Mamba hybrid state + multi-connector PD disaggregation" stacks three amplifiers:

- **KV offload (to CPU / NVMe)**: state no longer lives only in GPU memory; it moves in and out of storage tiers, adding a serialization / deserialization and addressing step.
- **Multi-connector**: state may flow between Prefill and Decode over different channels; routing gets more complex and block-misplacement probability rises.
- **Hybrid state**: the two factors above hit both KV and SSM, and the SSM stream has no token-id check.

Together, without forced verification on SSM blocks, the chance that the Decode side "hits" a wrong / stale state without error rises markedly.

## 5. How to Fix Cache Correctness: Fingerprint the Hybrid State

The answer is not "no offload" or "no multi-connector," but to **give the hybrid state (KV + SSM together) a verifiable identity**:

- **Fingerprint**: derive a unique fingerprint per state block from `layer id + request id + step id + prefix hash`, and compare on transfer and on load.
- **Version / sequence number**: Prefill-tagged state carries a version; Decode verifies version match before loading and rejects stale blocks.
- **Connector-level state-id**: in multi-connector routing, bind each state block to a globally unique state-id and route by id rather than by "position," avoiding mis-numbering.

One sentence: **give the SSM state an identity that, like KV, can be verified — turning misplacement from "silent" into "error, therefore discoverable."**

<div class="keybox">
<strong>In one line:</strong> "Mamba hybrid state" = the bundle of KV cache + Mamba SSM recurrent state that a hybrid model must carry across the PD boundary; its SSM part has no token-id key, so under offload + multi-connector it is the most likely to mis-hit silently — hence fingerprint / version / connector-level state-id are what guard cache correctness.
</div>

## 6. Engineering View: Why This Keeps Getting More Important

- **Hybrid architectures are multiplying**: Jamba, Zamba, Griffon and others interleave Mamba / SSM with Attention and have become a mainstream frontier route.
- **PD disaggregation + offload are cost-cutting defaults**: long-context, large-model inference needs cost cuts, so PD split and KV offload are nearly mandatory, and multi-connector is common.
- **Correctness at their intersection is the "safe to ship" baseline**: a silently misplaced SSM state can quietly degrade an entire service's output without alerting — a hidden incident in production.

> Extension: if you run a hybrid model on vLLM / SGLang with PD disaggregation, check whether the SSM state goes through the same "prefix hash / block check" as KV; many implementations still verify only KV and "bare-copy" the SSM state — exactly the breeding ground for silent hit misplacement.
