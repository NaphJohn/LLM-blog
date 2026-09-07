---
title: 'Frontier Architecture Decoding Notes (6): Gated DeltaNet — Conv1D Captures Locality, the Gated Delta Rule Compresses Globality'
description: 'A full breakdown of the Gated DeltaNet (GDN) layer: operator 1 is a causal Conv1D that provides local context and position awareness, operator 2 is a recurrent state update applying the Gated Delta Rule with a decay gate alpha, compressing global history at O(n*d^2) linear complexity. Cross-checked against the original paper (Yang et al., ICLR 2025) and the Qwen3.5/3.6/3.8 implementations, clarifying four easy mistakes: model naming, the complete formula, the absence of RoPE, and the state being a d_v by d_k matrix.'
pubDate: 2026-08-26
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa6-gated-deltanet
layout: ../../../layouts/BlogPost.astro
---

## 0. Why a Whole Post for This Layer

Posts fa1 to fa4 covered **how to make attention lighter** (sparsity, compression, hybrid SSM). Gated DeltaNet (GDN), the subject here, goes further — it is **not attention** but a **linear attention / recurrent state-space** layer whose core is a fixed-size "state matrix" updated incrementally at every token, compressing global history into an O(d²) fixed-length tensor.

Its layer structure is unusually clean, with only two operators:

- **Operator 1, Causal Conv1D**: after the Q/K/V projections, each of the three branches gets a depthwise separable causal convolution plus SiLU, providing **local context and position awareness**.
- **Operator 2, Recurrent State Update**: takes the convolved Q'/K'/V' plus data-dependent decay and write gates alpha_t and beta_t, applies the **Gated Delta Rule**, and updates the fixed-size `ssm_state` matrix to produce the output.

Together they form a "**local convolution first, global linear attention second**" hierarchy: Conv1D handles short-range dependency and position, while the recurrent state compresses long history into a fixed-length state at O(n·d²). The diagram below is the complete path.

<figure class="arch-fig">
<svg viewBox="0 0 680 540" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gated DeltaNet layer: causal Conv1D followed by recurrent state update with the Gated Delta Rule">
  <defs>
    <marker id="arrowEn6" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#8B4513"/>
    </marker>
    <marker id="arrowBEn6" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#1d4ed8"/>
    </marker>
    <style>
      .box{fill:#fff7ed;stroke:#c2410c;stroke-width:1.5}
      .box2{fill:#eff6ff;stroke:#1d4ed8;stroke-width:1.5}
      .gate{fill:#fef9c3;stroke:#a16207;stroke-width:1.5}
      .ssm{fill:#ecfdf5;stroke:#047857;stroke-width:1.5}
      .lab{font:600 13px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .sub{font:11px -apple-system,'PingFang SC',sans-serif;fill:#475569}
      .form{font:700 14px -apple-system,'PingFang SC',sans-serif;fill:#7c2d12}
      .conn{font:10px -apple-system,'PingFang SC',sans-serif;fill:#6b7280}
      .cell{font:600 12px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
    </style>
  </defs>

  <rect class="ssm" x="270" y="14" width="140" height="36" rx="8"/>
  <text class="lab" x="340" y="37" text-anchor="middle">x_t (current token)</text>
  <line x1="340" y1="50" x2="340" y2="68" stroke="#8B4513" stroke-width="2" marker-end="url(#arrowEn6)"/>

  <rect class="box" x="200" y="68" width="280" height="44" rx="8"/>
  <text class="lab" x="340" y="91" text-anchor="middle">Q / K / V projection (Linear), no RoPE</text>
  <text class="sub" x="340" y="118" text-anchor="middle">Position is left to Conv1D; RoPE is not introduced</text>

  <path d="M300,112 C250,128 200,135 130,150" fill="none" stroke="#8B4513" stroke-width="1.5" marker-end="url(#arrowEn6)"/>
  <line x1="340" y1="112" x2="340" y2="150" stroke="#8B4513" stroke-width="1.5" marker-end="url(#arrowEn6)"/>
  <path d="M380,112 C430,128 480,135 550,150" fill="none" stroke="#8B4513" stroke-width="1.5" marker-end="url(#arrowEn6)"/>

  <rect class="box" x="30" y="150" width="620" height="118" rx="10"/>
  <text class="lab" x="340" y="174" text-anchor="middle">Operator 1 - Causal Conv1D (depthwise separable causal conv + SiLU)</text>
  <text class="sub" x="340" y="193" text-anchor="middle">Maintains a conv_state sliding window, giving local context plus position awareness</text>
  <rect class="gate" x="55" y="206" width="170" height="46" rx="6"/>
  <text class="cell" x="140" y="228" text-anchor="middle">Q1 = Conv1D(Q)</text>
  <text class="sub" x="140" y="244" text-anchor="middle">+ SiLU</text>
  <rect class="gate" x="255" y="206" width="170" height="46" rx="6"/>
  <text class="cell" x="340" y="228" text-anchor="middle">K1 = Conv1D(K)</text>
  <text class="sub" x="340" y="244" text-anchor="middle">+ SiLU</text>
  <rect class="gate" x="455" y="206" width="170" height="46" rx="6"/>
  <text class="cell" x="540" y="228" text-anchor="middle">V1 = Conv1D(V)</text>
  <text class="sub" x="540" y="244" text-anchor="middle">+ SiLU</text>

  <path d="M140,252 C110,272 100,290 90,300" fill="none" stroke="#1d4ed8" stroke-width="1.5" marker-end="url(#arrowBEn6)"/>
  <line x1="340" y1="268" x2="340" y2="300" stroke="#1d4ed8" stroke-width="1.5" marker-end="url(#arrowBEn6)"/>
  <path d="M540,252 C570,272 580,290 590,300" fill="none" stroke="#1d4ed8" stroke-width="1.5" marker-end="url(#arrowBEn6)"/>

  <rect class="box2" x="30" y="300" width="620" height="172" rx="10"/>
  <text class="lab" x="340" y="324" text-anchor="middle">Operator 2 - Recurrent State Update (Gated Delta Rule)</text>

  <rect class="gate" x="55" y="340" width="200" height="44" rx="6"/>
  <text class="cell" x="155" y="362" text-anchor="middle">Gates alpha_t, beta_t (data dependent)</text>
  <text class="sub" x="155" y="378" text-anchor="middle">alpha forgets, beta writes</text>

  <rect class="ssm" x="455" y="340" width="180" height="44" rx="6"/>
  <text class="cell" x="545" y="358" text-anchor="middle">S_t: one per head</text>
  <text class="cell" x="545" y="375" text-anchor="middle">d_v x d_k matrix (fixed size)</text>

  <text class="form" x="340" y="420" text-anchor="middle">S_t = S_{t-1} * alpha_t ( I - beta_t k_t k_t^T ) + beta_t v_t k_t^T</text>
  <text class="sub" x="340" y="440" text-anchor="middle">delta-rule write, minus decay gate alpha_t forgetting history; complexity O(n*d^2)</text>

  <path d="M635,362 C660,400 660,460 545,470 L95,470 C30,470 30,410 95,388" fill="none" stroke="#047857" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#arrowBEn6)"/>
  <text class="conn" x="345" y="490" text-anchor="middle">Recurrence: S_t becomes S_{t-1} at the next step, so the fixed state rolls across tokens</text>

  <line x1="340" y1="472" x2="340" y2="492" stroke="#8B4513" stroke-width="2" marker-end="url(#arrowEn6)"/>
  <rect class="ssm" x="270" y="492" width="140" height="34" rx="8"/>
  <text class="lab" x="340" y="514" text-anchor="middle">o_t (layer output)</text>
</svg>
<figcaption>Figure: the two operators of a Gated DeltaNet layer. Operator 1 applies a depthwise separable causal convolution plus SiLU to each of Q, K and V after projection (maintaining a conv_state sliding window); operator 2 takes the convolved Q1/K1/V1 with data-dependent gates alpha_t and beta_t, applies the Gated Delta Rule to update the fixed-size state matrix S_t, and emits the output. Conv1D handles locality and position; the recurrent state handles global linear attention.</figcaption>
</figure>

## 1. Operator 1: Causal Conv1D (Locality Plus Position)

It sits **after** the Q/K/V projection and does the following **separately** to Q, K and V:

- **Depthwise separable causal Conv1D**: a one-dimensional causal convolution applied independently per channel; the kernel slides only over the current and past positions, preserving causality (no peeking at the future).
- **SiLU activation** after the convolution.
- **conv_state sliding window**: a fixed-length sliding-window state is maintained during training and inference, so recurrent stepping only needs to cache the most recent `kernel_size` positions — it does not grow with the sequence.

It serves two purposes:

1. **Short-range dependency modeling.** Convolution naturally aggregates neighboring tokens, covering the weakness of linear attention, which "only looks at the global state and is insensitive to locality."
2. **Position awareness (replacing RoPE).** The GDN layer **uses no RoPE at all**; position information comes from the local receptive field of this causal convolution — so the judgment that "Causal Conv1D acts like a positional encoding or short-range dependency model" is accurate.

## 2. Operator 2: Recurrent State Update (Gated Delta Rule)

This is the core of GDN. It takes the convolved Q'/K'/V' along with two **data-dependent** gate signals:

- **alpha_t (decay gate, in (0,1))**: controls how strongly the historical state is exponentially forgotten.
- **beta_t (write gate)**: controls how strongly the current token is written into the state.

**Complete formula** (including the decay gate alpha_t):

```text
S_t = S_{t-1} * alpha_t ( I - beta_t k_t k_t^T )  +  beta_t v_t k_t^T
```

Term by term:

- `S_{t-1} * alpha_t`: the previous state is first scaled by the decay gate as a whole, so **history is forgotten in proportion to alpha_t** (this is exactly the term the naive delta rule lacks).
- `alpha_t ( I - beta_t k_t k_t^T )`: on top of the decayed state, a "delta correction along the current key direction" is applied — `beta_t k_t k_t^T` subtracts from the old state the projection of "the old prediction associated with the current key," freeing up capacity.
- `+ beta_t v_t k_t^T`: the current value is written into the state scaled by the write gate beta_t, as the outer product of "key to value."

> Note: the form `S_t = S_{t-1} - beta_t (S_{t-1} phi(k_t)) phi(k_t)^T + beta_t v_t phi(k_t)^T` that appears in some documentation is the **naive delta rule without the alpha_t decay gate**, matching the original paper; but the **complete** GDN formula must include the alpha_t decay term, otherwise it degenerates into linear attention with no forgetting mechanism.

## 3. Positional Encoding: No RoPE

The GDN layer **does not use rotary position embedding (RoPE)**. All absolute and relative position cues in the layer come from:

- the **causal convolution receptive field** of operator 1 (local ordering);
- the recurrent state `S_t` itself rolling forward in time order (implicit temporal structure).

That is precisely why GDN can keep linear complexity without leaning heavily on injected positional encodings.

## 4. State Shape: A Matrix, Not a Vector

`ssm_state` is shaped as **one `d_v x d_k` matrix per attention head** (not a single vector):

- `d_k` is the key dimension (one side of the outer product `k_t k_t^T`);
- `d_v` is the value dimension (the other side of the outer product `v_t k_t^T`);
- the matrix size is **independent of sequence length**, so the state stays constant in size — this is the foundation of O(n·d²) linear complexity.

The "state matrix (or vector)" wording in some documentation is vague; it should be stated plainly as a **matrix**: each head holds a fixed-size `d_v x d_k` table on which the recurrence performs "forget, correct, write."

## 5. Complexity: Why It Is Linear

- Naive attention: every token takes a dot product against the entire history, complexity **O(n²·d)**.
- GDN recurrent state: every token only does "read state, update a fixed-size matrix, write state," with state size fixed at `d_v·d_k`, complexity **O(n·d²)**.

When the sequence is long (n ≫ d), `O(n·d²)` is dramatically lower than `O(n²·d)`. That is the fundamental reason GDN sustains long context without a KV cache that explodes with sequence length — its "memory" is a fixed-size matrix, not a per-token KV list.

## 6. Relationship to Qwen3.5 / 3.6 / 3.8 (Model Name Correction)

The exact model name **`qwen3.8-27B`** appearing in the documentation **is now confirmed to exist**. An earlier judgment in this post that it "does not exist" was wrong and has been corrected against a 2026-08-28 repository snapshot scan (see post 7 for cross-validation of the Qwen3.8 dual checkpoints). The main Qwen3.8 tier is a **MoE model such as 2.4T-A95B**, and a **27B dense** checkpoint also exists.

One thing, however, is certain: **the Gated DeltaNet architecture (the GDN layer) really is a core component of the Qwen3.5 / 3.6 / 3.8 family**, and a key module behind that family's linear inference cost at long context. Therefore:

- Correction: "Qwen3.8-27B does exist" (a 27B dense checkpoint, confirmed by a 2026-08-28 repository scan; the earlier "does not exist" judgment is withdrawn).
- Correct: "GDN is a core architecture component of the Qwen3.5/3.6/3.8 family (including the 2.4T-A95B MoE and the 27B dense variant)."

## 7. Four Corrections After Cross-Checking the Documentation

Cross-checking the original paper (Yang et al., *Gated DeltaNet*, ICLR 2025) against the Qwen3.5/3.6/3.8 implementation code, the overall description is essentially correct, with four points to fix:

| # | Point | Documentation said | Correct statement |
|---|---|---|---|
| 1 | Model name | `qwen3.8-27B` | The 27B dense checkpoint does exist (confirmed by a 2026-08-28 repository scan; the earlier "does not exist" judgment is withdrawn); GDN is a core component of Qwen3.5/3.6/3.8 ✅ |
| 2 | Complete formula | Delta rule with beta only | Must add the decay gate alpha: `S_t = S_{t-1} * alpha_t (I - beta_t k_t k_t^T) + beta_t v_t k_t^T` |
| 3 | Positional encoding | A passing mention that Conv1D is "like a positional encoding" | Accurate — GDN has **no RoPE at all**; position comes from the causal convolution plus state rolling |
| 4 | State shape | "State matrix (or vector)", slightly vague | Precisely **one `d_v x d_k` matrix per head** (fixed size, independent of sequence length) |

<div class="warnbox">
<strong>One line to remember:</strong> Gated DeltaNet = Conv1D for locality and position plus the Gated Delta Rule for global compression; the state is one fixed-size d_v x d_k matrix per head, forgetting with alpha and writing with beta, with no RoPE anywhere and O(n*d^2) complexity. It is one of the pillars of linear inference cost in the Qwen3.5/3.6/3.8 family.
</div>

---

*Next up: a horizontal comparison of GDN with Mamba and the linear attention family (GLA, RWKV, RetNet) — all of them "replace the KV list with a fixed-size state," but their state update rules are completely different.*
