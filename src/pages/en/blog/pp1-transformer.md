---
title: Transformer Deep Read — Replacing Recurrence with Attention, Making Sequence Modeling Parallel
description: A close read of Attention Is All You Need (Vaswani et al., 2017, Google) — how self-attention replaces RNNs/CNNs, what multi-head attention actually computes, why positional encoding is mandatory, and why it became the foundation of every LLM since.
pubDate: 2026-08-12
series: Paper Primer Notes
lang: en
altLang: zh
altHref: /blog/pp1-transformer
layout: ../../../layouts/BlogPost.astro
---

## 1. The One-Line Takeaway

**The core claim of the Transformer is this: sequence modeling does not need to read "one token at a time" (RNN), nor "through a sliding window" (CNN). Instead, every position directly "looks up" at every other position via a mechanism called *self-attention*, computing who relates to whom in a single step.** This makes it natively parallel, stackable, and scalable to very long contexts — and every LLM you use today is, at its core, a Transformer.

## 2. Why Replace the RNN?

Before 2017, the workhorses for language, speech, and time series were **RNN / LSTM**:

- They process tokens **left-to-right**, like reading a book: step `t` must wait for step `t-1`.
- Two fatal consequences:
  1. **No parallelism**: a 1000-token sentence means 1000 serial steps — thousands of GPU cores sit idle.
  2. **Long-range forgetting**: information must propagate from token 1 to token 1000, diluting along the way (vanishing/exploding gradients). LSTMs mitigate this with gates, but the essence remains.

CNNs can parallelize, but their **receptive field is bounded by kernel size**; capturing distant dependencies needs many stacked layers.

The Transformer's bet: **model the dependency between any two positions directly, where distance costs only one extra compute step**.

## 3. Self-Attention: Every Word "Looks at the Whole Field"

Self-attention takes a set of vectors `X = [x₁, x₂, …, xₙ]` (each token's embedding). For every position it computes three vectors:

- **Q (Query — what am I looking for)**
- **K (Key — what can I offer)**
- **V (Value — the information I actually carry)**

Obtained by linear projection: `Q = XW_Q`, `K = XW_K`, `V = XW_V`. Then for each position, compute relevance to all others:

```
score(i, j) = Q_i · K_j / √d_k
attention(i) = softmax_j( score(i, j) ) · V_j
```

Intuition: the more `Q_i` resembles `K_j` (larger dot product), the more word `i` "attends" to word `j`; the output is the **weighted sum** of all positions' `V` by attention weights.

> **Role of √d_k**: with large `d_k`, dot products explode. Dividing by its square root stabilizes the softmax gradient — a small but critical trick in the paper.

Key point: **any word looking at another is just one matrix multiplication away**, regardless of how far apart they are. That is "global dependency in one step".

## 4. Multi-Head Attention: Not Just One Angle

A single attention learns only one "relationship view". The Transformer uses **Multi-Head Attention**: split `Q/K/V` into `h` heads, each computing attention in its own subspace, then concatenate and project:

```
head_h = Attention(XW_Q^h, XW_K^h, XW_V^h)
MultiHead = Concat(head₁, …, head_h) · W_O
```

Different heads specialize: some watch syntax, some coreference, some semantics. The paper uses `h = 8` heads, each of dimension `64` (total `512`).

## 5. Positional Encoding: Attention Itself Is Order-Agnostic

Self-attention is **permutation-invariant** — shuffle the sentence and each word's weighted sum depends only on *which* words appear, not *who comes first*. But language has order.

Fix: add a **positional encoding** `PE(pos)` to each position:

- The paper uses **fixed sine/cosine**: `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)`.
- Benefit: it lets the encoding at `pos+k` be expressed linearly from the encoding at `pos`, so the model easily learns *relative* position.

Later models (e.g. GPT) mostly use **learnable positional embeddings** — same idea: order information must be fed in.

## 6. The Encoder–Decoder Skeleton

The original Transformer was built for translation, in two halves:

- **Encoder**: stacks of self-attention + feed-forward, compressing the source into context-rich representations. Its attention is **bidirectional**.
- **Decoder**: beyond self-attention, it adds **Masked Self-Attention** (may only see already-generated tokens, never the future) and **Cross-Attention** (Q from decoder, K/V from encoder), generating the translation token by token.
- Every sublayer uses **residual connection + LayerNorm**: `output = LayerNorm(x + Sublayer(x))` — the key to stable deep training.

## 7. Why It Became the LLM Foundation?

| Property | RNN/LSTM | Transformer |
|----------|----------|-------------|
| Parallelism | Serial, near-zero | Whole-sequence matmul, GPU-friendly |
| Long-range | Long path, forgets | Any two points, one step |
| Scalability | Hard to deepen | Residual+LayerNorm, 100+ layers |
| Hardware fit | Irregular loops | Pure MatMul |

Because of these, **GPT (decoder-only), BERT (encoder-only), ViT (images-as-sequences), whisper, Sora, and even the action head of Diffusion Policy** all sit on this attention mechanism. One 2017 paper defined the "universal compute primitive" of deep learning for the next eight years.

## 8. Summary & Next Steps

- Transformer = self-attention + multi-head + positional encoding + residual/LayerNorm.
- It trades "serial recurrence" for "global one-step attention", turning sequence modeling into parallel, scalable matrix math.
- To go deeper: the GPT family (how decoder-only generates), BERT (bidirectional understanding), and this blog's "vLLM & SGLang Notes" on how KV Cache became the engineering bottleneck of attention.

> Next (pp2): **Diffusion Policy** — it ports the *other* lineage (diffusion denoising, alongside the attention paradigm) into robot action generation, linking neatly with this blog's "VLA Notes".
