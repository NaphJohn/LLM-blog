---
title: "Attention & Positional Encoding: FlashAttention / RoPE / GQA / MLA / RMSNorm"
description: The "standard five-piece set" of mainstream Decoder LLMs—FlashAttention (IO-aware attention kernel), RoPE (rotary position embedding), GQA (grouped-query attention), MLA (multi-head latent attention), RMSNorm (simplified layer norm). One post on each one's role, adopters, and key pitfalls.
pubDate: 2026-07-31
series: AI Knowledge Base
lang: en
altLang: zh
altHref: /blog/kb-attention-posenc
layout: ../../../layouts/BlogPost.astro
---

## 1. FlashAttention (FA-1 → FA-4): IO-aware attention kernel

- **Mechanism**: tiling + online softmax + kernel fusion; only Q/K/V/O live in HBM, the **N×N attention matrix stays in SRAM**, never touching HBM.
- **bit-exact**: numerically identical to the naive implementation (FP32 main path), not an approximation.
- **Gains**: 2~4× on A100, 128K context without OOM; H100 FA-3 FP8 hits 740 TFLOPs/s.
- **Ecosystem**: default in Qwen3/3.5, DeepSeek V3/V4, MiniMax, GLM/Kimi/Llama 5.

## 2. RoPE (Rotary Position Embedding, Su et al. 2021)

- **Mechanism**: rotate Q/K within 2D subspaces to encode **relative position**, so the attention score depends only on offset `m−n`: `⟨Q_m', K_n'⟩ = f(q, k, m−n)`.
- **Key points**: `θ_i = 10000^(−2i/d)`; zero-parameter explicit relative position + NTK/YaRN extension to very long context; **V is not rotated** (common pitfall).
- **Adopters**: Qwen3/3.5 (NTK-aware to 128K/1M), DeepSeek V4 (decoupled RoPE + MLA), MiniMax (Lightning+RoPE hybrid), Llama 5 / GLM / Kimi.

## 3. GQA (Grouped-Query Attention)

- **Mechanism**: split H Query Heads into G groups that **share KV Heads within a group**, compressing KV Cache 4~8×.
- **Key points**: G=8 is the industry sweet spot (< 0.5% quality loss); in production, `repeat_interleave` shares implicitly inside the FA kernel, zero extra memory traffic.
- **Adopters**: Llama 2/3 (H=64, G=8, 8:1), all Qwen, Mistral/Mixtral, DeepSeek V2-V4 (GQA + MLA double compression), MiniMax.

## 4. MLA (Multi-head Latent Attention)

- **Mechanism**: jointly compress K/V into a **low-rank latent vector c**, cache only c, up-project on demand → memory drops to 1/4~1/10 with almost no quality loss.
- **Key point**: **requires decoupled RoPE** (MLA compresses KV; position info travels a separate path).
- **Adopters**: DeepSeek V2/V3/V4 (V4 further evolves with CSA/HCA hybrid attention).

## 5. RMSNorm (Root Mean Square LayerNorm)

- **Mechanism**: simplified LayerNorm—drops mean-centering and the β bias, keeps only RMS rescaling.
- **Gains**: ~7~15% speedup, comparable quality; directly valuable for latency in on-device VLA deployment.
- **Adopters**: Llama, Qwen3/3.5, DeepSeek V4, MiniMax, Mistral, Gemma.

## 6. Practical takeaways

- The key to long-context / low-cost inference is KV Cache: GQA/MLA "save memory", FlashAttention "saves IO", RMSNorm "saves compute"—only together do they form today's efficiency baseline.
- When comparing model efficiency, check whether MLA/GQA + FA + quantized KV are all used; this directly decides concurrency and cost on the same hardware.
