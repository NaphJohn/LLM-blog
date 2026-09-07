---
title: 'Inference Systems Infrastructure Notes (2): Sequence Parallelism + Ring Attention — Making 1M+ Token Training Routine'
description: 'Ring Attention lets multiple GPUs compute a result that is mathematically identical to full attention by passing K/V blocks around a ring and accumulating with online softmax, while per-GPU activation memory stays independent of sequence length. Covers the core method, comparison with naive and Megatron-SP, Zig-Zag blocking, synergy with HCA/CSA, plus a ring-communication diagram.'
pubDate: 2026-08-26
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys2-ring-attention
layout: ../../../layouts/BlogPost.astro
---

## 0. One-Line Positioning

> **Sequence Parallelism shards a long sequence across GPUs along the token dimension; Ring Attention lets those GPUs compute a result mathematically identical to full attention by passing K/V blocks around a ring and accumulating with online softmax — with per-GPU activation memory independent of sequence length.**

This is the key technique for breaking the single-GPU sequence-length ceiling, and it turns 1M+ token training and inference into routine work.

## 1. Landscape at a Glance

- **Standard techniques (the cost-efficiency basics)**: INT8 / FP8 / INT4 quantization, structured and unstructured pruning, knowledge distillation, KV cache management (PagedAttention), continuous batching, mixed-precision training (FP16 / BF16 / FP8), gradient checkpointing, parallelism strategies (data / model / pipeline), LoRA / QLoRA.
- **Frontier techniques (2025-2026)**: sparse MoE with expert parallelism, MLA and KV compression, linear attention, speculative decoding, PD disaggregation, BitNet 1-bit LLM, FP4 QAT, and **Sequence Parallelism + Ring Attention (this post)**.

## 2. Core Method: Online Softmax + Ring K/V Communication

Assume an 8-GPU ring, with the sequence split into 8 token segments of length `L/N`. Each GPU holds its own query `Q_i` (which never moves) while K/V blocks flow around the ring:

```text
m_i, l_i, o_i = -inf, 0, 0     # running max / running sum / output

for step in range(N):           # N steps around the ring, one K/V block per step
    K_block, V_block = recv_from_prev()   # from the previous GPU
    send_to_next(my_KV)                   # simultaneously send our own block onward

    s_ij = Q_i @ K_block^T / sqrt(d)               # (L/N, B)
    m_new = max(m_i, s_ij.max(-1))                 # update running max
    p_ij  = exp(s_ij - m_new[:, None])
    l_new = exp(m_i - m_new) * l_i + p_ij.sum(-1)
    o_i   = exp(m_i - m_new)[:, None] * o_i + p_ij @ V_block
    m_i, l_i = m_new, l_new

O_i = o_i / l_i[:, None]    # normalize; result identical to full attention
```

Notation: `Q_i` is the local query on GPU i; `K_block, V_block` are the K/V blocks traveling around the ring; `m_i, l_i, o_i` are the online softmax running max, running sum, and partial output.

## 3. Ring Communication Structure

<div class="fig">
<svg viewBox="0 0 680 410" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Ring Attention ring communication: multi-GPU ring, K/V blocks flowing, online softmax accumulating m, l and o">
  <rect x="0" y="0" width="680" height="410" fill="none"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Ring Attention: ring-passing K/V blocks plus online softmax accumulation</text>

  <rect x="44" y="70" width="128" height="104" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="108" y="92" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">GPU 0</text>
  <text x="108" y="114" font-size="11" fill="#374151" text-anchor="middle">Q0 (local, fixed)</text>
  <rect x="64" y="126" width="88" height="34" rx="5" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="108" y="147" font-size="10.5" fill="#b45309" text-anchor="middle">K/V block j</text>

  <rect x="192" y="70" width="128" height="104" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="256" y="92" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">GPU 1</text>
  <text x="256" y="114" font-size="11" fill="#374151" text-anchor="middle">Q1 (local, fixed)</text>
  <rect x="212" y="126" width="88" height="34" rx="5" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="256" y="147" font-size="10.5" fill="#b45309" text-anchor="middle">K/V block j</text>

  <rect x="340" y="70" width="128" height="104" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="404" y="92" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">GPU 2</text>
  <text x="404" y="114" font-size="11" fill="#374151" text-anchor="middle">Q2 (local, fixed)</text>
  <rect x="360" y="126" width="88" height="34" rx="5" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="404" y="147" font-size="10.5" fill="#b45309" text-anchor="middle">K/V block j</text>

  <rect x="488" y="70" width="128" height="104" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="552" y="92" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">GPU 3</text>
  <text x="552" y="114" font-size="11" fill="#374151" text-anchor="middle">Q3 (local, fixed)</text>
  <rect x="508" y="126" width="88" height="34" rx="5" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="552" y="147" font-size="10.5" fill="#b45309" text-anchor="middle">K/V block j</text>

  <line x1="172" y1="143" x2="190" y2="143" stroke="#f59e0b" stroke-width="2" marker-end="url(#rgEn)"/>
  <line x1="320" y1="143" x2="338" y2="143" stroke="#f59e0b" stroke-width="2" marker-end="url(#rgEn)"/>
  <line x1="468" y1="143" x2="486" y2="143" stroke="#f59e0b" stroke-width="2" marker-end="url(#rgEn)"/>
  <path d="M552,174 C 640,210 640,250 552,250 L 128,250 C 40,250 40,210 128,174" fill="none" stroke="#f59e0b" stroke-width="2" marker-end="url(#rgEn)"/>
  <text x="330" y="200" font-size="11" fill="#b45309" text-anchor="middle">K/V blocks flow around the ring (one block per step)</text>
  <text x="330" y="216" font-size="10.5" fill="#b45309" text-anchor="middle">GPU3 closes the ring back to GPU0</text>

  <rect x="44" y="280" width="592" height="64" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="60" y="302" font-size="11.5" font-weight="700" fill="#1a1a1a">Online softmax accumulation (each step):</text>
  <text x="60" y="322" font-size="11" fill="#374151">s_ij = Q_i * K_block^T / sqrt(d);  m_new = max(m, s.max);  p = exp(s - m_new)</text>
  <text x="60" y="339" font-size="11" fill="#374151">l_new = exp(m - m_new) * l + p.sum;  o_new = exp(m - m_new) * o + p * V_block;  after N steps O_i = o_i / l_i</text>

  <rect x="44" y="360" width="592" height="36" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="60" y="383" font-size="11" fill="#047857">Each GPU caches only its own Q plus one incoming K/V block, so activation memory is independent of total sequence length: 8 GPUs run 1M tokens, and in theory 100 GPUs reach 100M.</text>

  <defs>
    <marker id="rgEn" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b"/></marker>
  </defs>
</svg>
<p class="cap">Figure: Ring Attention communication. The sequence is sharded across 8 GPUs by token; K/V blocks travel one step at a time around the ring, and each GPU accumulates the incoming blocks with online softmax. After N steps the result is mathematically equivalent to full attention.</p>
</div>

## 4. Comparison with Naive Approaches

| Approach | Activation memory | Can it run 1M tokens? | Notes |
|---|---|---|---|
| Naive (full Q x K^T) | O(L²) | No, a single GPU always blows up | 1M tokens overflows one card outright |
| Megatron-LM SP (all-gather) | Temporary full KV | Marginal, tens of thousands of tokens | Each GPU still needs the full KV temporarily inside attention |
| **Ring Attention (this approach)** | **Independent of L** | Yes, 1M on 8 GPUs, 100M on 100 | Communication hides under matmul, wall clock barely grows |

Communication overhead is hidden by the matmul (each hop's transfer covers the previous matmul), so wall-clock latency barely increases.

## 5. Measured Results and Representative Work

- **Liu et al., UC Berkeley, ICLR 2024**: pushed sequence length to **100M tokens** on 64 GPUs, mathematically identical to single-GPU full attention.
- **DeepSpeed Ulysses** (Microsoft, 2023): replaces ring communication with all-to-all, 2.5x faster on high-speed interconnect clusters.
- **OpenRLHF / ring-flash-attention**: production-grade implementation; `--ds.ring_attn_size 8` is enough to train a 1M context on 8 GPUs.
- **DeepSeek V4 / Qwen3.8-Max** at 1M context: both rely on Ring Attention or an equivalent distributed long-context scheme.
- **Zig-Zag Ring Attention**: replaces in-order blocking with interleaved assignment `[0,4,8...]`, `[1,5,9...]`, fixing the load imbalance under a causal mask where later GPUs wait on earlier ones, pushing GPU utilization close to 100%.

## 6. Connection to Embodied Intelligence

1. **Video world models** (V-JEPA 2 / Genie) need 1M+ token video clips as input; Ring Attention is standard in multi-node V-JEPA 2 training.
2. **Robot VLA** scenarios with long operation logs plus video history planning: SP + Ring makes "compute the entire operation log at once" feasible.
3. **Synergy with HCA / CSA** — HCA compresses KV one more notch, so what travels around the ring is a compressed "directory block": 1M tokens become 8000 directory blocks, cutting bandwidth pressure by another 128x (see sys1 in this series on the NUMA path, and fa4 on HCA).

## 7. Study Tips and Common Pitfalls

1. **You must use online softmax with a running max**, or accumulated error skews the result;
2. **Use Zig-Zag blocking under a causal mask**, or later GPUs idle;
3. **Communication bandwidth is the real bottleneck** — NVLink or InfiniBand is not optional (on plain Ethernet, once transfer time exceeds compute time, "compute while transferring" degrades into "waiting for data");
4. **When mixing with tensor parallelism, keep the TP and SP communication directions orthogonal**, or all-gather and ring traffic collide.
