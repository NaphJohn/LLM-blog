---
title: 推理系统基础设施（二）：Sequence Parallelism + Ring Attention——把 1M+ Token 训成常态
description: Ring Attention 让多张 GPU 用「环形传递 K/V 块 + 在线 softmax 累加」算出一个数学上完全等价于全量 attention 的结果，而每张卡的激活内存与序列长度无关。本文讲清核心做法、与朴素/Megatron-SP 的对比、Zig-Zag 切块、以及与 HCA/CSA 协同，并附环通信结构图。
pubDate: 2026-08-26
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys2-ring-attention
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话定位

> **Sequence Parallelism 把长序列在 token 维度切到多卡；Ring Attention 让多卡用「环形传递 K/V 块 + 在线 softmax 累加」算出一个数学上完全等价于全量 attention 的结果，每张卡的激活内存与序列长度无关。**

这是突破单卡序列长度天花板的关键手段，让 1M+ Token 的训练/推理成为常规操作。

## 1. 视野速览

- **常规手段（降本增效基本功）**：INT8/FP8/INT4 量化、结构化/非结构化剪枝、知识蒸馏、KV Cache 管理（PagedAttention）、continuous batching、混合精度训练（FP16/BF16/FP8）、梯度检查点、并行策略（数据/模型/流水线）、LoRA/QLoRA。
- **前沿手段（2025–2026）**：稀疏 MoE 与专家并行、MLA 与 KV 压缩、线性注意力、Speculative Decoding、PD 分离、BitNet 1-bit LLM、FP4 QAT、**Sequence Parallelism + Ring Attention（本期）**。

## 2. 核心做法：在线 softmax + 环形 K/V 通信

假设 8 卡环，序列按 token 切成 8 段，每段长度 `L/N`。每张卡持自己的 query `Q_i`（不动），让 K/V 块沿环流动：

```text
m_i, l_i, o_i = -inf, 0, 0     # 运行时 max / 累加和 / 输出

for step in range(N):           # 沿环走 N 步，每步拿到一个 K/V 块
    K_block, V_block = recv_from_prev()   # 来自上一张卡
    send_to_next(my_KV)                    # 同时把自己的块传给下一张卡

    s_ij = Q_i @ K_block^T / sqrt(d)               # (L/N, B)
    m_new = max(m_i, s_ij.max(-1))                 # 更新 running max
    p_ij  = exp(s_ij - m_new[:, None])
    l_new = exp(m_i - m_new) * l_i + p_ij.sum(-1)
    o_i   = exp(m_i - m_new)[:, None] * o_i + p_ij @ V_block
    m_i, l_i = m_new, l_new

O_i = o_i / l_i[:, None]    # 归一化，结果与全量 attention 完全一致
```

符号：`Q_i` = 第 i 卡本地 query；`K_block, V_block` = 沿环流动的 K/V 块；`m_i, l_i, o_i` = 在线 softmax 的运行 max / 累加和 / 部分输出。

## 3. 环通信结构图

<div class="fig">
<svg viewBox="0 0 680 410" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Ring Attention 环通信：多 GPU 环、K/V 块流动、在线 softmax 累加 m/l/o">
  <rect x="0" y="0" width="680" height="410" fill="none"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Ring Attention：环形传递 K/V 块 + 在线 softmax 累加</text>

  <!-- GPU cards -->
  <!-- GPU0 -->
  <rect x="44" y="70" width="128" height="104" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="108" y="92" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">GPU 0</text>
  <text x="108" y="114" font-size="11" fill="#374151" text-anchor="middle">Q₀（本地，不动）</text>
  <rect x="64" y="126" width="88" height="34" rx="5" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="108" y="147" font-size="10.5" fill="#b45309" text-anchor="middle">K/V 块ⱼ</text>

  <!-- GPU1 -->
  <rect x="192" y="70" width="128" height="104" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="256" y="92" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">GPU 1</text>
  <text x="256" y="114" font-size="11" fill="#374151" text-anchor="middle">Q₁（本地，不动）</text>
  <rect x="212" y="126" width="88" height="34" rx="5" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="256" y="147" font-size="10.5" fill="#b45309" text-anchor="middle">K/V 块ⱼ</text>

  <!-- GPU2 -->
  <rect x="340" y="70" width="128" height="104" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="404" y="92" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">GPU 2</text>
  <text x="404" y="114" font-size="11" fill="#374151" text-anchor="middle">Q₂（本地，不动）</text>
  <rect x="360" y="126" width="88" height="34" rx="5" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="404" y="147" font-size="10.5" fill="#b45309" text-anchor="middle">K/V 块ⱼ</text>

  <!-- GPU3 -->
  <rect x="488" y="70" width="128" height="104" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="552" y="92" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">GPU 3</text>
  <text x="552" y="114" font-size="11" fill="#374151" text-anchor="middle">Q₃（本地，不动）</text>
  <rect x="508" y="126" width="88" height="34" rx="5" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="552" y="147" font-size="10.5" fill="#b45309" text-anchor="middle">K/V 块ⱼ</text>

  <!-- ring arrows (K/V block flowing) -->
  <line x1="172" y1="143" x2="190" y2="143" stroke="#f59e0b" stroke-width="2" marker-end="url(#rg)"/>
  <line x1="320" y1="143" x2="338" y2="143" stroke="#f59e0b" stroke-width="2" marker-end="url(#rg)"/>
  <line x1="468" y1="143" x2="486" y2="143" stroke="#f59e0b" stroke-width="2" marker-end="url(#rg)"/>
  <path d="M552,174 C 640,210 640,250 552,250 L 128,250 C 40,250 40,210 128,174" fill="none" stroke="#f59e0b" stroke-width="2" marker-end="url(#rg)"/>
  <text x="330" y="200" font-size="11" fill="#b45309" text-anchor="middle">K/V 块沿环流动（每步传一个块）</text>
  <text x="330" y="216" font-size="10.5" fill="#b45309" text-anchor="middle">GPU3 → GPU0 闭合环</text>

  <!-- online softmax state -->
  <rect x="44" y="280" width="592" height="64" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="60" y="302" font-size="11.5" font-weight="700" fill="#1a1a1a">在线 softmax 累加（每步）：</text>
  <text x="60" y="322" font-size="11" fill="#374151">s_ij = Q_i·K_blockᵀ / √d ； m_new = max(m, s.max) ； p = exp(s − m_new)</text>
  <text x="60" y="339" font-size="11" fill="#374151">l_new = exp(m − m_new)·l + p.sum ； o_new = exp(m − m_new)·o + p·V_block → 走完 N 步 O_i = o_i / l_i</text>

  <!-- bottom note -->
  <rect x="44" y="360" width="592" height="36" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="60" y="383" font-size="11" fill="#047857">每张卡只缓存「自己的 Q + 1 个流入 K/V 块」→ 激活内存与总序列长度无关；8 卡可跑 1M token，理论扩到 100 卡 = 100M token。</text>

  <defs>
    <marker id="rg" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b"/></marker>
  </defs>
</svg>
  <p class="cap">图：Ring Attention 的环通信。序列按 token 切到 8 卡，K/V 块沿环逐步传递，每卡用在线 softmax 把流入的 K/V 块累加起来；走完 N 步，结果与全量 attention 数学等价。</p>
</div>

## 4. 与朴素方案对比

| 方案 | 激活内存 | 能否跑 1M token | 备注 |
|---|---|---|---|
| 朴素（全量 Q×Kᵀ） | O(L²) | ❌ 单卡必爆 | 1M token 单卡显存直接炸 |
| Megatron-LM SP（all-gather） | 临时全量 KV | ⚠️ 几万 token | attention 内每卡仍要临时全量 KV |
| **Ring Attention（本方案）** | **与 L 无关** | ✅ 8 卡 1M，100 卡 100M | 通信被 matmul 掩盖，墙钟几乎不增 |

通信开销被 matmul 掩盖（每 hop 通信正好盖住上一次 matmul），所以墙钟延迟几乎不增加。

## 5. 实测收益与代表工作

- **Liu et al., UC Berkeley, ICLR 2024**：在 64 卡上把序列长度推到 **100M token**，数学上与单卡 full attention 完全一致。
- **DeepSpeed Ulysses**（Microsoft, 2023）：用 all-to-all 替代环形通信，在高速互联集群上 2.5× faster。
- **OpenRLHF / ring-flash-attention**：生产级实现，`--ds.ring_attn_size 8` 即可在 8 卡上把 1M context 训练跑起来。
- **DeepSeek V4 / Qwen3.8-Max** 1M context：都依赖 Ring Attention 或等价的长上下文分布式方案。
- **Zig-Zag Ring Attention**：把 GPU 顺序切块换成交错分配 `[0,4,8…]`、`[1,5,9…]`，解决因果 mask 下「后卡等前卡」的负载不均，GPU 利用率近 100%。

## 6. 具身智能关联

1. **视频世界模型**（V-JEPA 2 / Genie）需要 1M+ token 的视频片段输入，Ring Attention 是 V-JEPA 2 多机训练时的标配；
2. **机器人 VLA** 长操作日志 + 视频历史规划场景，SP+Ring 让「全操作日志一起算」成为可能；
3. **与 HCA/CSA 协同**——HCA 把 KV 再压一档，让 Ring Attention 在环里传的是压缩后的「目录块」，1M token → 8000 目录块的环通信，进一步把带宽压力降到 1/128（详见本系列 sys1 的 NUMA 链路与 fa4 的 HCA）。

## 7. 学习建议 / 常见坑

1. **必须用在线 softmax + running max**，否则累计误差会让结果偏；
2. **因果 mask 下用 Zig-Zag 切块**，否则后卡空转；
3. **通信带宽是真正瓶颈**——NVLink / IB 不可省（普通以太网一旦通信时间 > 计算时间，「边算边传」就退化成「等数据」）；
4. **与张量并行混用时，TP×SP 通信方向要正交**，否则 all-gather 和环通信会撞车。
