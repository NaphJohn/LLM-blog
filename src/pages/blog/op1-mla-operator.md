---
title: 算子讲解手记（一）：MLA 多头潜在注意力的算子级实现
description: 把"多头潜在注意力（MLA）"从概念拉到算子。给出 c_kv / W_DKV / W_UK / W_UV 的逐行伪代码，量化它如何把每 token 的 KV 缓存从 2·n_h·d_h 压到 d_c（DeepSeek-V2 约 1/64），并说明为什么推理侧只缓存 c_kv 这一条低维潜向量。作为"算子讲解"小系列的开篇，后续将覆盖更多注意力 / 归一化 / 路由算子的可落地实现。
pubDate: 2026-08-07
series: 算子讲解手记
lang: zh
altLang: en
altHref: /en/blog/op1-mla-operator
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么开「算子讲解」这个小系列

前面几篇（mm5 高效系统、fa4 DeepSeek V4）都把 MLA 当作"概念"讲：把 K/V 联合压成一个低维潜向量 c，KV 缓存因此大幅缩小。但"压成潜向量"到底怎么算、缓存里到底存的是什么、比原来小多少——这些只有落到**算子级**才说得清。

这个小系列就干这件事：把前沿架构里常被一笔带过的算子，写成能直接照着实现的逐行伪代码。开篇选 MLA，因为它是 2026 年"压 KV"这条主线（mm5 / fa4）的共同地基，也是理解 DeepSeek 系模型推理降本的第一块砖。

## 1. MLA 在做什么（一句话回顾）

标准多头注意力（MHA）每个 token 都要把完整的 K、V 存进 KV 缓存：

- K 维度 = n_h · d_h（头数 × 每头维度）
- V 维度 = n_h · d_h
- 每 token 缓存 = 2 · n_h · d_h

当 n_h=128、d_h=128 时，每个 token 要存 32768 个数。1M 上下文就是 327.68 亿个数——这正是"长上下文成本平方爆炸"的来源。

MLA（Multi-head Latent Attention）的做法：不直接存 K/V，而是把 K/V **联合投影到一个低维潜空间**，只缓存这个潜向量 c_kv。推理时再用两个上投影把 c_kv 还原成 K、V。

## 2. 算子逐行伪代码

维度约定（以 DeepSeek-V2 风格为例）：

- d_model：模型隐藏维
- n_h：注意力头数，d_h：每头维度
- d_c：潜向量维度（KV 压缩后的维度，远小于 n_h·d_h）

```python
# 下沉投影（KV 压缩）—— 唯一进入 KV 缓存的矩阵
W_DKV : [d_model, d_c]         # 把 x 压成低维潜向量
# 上投影 —— 用潜向量还原 K、V
W_UK  : [d_c, n_h * d_h]       # c_kv -> K
W_UV  : [d_c, n_h * d_h]       # c_kv -> V
# Query 侧（最简版；真实 MLA 中 Q 也先压成 c_q 再升，见 §4）
W_Q   : [d_model, n_h * d_h]
# 输出投影
W_O   : [n_h * d_h, d_model]

# ---- 每个 token 的 Prefill ----
c_kv = x @ W_DKV              # [B, T, d_c]   <- 唯一缓存的 KV 状态

# ---- 用 c_kv 还原 K、V（不进缓存，现算现用）----
K = c_kv @ W_UK               # [B, T, n_h * d_h]
V = c_kv @ W_UV               # [B, T, n_h * d_h]

# ---- 标准缩放点积注意力 ----
Q     = x @ W_Q               # [B, T, n_h * d_h]
scores = Q @ K^T / sqrt(d_h)  # [B, n_h, T, T]
attn   = softmax(scores, axis=-1)
O     = attn @ V              # [B, T, n_h * d_h]
out   = O @ W_O               # [B, T, d_model]
```

关键点：**整条链路里，进 KV 缓存的只有 `c_kv`（d_c 维）**。`K`、`V` 是在注意力计算前"临时还原"出来的，算完即弃，不占缓存。

## 3. 缓存缩小的量级

| 方案 | 每 token 缓存维度 | 1M 上下文（n_h=128, d_h=128） |
|---|---|---|
| 标准 MHA | 2 · n_h · d_h = 32768 | 约 327.7 亿 |
| MLA（d_c=512） | d_c = 512 | 约 5.1 亿 |

缓存降到约 **1/64**（~98.4% 下降）。这正是 mm5 说的"压宽度"：KV 的**宽度**被压成 d_c，而注意力质量靠上投影 W_UK / W_UV 在还原时补回。代价是多了一次上投影的算力（但远比省下的 KV 访存与显存划算）。

<div class="keybox">
<strong>要点：</strong>MLA 的省，省在<strong>显存与访存</strong>（KV 缓存），不是省算力。它用"低维潜向量 + 两次小矩阵乘"换"不再存完整 K/V"，把长上下文的成本从 O(n·d_kv) 压到 O(n·d_c)。
</div>

## 4. 细节：Query 也压，但推理侧只缓存 c_kv

真实 MLA 里 Query 同样被压缩：先 `c_q = x @ W_DQ`（d_c 维），再 `Q = c_q @ W_UQ`。训练时压 Q 是为了砍掉**激活内存**（前向/反向都要留 Q 的显存）；到了推理，Query 是从当前输入 x 现算的，根本不进 KV 缓存，所以缓存里唯一要留的，就是 c_kv 这一条。

这也是为什么 MLA 在"训练省显存、推理省 KV"两边通吃——它压的是同一个潜向量 c，只是用途不同。

## 5. 和本系列其他篇的衔接

- mm5 把 MLA 放进"注意力五件套"里做概念定位，并画出 DeepSeek-V4 的"MLA → NSA → DSA → CSA+HCA"四级演进图——MLA 是起点。
- fa4（DeepSeek V4）的 CSA/HCA 是在 MLA 这条"压 KV"主线上再往前走：把 KV 压到 4-token 块、甚至 128× 重度压缩。

把 MLA 的算子吃透，再看 V4 的压缩注意力，就会变成"同一个潜向量思路的连续升级"，而不是一堆陌生名词。

> 下一篇准备写 **GQA / MQA 的算子对比**（它们压的是"头"而非"宽度"），以及 **MHA 到 MLA 的参数量守恒**——关注本系列即可收到更新。
