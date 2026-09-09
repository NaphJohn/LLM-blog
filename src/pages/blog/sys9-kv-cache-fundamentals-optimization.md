---
title: '推理系统基础设施手记（九）：KV Cache 全景——它为什么同时决定你的吞吐、延迟和能开多少并发'
description: '从"冗余计算"讲起：为什么自回归生成可以缓存、缓存的边界在哪；KV Cache 到底多大（Qwen3-32B 实测 32GiB）；它如何吃掉 TTFT/TPOT/并发；以及 token 级 / 模型级 / 系统级三层优化分类与"减少序列长度/头数/key_bits/头维度/layers"五个旋钮。'
pubDate: 2026-09-09
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys9-kv-cache-fundamentals-optimization
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

KV Cache 是**用空间换时间**的产物：它把自回归生成里的重复计算一次性做完并缓存下来，代价是显存随 batch × 序列长度线性膨胀，最终反过来卡住并发。本篇讲清三件事——**它为什么能缓存、它到底有多大、以及业界到底从哪几个方向在削它**。

## 1. 先看清敌人：没有 KV Cache 时，我们在重复算什么

LLM 推理是自回归的：生成第 n 个 token 时，需要用到前 n−1 个 token 的信息。朴素实现下，每生成一个 token 就要把**全部历史 token 重新走一遍前向**：

| 模块 | 冗余在哪 |
|------|----------|
| Embedding | 历史 token 的 embedding 反复生成 |
| K / V 生成 | 历史 token 的 K、V 反复计算 |
| QKᵀ | 注意力分数矩阵反复计算，复杂度随序列平方增长 |
| Softmax × V | 加权求和反复计算 |

结果是生成 N 个 token 的总复杂度约为 **O(N²)**，序列越长越慢，而且慢得非常难看。

<div class="fig">
<svg viewBox="0 0 640 250" role="img" aria-label="prefill 与 decode 阶段 KV Cache 的变化">
  <title>图 1：prefill 与 decode 阶段的 KV Cache 变化</title>
  <text x="16" y="26" font-size="13" fill="#6b7280">prefill：整段 prompt 一次算完，生成第一个 token</text>
  <rect x="16" y="44" width="46" height="38" rx="4" fill="#f6f8fa" stroke="#30363d"/>
  <text x="33" y="69" font-size="15" fill="#1a1a1a" text-anchor="middle">新</text>
  <rect x="68" y="44" width="46" height="38" rx="4" fill="#f6f8fa" stroke="#30363d"/>
  <text x="85" y="69" font-size="15" fill="#1a1a1a" text-anchor="middle">年</text>
  <rect x="120" y="44" width="46" height="38" rx="4" fill="#f6f8fa" stroke="#30363d"/>
  <text x="137" y="69" font-size="15" fill="#1a1a1a" text-anchor="middle">快</text>
  <line x1="172" y1="63" x2="206" y2="63" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="206,63 198,59 198,67" fill="#2563eb"/>
  <rect x="214" y="38" width="196" height="50" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="312" y="30" font-size="12" fill="#2563eb" text-anchor="middle">KV Cache · 3 slots</text>
  <rect x="224" y="50" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <rect x="282" y="50" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <rect x="340" y="50" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <line x1="416" y1="63" x2="450" y2="63" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="450,63 442,59 442,67" fill="#2563eb"/>
  <rect x="458" y="44" width="46" height="38" rx="4" fill="#ecfdf5" stroke="#10b981"/>
  <text x="481" y="69" font-size="15" fill="#047857" text-anchor="middle">乐</text>
  <text x="16" y="146" font-size="13" fill="#6b7280">decode：每步只算 1 个 token，但注意力要读全部历史 KV</text>
  <rect x="16" y="164" width="46" height="38" rx="4" fill="#f6f8fa" stroke="#30363d"/>
  <text x="39" y="189" font-size="15" fill="#1a1a1a" text-anchor="middle">乐</text>
  <line x1="68" y1="183" x2="206" y2="183" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="206,183 198,179 198,187" fill="#2563eb"/>
  <rect x="214" y="158" width="262" height="50" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="345" y="150" font-size="12" fill="#2563eb" text-anchor="middle">KV Cache · 4 slots（+1）</text>
  <rect x="224" y="170" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <rect x="282" y="170" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <rect x="340" y="170" width="50" height="26" rx="3" fill="#ffffff" stroke="#85b7eb"/>
  <rect x="398" y="170" width="50" height="26" rx="3" fill="#dcfce7" stroke="#10b981"/>
  <line x1="482" y1="183" x2="516" y2="183" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="516,183 508,179 508,187" fill="#2563eb"/>
  <rect x="524" y="164" width="46" height="38" rx="4" fill="#ecfdf5" stroke="#10b981"/>
  <text x="547" y="189" font-size="15" fill="#047857" text-anchor="middle">万</text>
  <text x="16" y="234" font-size="12" fill="#6b7280">绿色 = 本步新增。decode 每步计算量恒定（1 个 token），但要读的 KV 线性增长——这就是"访存密集"的来源。</text>
</svg>
<figcaption>图 1：prefill 一次性为整段 prompt 建立 KV Cache；decode 每步只算 1 个 token，缓存 +1，但注意力要读全部历史。</figcaption>
</div>

## 2. 为什么可以缓存：逐模块拆一遍

缓存能成立，靠的是两个性质：**因果掩码**和**位置无关性**。

### Attention：因果掩码让"过去"不再变化

- 推理时第 i 个位置的 query 只会关注 1..i，**永远看不到后面**。所以在单向 attention 下，`i=3` 时算出的 k₂ 和 `i=4` 时算出的 k₂ **完全相同**。
- 换句话说：历史 token 的 K/V 是"已经定死的值"，没有重算的理由。每步只需算**最新那个 token**的 K/V，然后拼到缓存后面。

### FFN / LayerNorm / Linear：位置之间不交互

- **FFN**：序列各位置的特征不交互，第 i 个输出只由第 i 个输入决定（Y₁ 只和 X₁ 有关）。
- **LayerNorm**：沿 d_model 方向算均值方差，输出也只和当前行的 hidden_state 有关。
- **Linear（lm_head）**：矩阵乘的性质决定了 logits 的最后一行只由 hidden_state 的最后一行决定。
- **Softmax**：只要保留之前的计算结果，就能和新结果合并。

### 一句话的数学支撑

矩阵乘法可以分块：A 拆成 `[:s]` 和 `[s:]` 分别与 B 相乘，结果直接拼接，与整体相乘一致。注意力和 FFN 都是矩阵乘，所以缓存 `[:s]` 部分的结果、只算新增的 `[s:]`，结果完全等价——**这是一次无损优化**。

## 3. prefill 与 decode：两个阶段的性格完全不同

| 阶段 | 做什么 | 计算特征 | 关键指标 |
|------|--------|----------|----------|
| **prefill** | 一次性处理整个 prompt，建立 KV Cache，产出第 1 个 token | **算力密集**（大矩阵乘，并行度高） | TTFT |
| **decode** | 逐个生成后续 token，每步读全部历史 KV | **访存密集**（计算量小，但要搬 KV） | TPOT |

这个"一个要算力、一个要带宽"的分裂，正是后面 PD 分离（sys8）存在的全部理由。

## 4. KV Cache 到底多大

公式（标准 MHA/GQA）：

```
KV Cache = 2 (K & V) × layers × batch × seq_len × kv_dim × dtype_bytes
```

其中 `kv_dim = num_kv_heads × head_size`。

**实测案例（Qwen3-32B）**：

- 模型 64 层，GQA 实际 KV 维度 1024，BF16
- batch=4、seq=32768 时：
  `2 × 64 × 4 × 32768 × 1024 × 2B = 32 GiB`
- batch 提到 8 → **64 GiB**，已经接近模型权重本身（BF16 约 64 GB）

**结论**：KV Cache 随 batch 和 seq_len **线性增长**，在高并发 + 长上下文场景下，它才是限制服务并发能力的那个瓶颈，而不是模型权重。

## 5. 它如何吃掉性能：三个指标与三段显存

| 指标 | 含义 | 被什么影响 |
|------|------|------------|
| **TTFT** | 首 token 延迟 | 主要是 prefill 耗时；命中前缀缓存可大幅下降 |
| **TPOT** | 单 token 生成时间 | decode 每步耗时；随 KV 变长而上升（访存量增大） |
| **吞吐** | 单位时间输出 token 数（服务成本指标） | batch 越大吞吐越高，但被显存上限卡住 |

显存视角：

```
峰值显存 = 模型权重（固定） + KV Cache（随 batch × seq 增长） + 激活值
总延迟   = TTFT + TPOT × 生成 token 数
```

所以优化 KV Cache 有双重收益：**同样的卡能开更高并发**（省显存）、**同样的请求更快**（省带宽）。

## 6. 优化全景：三层分类

论文 *A Survey on Large Language Model Acceleration based on KV Cache Management* 把优化分为三层：

### 6.1 token 级优化（不动模型结构、不动并行）

在 token 粒度上做选择、组织、压缩：

| 方向 | 做法 | 代表 |
|------|------|------|
| **选择** | 只存最重要的 token | H2O（Heavy-Hitter）、Keyformer、SnapKV、Quest |
| **预算分配** | 层间/头间动态分配缓存额度 | PyramidKV、PyramidInfer、AdaKV、DuoAttention |
| **合并** | 合并相似/重叠的 KV 对 | 层内/层间合并、Prompt Cache |
| **量化** | 降低存储精度（FP16→INT8/INT4/FP8） | KV Cache 量化，见 op2 |
| **低秩分解** | 把 KV 矩阵压成低维表示 | LoRA 式压缩、MLA 的思路之一 |

### 6.2 模型级优化（改架构）

- **注意力分组与共享**：MQA（多头共享一份 KV）、GQA（分组共享）
- **架构更改**：MLA（低秩潜在压缩，见 op1）、YOCO、CLA、MLKV、NSA
- **非 Transformer 架构**：线性注意力、RWKV、Mamba——从根本上不需要 KV Cache

### 6.3 系统级优化（改调度与内存管理）

- **内存管理**：PagedAttention（分页）、虚拟内存适配、前缀共享（RadixAttention）
- **调度策略**：前缀感知调度提高命中率、抢占式上下文切换
- **硬件感知设计**：GPU/CPU/SSD 多级卸载（HiCache、AttentionStore、LMCache）

## 7. 五个旋钮：把"减小 KV Cache"拆成可量化的动作

回到公式，KV Cache 大小由五个因子相乘决定，每个因子都是一个独立的优化入口：

| 旋钮 | 怎么拧 | 代价 |
|------|--------|------|
| **序列长度** | 稀疏化（静态窗口 / 动态淘汰）、前缀复用 | 可能丢关键信息，长距离依赖任务掉点 |
| **注意力头数** | MQA / GQA（减少 KV head 数） | 表达能力略降，但已是主流默认 |
| **key_bits** | 量化到 INT8 / INT4 / FP8 | 精度损失，需校准（见 op2） |
| **头维度** | MLA 把 KV 压成低秩 latent；Double Sparsity 做 channel 稀疏 | 结构改动大 |
| **层数** | 只缓存部分层（YOCO / CLA / MLKV / LayerSkip） | 需训练配合或接受近似 |

**理解方式**：这五个旋钮不是互斥的，工程上通常叠加使用——比如"GQA（头数）+ FP8 KV（bits）+ 前缀复用（序列长度）+ PagedAttention（系统级）"是当下最常见的组合。

## 8. 与已有文章的衔接

- 本篇是**原理与分类**，回答"KV Cache 是什么、为什么、能怎么削"。
- **sys5** 讲的是压缩方向的**演进主线**（MHA → MSA → HCA），偏"路线怎么变过来的"。
- **sys6（HiSparse）**、**sys7（DCP）**、**sys1（HiCache）** 是系统级优化的三个具体落地。
- **op1（MLA）**、**op2（量化）** 对应模型级与量化两个旋钮。
- **下一篇 sys10** 从原理走到**工程实现**：SGLang 的两级内存池、vLLM 的 Block 管理、以及百度 AttentionStore 在昆仑芯上的实测数据。
