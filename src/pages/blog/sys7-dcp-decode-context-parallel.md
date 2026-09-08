---
title: '推理系统基础设施手记（七）：DCP —— 解码上下文并行，把 KV 缓存沿序列维度分片'
description: '长上下文 Decode 阶段的显存与通信优化：DCP 将冗余复制的 KV 缓存沿序列维度分片，并复用 Scratch Buffer 做 all_gather/reduce_scatter，与 Prefix Caching、推测解码、P/D 分离协同。'
pubDate: 2026-09-08
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys7-dcp-decode-context-parallel
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

在长上下文场景里，Decode 阶段的瓶颈不是算力而是 **KV 缓存在各卡间的冗余复制**。DCP（Decode Context Parallelism，解码上下文并行）把 KV 缓存沿**序列长度维度**分片到不同 GPU，并用 Scratch Buffer 复用做集合通信——既省显存，又降通信延迟，端到端吞吐最高可提升 **1.5%~4%**。本篇把它和本系列前面的 HiCache（sys1）、HiSparse（sys6）、P/D 分离串起来。

## 1. 为什么需要 DCP：Decode 阶段的 KV 墙

标准张量并行（TP）下，每一层注意力要把 KV 缓存放到参与该层的每张卡上。序列越长，KV 越大，而 TP 沿**隐藏维度**切分、并不切序列维度，于是：

- 每张卡都完整保存了一份 KV 缓存副本 → **大量冗余复制**，显存被无谓吃掉。
- 单卡显存先耗尽，Batch Size 上不去 → 吞吐受限，长上下文直接 OOM。

Decode 是逐 token 生成、计算强度低但访存密集，所以 KV 的"存得下、取得快"比矩阵算力更关键。DCP 正是冲着这一点来的。

## 2. 核心机制：KV 沿序列维度分片

DCP 把 KV 缓存从"每卡全量复制"改成"每卡只存自己那段"：

```
TP 切分：KV 在 hidden 维复制 → 每卡一份全序列 KV（冗余）
DCP 切分：KV 在 sequence 维分片 → 每卡只存 [start:end] 这一段（无冗余）
```

- 每个 GPU 只负责序列的一段 KV，显存占用按分片数线性下降。
- Decode 时各 GPU 并行处理自己负责的 KV 分片，再经集合通信拼回完整注意力结果。
- 与 TP 正交：TP 管权重/激活的 hidden 切分，DCP 管 KV 的 sequence 切分，二者可叠加。

## 3. 通信优化：复用 Scratch Buffer

分片带来的代价是 Decode 每步都要跨卡通信。DCP 的关键工程技巧是**复用 Scratch Buffer（临时缓冲区）**执行 `all_gather` / `reduce_scatter` 等集合通信：

- 不必为每次通信临时分配显存、再拷数据 → 省掉频繁 malloc/free 与多余拷贝。
- 通信延迟下降，端到端吞吐提升（框架实测最高 **1.5%~4%**）。
- 这点和 vLLM 的许多算子优化思路一致：把"分配+拷贝"变成"复用同一块缓冲"。

## 4. 影响的解码与预填充环节

DCP 不只改 Decode，也触达预处理：

- **Decode（核心）**：逐 token 生成时，各卡分片 KV + 集合通信拼注意力，直接加速。
- **分块预填充（Chunked Prefill）**：超长输入切成块并行预填充，压低首 Token 延迟（TTFT）。
- **缓存预填充（Cached Prefill）**：复用已计算的 KV 分片，避免重复 prefill 整段上下文。

## 5. 收益场景与协同特性

DCP 的价值在以下场景最突出：

- **长上下文推理**：数十万~上百万 token（Agent、长文档分析）时 KV 极庞大，分片让有限显存装下更长序列。
- **提高吞吐**：显存释放 → 更大 Batch Size → 整体吞吐上升。
- **与高级特性协同**：
  - **前缀缓存（Prefix Caching）**：共享前缀的 KV 分片可复用，DCP 负责存、Prefix Caching 负责命中。
  - **推测解码（Speculative Decoding）**：DCP 省下的 KV 余量让 draft model 能更重、batch 更大，加速更明显。
  - **P/D 分离（P/D Disaggregation）**：Prefill 与 Decode 拆到不同实例，DCP 在 Decode 侧继续做 KV 分片，二者正交互补。

## 6. 和本系列其他优化的位置

| 优化 | 解决什么 | 维度 |
|------|----------|------|
| HiCache（sys1） | Host KV 放哪个 NUMA 节点、NIC 怎么一起看 | 硬件拓扑/数据链路 |
| HiSparse（sys6） | HBM 当缓存、主机 DRAM 存全量 KV、GPU 只留热槽 | 分层缓存（容量墙） |
| **DCP（sys7）** | KV 沿序列维度分片 + Scratch Buffer 通信 | 并行分片（冗余/通信墙） |
| P/D 分离 | Prefill 与 Decode 实例解耦 | 架构拆分 |

三者是互补视角：HiCache 管"放哪"，HiSparse 管"留多少"，DCP 管"怎么并行切"。

## 7. ⚠️ 术语消歧：DCP 的三义

在高性能推理框架（vLLM / SGLang）语境下，DCP 几乎总是指 **Decode Context Parallelism（解码上下文并行）**。但在别处它也可能指：

- **Dynamic Compressing Prompts（动态提示词压缩）**：智能删冗余 token 压缩输入 prompt。
- **Dual-Cue Pruning（双线索剪枝）**：用于多模态（视觉-语言）模型，结合文本+视觉信息选重要视觉 token。

写文/读论文时先确认语境，别把"并行分片"和"提示压缩/视觉剪枝"混为一谈。

## 8. 小结

DCP 是长上下文 Decode 的显存/通信优化：KV 沿序列维度分片消除冗余复制，Scratch Buffer 复用压低通信延迟，与 Prefix Caching、推测解码、P/D 分离天然协同。它和 HiSparse、HiCache 一起，构成"长上下文服务三件套"的不同切面——容量墙、数据链路、并行分片。

*注：本篇基于 2026-09-08 用户提供的 DCP 技术说明整理成系统条目，吞吐量 1.5%~4% 为框架实测区间，实际收益随序列长度/批大小/通信拓扑变化。*
