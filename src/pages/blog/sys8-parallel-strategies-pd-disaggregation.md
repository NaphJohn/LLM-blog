---
title: '推理系统基础设施手记（八）：并行切分策略全景与 PD 分离——TP/PP/EP/SP/CP 怎么切，PD 实例怎么配比'
description: '把五把切分刀（TP/PP/EP/SP/CP）一次讲清：各切什么维度、用什么通信原语、通信量多大、怎么组合；再系统讲 PD 分离为什么分、代价是什么，以及最关键的 prefill:decode 实例配比怎么算。'
pubDate: 2026-09-08
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys8-parallel-strategies-pd-disaggregation
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

大模型放不进一张卡，于是有了**切分**；prefill 和 decode 两个阶段的硬件脾气完全相反，于是有了**分离**。本篇讲两件事：五把切分刀（TP / PP / EP / SP / CP）各自切什么、代价是什么、怎么组合；以及 PD 分离里最容易被拍脑袋决定、但其实可以算出来的那个数——**prefill 实例 : decode 实例 = 多少**。

## 1. 五把刀：切分策略全景

| 策略 | 切什么 | 通信原语 | 通信量 | 典型约束 |
|------|--------|----------|--------|----------|
| **TP** 张量并行 | 权重矩阵（按 hidden / head 维切） | all-reduce / all-gather | **大**（每层 2 次） | 通信极频繁，基本锁在节点内（NVLink），TP ≤ 8 |
| **PP** 流水并行 | 层（按 layer 切 stage） | 点对点传 activation | **小**（只传 stage 边界） | 有流水线气泡（bubble），需 micro-batch 填满 |
| **EP** 专家并行 | MoE 的专家 | all-to-all | **中，且随 top-k / batch 变化** | 只在 MoE 模型上有意义；需要 expert 负载均衡 |
| **SP** 序列并行 | 序列（TP 切不到的 LayerNorm / Dropout / 残差部分） | reduce-scatter / all-gather | 中 | 与 TP 配套，见 sys2 |
| **CP** 上下文并行 | 序列（注意力沿序列维分片） | ring 通信 | 小（与序列长无关地摊薄） | 超长上下文，见 sys2、sys7 |

一句话区分：**TP 切"胖"（矩阵内部）、PP 切"深"（层）、EP 切"专家"、SP/CP 切"长"（序列）**。

## 2. 三把主刀展开

### TP（Tensor Parallel）

把一个大矩阵乘法按列（或行）拆到多张卡，每张卡算一部分再合并。

- **代价**：每一层的注意力与 FFN 前后都要 all-reduce，**通信频率极高**。因此 TP 几乎只能跑在节点内的 NVLink 上，跨节点用 TP 会被 PCIe/网络延迟打死。
- **实践**：单机 8 卡 → TP=8 是常见上限；再大就要换策略或混 PP。
- **易混点**：TP 切的是权重矩阵内部，所以**模型结构越小、单卡能装下时越不该用 TP**——它换来的是延迟下降，不是吞吐。

### PP（Pipeline Parallel）

按层切成若干 stage，每个 stage 一张/一组卡，像流水线一样把 micro-batch 推过去。

- **代价**：**流水线气泡**。填充阶段和排空阶段 GPU 是闲着的，micro-batch 数量越少，气泡占比越高。
- **优势**：只传 stage 边界的 activation，**通信量远小于 TP**，所以 PP 是跨节点扩展的主力。
- **实践**：用足够多的 micro-batch 把气泡压下去（micro-batches ≫ stages）。

### EP（Expert Parallel）

MoE 独有。把不同的专家放到不同卡上，token 按路由结果发到对应专家所在卡，算完再发回来。

- **代价**：**all-to-all**，通信量随 top-k、batch size 和专家数变化；更麻烦的是**专家负载不均**——热门专家那张卡会先成为瓶颈。
- **实践**：EP 常与 TP 组合（EP×TP），并且需要 capacity factor、负载均衡损失（auxiliary loss）来防止某些专家被喂爆。
- **注意**：EP 只对 MoE 有效，稠密模型没有 EP 这回事。

## 3. 怎么组合：一条常用的经验链

1. **先把 TP 顶到单节点上限**（吃满 NVLink，降低单请求延迟）。
2. **节点不够再上 PP**（跨节点通信量小，扩展主力）。
3. **MoE 模型把 EP 叠上去**，并配负载均衡。
4. **超长上下文再叠 SP / CP**（见 sys2 的 Ring Attention、sys7 的 DCP）。
5. 混用时注意**通信方向正交**：sys2 里就提过，TP×SP 混用时 all-gather 和环通信会撞车，得错开。

## 4. PD 分离：为什么要把两阶段拆开

Prefill 和 Decode 的硬件脾气是**相反**的：

| | 特征 | 瓶颈 | 适合什么卡 |
|---|---|---|---|
| **Prefill** | 一次性算完整个 prompt，大矩阵乘法 | **算力密集**（FLOPs） | 强 Tensor Core、高算力 |
| **Decode** | 每步只算一个新 token，但要反复读权重和 KV | **访存密集**（带宽） | 高 HBM 带宽、大显存 |

混在一起部署（colocated）的后果：**prefill 跑的时候算力忙、带宽闲；decode 跑的时候带宽忙、算力闲**——两边都在互相拖后腿，而且两个阶段还会互相抢调度、产生干扰。

PD 分离（Prefill-Decode Disaggregation）就是把两阶段放到不同的 GPU 池，prefill 算完把 KV Cache 传给 decode 节点（本博客 fw4 第 3 节有专门讲解与架构图）。

**收益（公开数据）**：DistServe 在同等硬件下 goodput 最高 **4.48×**、SLO 可收紧 **10.2×**；Mooncake 在 Kimi 线上负载上 **1.5–2.5×** 吞吐；Splitwise 每美元吞吐 **1.4–2.1×**（随 prompt 长度分布变化）。

## 5. PD 分离的代价与适用边界

**代价是 KV 传输。** prefill 算完，整个序列的 KV 要搬到 decode 节点：

- 量级示例：70B 模型、4000 token prompt、FP16 → KV 约 **600 MB**；走 200 Gbps InfiniBand 约 **24 ms**。
- 这 24 ms 是**直接加在请求路径上的延迟**。prompt 越短，这笔固定开销越容易吃掉分离带来的收益。

**什么时候该分**：

| 适合分离 | 不适合分离 |
|---|---|
| 长 prompt 的 RAG / Agent 负载 | 短 prompt、长生成的 chat |
| 代码补全（prompt 大、生成小） | 单机 8 卡跑中小模型 |
| 高并发、两阶段资源失衡明显 | 没有高速互联（无 NVLink / RDMA） |

> 判断口诀：**规模足够大 + 长 prompt 占比高 + 有高速互联**，三条同时满足才划算；否则老老实实混合部署。

另外一个常被忽略的点：**prefill 池和 decode 池可以用不同的 TP size**（Mooncake 已支持），因为两阶段的瓶颈不同，没必要强行统一并行度。

## 6. PD 实例配比：这个数是可以算的（本篇核心）

行业里通常写成 **XpYd**：X 个 prefill 节点、Y 个 decode 节点（Mooncake 的叫法；vLLM 早期只支持 1P1D，现在已支持 xPyD 与不同 TP 组合）。

### 6.1 配比公式

设：

```
λ    = 请求到达率（QPS）
Tin  = 平均输入长度（ISL，token）
Tout = 平均输出长度（OSL，token）
P    = 单卡 prefill 吞吐（token/s）
D    = 单卡 decode 吞吐（token/s，在目标 batch 与 TPOT 约束下）
ρ    = 目标利用率（建议 ≤ 0.7，给突发留余量）
```

则两侧所需卡数：

```
N_p = λ × Tin  / (P × ρ)
N_d = λ × Tout / (D × ρ)
```

**两边相除，λ 和 ρ 同时约掉：**

```
N_p : N_d  =  (Tin / P)  :  (Tout / D)
```

> **关键洞察**：配比与 QPS **无关**。QPS 只决定你要开多少张卡（总规模），不决定 prefill 和 decode 之间怎么分。真正决定配比的是 **输入长度 / 输出长度** 与 **两侧单卡吞吐** 这两组量。这是很多人拍脑袋配 1:1 却配错的根本原因。

### 6.2 两个反直觉的算例

设某模型实测 `P = 8000 tok/s/卡`、`D = 2000 tok/s/卡`。

**场景 A：RAG / Agent（长输入、短输出）** `Tin=4000, Tout=500`

```
N_p : N_d = (4000/8000) : (500/2000) = 0.5 : 0.25 = 2 : 1
```

→ **prefill 是 decode 的两倍**。长 prompt 场景 prefill 才是大头。

**场景 B：Chat（短输入、长输出）** `Tin=500, Tout=2000`

```
N_p : N_d = (500/8000) : (2000/2000) = 0.0625 : 1 ≈ 1 : 16
```

→ **decode 占绝对多数**。短 prompt 长生成时，prefill 几乎不费力。

（对照：DistServe 在 chatbot 负载上的经典配置是 2 prefill : 6 decode，即 **1:3**，落在中间地带，与"chat 偏 decode 重"一致。）

### 6.3 实操四步法

1. **采直方图**：从真实流量采 prompt / output 长度分布——**不要只用平均值**，长尾会决定 tail 延迟。
2. **测单卡吞吐**：分别测 continuous batching 下的 prefill 吞吐 `P` 与 decode 吞吐 `D`，且 `D` 要在你的 TPOT 目标下测。
3. **代入算配比**，然后按 `ρ ≤ 0.7` 放大取整数卡数。
4. **用 tail TTFT / ITL 验证**，再微调。DistServe 把这整件事框架化为"在 TTFT / TPOT 双 SLO 下最大化 goodput"。

### 6.4 动态调整与工程要点

- **角色互换**：闲置的 prefill 实例可以转成 decode 实例（反之亦然），是提升利用率的常用手段。
- **弹性伸缩**：配比是随负载漂移的，集群要支持改配比而不重启。
- **路由亲和性**：prefill 与 decode 就近部署，减少 KV 传输跳数；跨机柜部署时要确认 `T_transfer = KV_size / BW` 不会主导 TTFT。
- **batch 策略相反**：prefill 适合**小 batch**（降延迟），decode 适合**大 batch**（提吞吐）。
- **KV 传输优化**：量化（FP8/INT8）与稀疏化只传重要 token，可直接砍掉一半以上的传输量（与 op2 的 KV Cache 量化是同一套思路）。

## 7. 选型清单

1. **先定并行再定分离**：TP 顶满节点内 → PP 跨节点 → MoE 叠 EP → 长上下文叠 SP/CP。
2. **TP 不要跨节点**，通信频率太高。
3. **PP 要有足够 micro-batch** 压气泡。
4. **PD 分离三条件**：规模大 + 长 prompt 占比高 + 有 NVLink/RDMA。
5. **配比别拍脑袋**：用 `(Tin/P) : (Tout/D)` 算，记得它与 QPS 无关。
6. **两侧可用不同 TP size**，不必强行统一。
7. **留 burst 余量**：目标利用率 ≤ 0.7。
8. **盯住 KV 传输量**：它既是延迟来源也是成本来源，能压缩就压缩。

**串起来看**：切分解决"模型放不下"，分离解决"两阶段互相拖后腿"，而配比决定"两个池子各放多少才不浪费"。这三层加在一起，才是把一个大模型真正跑成一项服务的完整答案。

*注：本文整合自 2026-08-11 至 2026-09-08 的每日 AI 热点追踪、社区跟踪（PD 分离打通、统一内存池 PD 传输）与公开论文数据（DistServe / Mooncake / Splitwise），重写为系统化版本。*
