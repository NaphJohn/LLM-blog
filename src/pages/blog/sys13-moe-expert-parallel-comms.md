---
title: '推理系统基础设施手记（十三）：MoE 专家并行的通信与计算——Dispatcher、DeepEP、Grouped GEMM、EPLB 怎么串成一条流水'
description: '把 MoE 推理的一次前向拆成五步：路由 → dispatch → Grouped GEMM → combine → 加权还原，看清四个组件各自解决哪一段问题。为什么 dispatch 的难点是「变长」而不是「发出去」、DeepEP 的 normal 与 low-latency 两种模式对应两种 SLA、Grouped GEMM 真正的收益不是省 launch 而是把 E 个尾巴摊成一个、EPLB 为什么只能在推理侧靠统计加搬运。附 decode 一步的时间账本与参数决策表。'
pubDate: 2026-11-17
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys13-moe-expert-parallel-comms
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

MoE 推理的一条数据流是固定的：**路由决定 token 去哪 → dispatch 把 token 发过去 → Grouped GEMM 把这批 token 一次算完 → combine 把结果拉回来 → 按 gate 权重加权还原**。

这四个组件经常被单独介绍，但它们的价值只有连起来看才成立：**dispatch 和 combine 决定了单步延迟的下限，Grouped GEMM 决定了这批 token 的算力利用率，EPLB 决定了跑几万步之后这条流水会不会歪掉**。单看任何一个都会得出错误结论——比如以为 Grouped GEMM 只是"省 kernel launch"，或者以为 EPLB 是可有可无的运维工具。

本篇不讲 kernel 内部（TMA、warp specialization、swizzle 那些留给专门的文章），只讲**系统层语义**：每一段在解决什么问题、代价记在谁头上、参数怎么定。

## 1. 先把三个词钉在正确的位置

### 1.1 EP 和 TP 的根本区别：谁在动

| 策略 | 切什么 | 通信原语 | 谁在动 | 通信量随什么变 |
|---|---|---|---|---|
| **TP** 张量并行 | 权重矩阵内部（按 hidden / head 切） | all-reduce / all-gather | 权重被切，**activation 不动** | 固定（每层两次，与 batch 相关） |
| **PP** 流水并行 | 层 | 点对点传 activation | activation 单向流过 | 固定（只在 stage 边界） |
| **EP** 专家并行 | 专家（整块放到不同卡） | **all-to-all** | 专家驻留不动，**token 在动** | **随 top-k、batch、路由分布变化** |

最后一行是 EP 所有麻烦的根源。TP 的通信量是**可以事先算出来的**——你知道 hidden size、知道 batch，就知道要传多少。EP 的通信量**必须在运行时才知道**：同样是 4096 个 token、top-8、128 个专家，路由均匀时每卡收 256 个，路由歪掉时有的卡收 800 个、有 的卡收 30 个。

这就是为什么 EP 需要专门造一个通信库（DeepEP），而 TP 的 all-reduce 用 NCCL 就够了：**NCCL 解的是"定长集合通信"，EP 需要的是"变长、稀疏、目的地由数据决定"的通信**。

### 1.2 "Dispatcher" 有两个意思，别搞混

站内 `fw7` 里出现的 dispatcher 是 **CUDA Graph 的图模式分发器**——vLLM 运行时按 batch 组成决定这一批走 FULL 图还是 PIECEWISE 图。它做的事是「选哪张图重放」。

本篇说的 dispatcher 是 **MoE 的 token 分发器**——按路由结果把 token 的 hidden state 送到专家所在的卡上。它做的事是「搬数据」。

两个东西同名不同物，唯一共同点是都叫 dispatch。搜索命中 `fw7` 时别以为那里讲了 MoE。

## 2. 一次 MoE 前向：五步与两个分界

<div class="fig">
<svg viewBox="0 0 680 300" width="100%" font-size="13" font-weight="400" role="img" aria-label="MoE 一次前向的五步数据流">
<title>图 1：一个 MoE 层的前向被切成五步，中间隔着两道通信</title>
<defs>
<marker id="m1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<text x="340" y="24" font-size="12" fill="#1a1a1a" font-weight="600" text-anchor="middle">单层 MoE：输入 x（每卡本地 token）</text>

<rect x="30" y="46" width="104" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="0.8"/>
<text x="82" y="68" font-size="11" fill="#1d4ed8" font-weight="600" text-anchor="middle">1 · Route</text>
<text x="82" y="86" font-size="10" fill="#1a1a1a" text-anchor="middle">gate → top-k</text>

<rect x="150" y="46" width="120" height="56" rx="6" fill="#fef2f2" stroke="#ef4444" stroke-width="1.2"/>
<text x="210" y="68" font-size="11" fill="#b91c1c" font-weight="600" text-anchor="middle">2 · Dispatch</text>
<text x="210" y="86" font-size="10" fill="#1a1a1a" text-anchor="middle">all-to-all（变长）</text>

<rect x="286" y="46" width="132" height="56" rx="6" fill="#ecfdf5" stroke="#10b981" stroke-width="0.8"/>
<text x="352" y="68" font-size="11" fill="#047857" font-weight="600" text-anchor="middle">3 · Grouped GEMM</text>
<text x="352" y="86" font-size="10" fill="#1a1a1a" text-anchor="middle">E 个专家一次算完</text>

<rect x="434" y="46" width="120" height="56" rx="6" fill="#fef2f2" stroke="#ef4444" stroke-width="1.2"/>
<text x="494" y="68" font-size="11" fill="#b91c1c" font-weight="600" text-anchor="middle">4 · Combine</text>
<text x="494" y="86" font-size="10" fill="#1a1a1a" text-anchor="middle">all-to-all（反向）</text>

<rect x="570" y="46" width="82" height="56" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="0.8"/>
<text x="611" y="68" font-size="11" fill="#1d4ed8" font-weight="600" text-anchor="middle">5 · 加权</text>
<text x="611" y="86" font-size="10" fill="#1a1a1a" text-anchor="middle">Σ gateᵢ·yᵢ</text>

<line x1="134" y1="74" x2="148" y2="74" stroke="#6b7280" stroke-width="1" marker-end="url(#m1)"/>
<line x1="270" y1="74" x2="284" y2="74" stroke="#6b7280" stroke-width="1" marker-end="url(#m1)"/>
<line x1="418" y1="74" x2="432" y2="74" stroke="#6b7280" stroke-width="1" marker-end="url(#m1)"/>
<line x1="554" y1="74" x2="568" y2="74" stroke="#6b7280" stroke-width="1" marker-end="url(#m1)"/>

<rect x="150" y="112" width="404" height="24" rx="4" fill="#fff5f5" stroke="#ef4444" stroke-width="0.5" stroke-dasharray="3 3"/>
<text x="352" y="128" font-size="10" fill="#b91c1c" text-anchor="middle">这两段是纯通信，无法被计算掩盖（除非做 chunk 级流水）</text>

<rect x="30" y="164" width="622" height="52" rx="6" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="46" y="182" font-size="11" fill="#1a1a1a" font-weight="600">被 dispatch 顺手完成的三件事（容易被忽略）</text>
<text x="46" y="200" font-size="10" fill="#6b7280">① 按专家重排 token（permute） ② 把变长的段拼成连续 buffer ③ 生成 offsets —— 正好是 Grouped GEMM 要的输入布局</text>

<rect x="30" y="232" width="622" height="52" rx="6" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="46" y="250" font-size="11" fill="#854f0b" font-weight="600">关键观察：Grouped GEMM 的输入不是「设计出来的」，是 dispatch 的副产物</text>
<text x="46" y="268" font-size="10" fill="#854f0b">所以这两个组件必须一起设计——先定 GEMM 的布局要求，再定通信的打包方式，反过来会多一次重排</text>
</svg>
</div>

五步里第一步和第五步是纯本地计算（gate 矩阵乘、按权重求和），中间三步是问题所在。两个分界点值得记住：

1. **dispatch 之前，token 是「按请求排的」；dispatch 之后，token 是「按专家排的」。** 这个重排不是附带效果，而是 Grouped GEMM 能成立的前提。
2. **combine 之后必须把所有分片收齐才能进下一层。** 所以 combine 是纯暴露的延迟，没有任何可以藏在后面的计算（除非把下一层的非依赖部分提前）。

## 3. Dispatch：难点不是"发出去"，是"变长"

### 3.1 普通 all-to-all 为什么不够用

标准的 all-to-all 假设**每个 rank 发给每个 rank 的数据量相等**。张量并行换 Shard 就是这种场景，所以 NCCL 的 `all_to_all` 够用。

MoE 的 dispatch 完全不符合这个假设。看图 2：

<div class="fig">
<svg viewBox="0 0 680 320" width="100%" font-size="13" font-weight="400" role="img" aria-label="变长 dispatch 的问题与两种解法">
<title>图 2：每卡发往每卡的数量都不同，定长 all-to-all 会浪费一半带宽</title>

<text x="40" y="26" font-size="11" fill="#1a1a1a" font-weight="600">实际负载：4 个 rank，各自的 token 按路由散到 4 个目的地</text>

<g font-size="9" fill="#6b7280">
<text x="66" y="48" text-anchor="middle">→ R0</text>
<text x="126" y="48" text-anchor="middle">→ R1</text>
<text x="186" y="48" text-anchor="middle">→ R2</text>
<text x="246" y="48" text-anchor="middle">→ R3</text>
</g>

<g font-size="10">
<text x="20" y="72" fill="#1a1a1a">R0 发出</text>
<text x="20" y="102" fill="#1a1a1a">R1 发出</text>
<text x="20" y="132" fill="#1a1a1a">R2 发出</text>
<text x="20" y="162" fill="#1a1a1a">R3 发出</text>
</g>

<rect x="50" y="58" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="84" y="58" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="118" y="58" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="152" y="58" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="186" y="58" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="220" y="58" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="254" y="58" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>

<rect x="50" y="88" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="84" y="88" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="118" y="88" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="152" y="88" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="186" y="88" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="220" y="88" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="254" y="88" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>

<rect x="50" y="118" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="84" y="118" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="118" y="118" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="152" y="118" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="186" y="118" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="220" y="118" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="254" y="118" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>

<rect x="50" y="148" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="84" y="148" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="118" y="148" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="152" y="148" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="186" y="148" width="32" height="20" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.5" stroke-dasharray="2 2"/>
<rect x="220" y="148" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="254" y="148" width="32" height="20" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>

<text x="306" y="72" font-size="10" fill="#b91c1c">最热：R0→R3 共 4 个</text>
<text x="306" y="102" font-size="10" fill="#6b7280">最冷：R1→R0 共 1 个</text>
<text x="306" y="132" font-size="10" fill="#6b7280">同一行内差 4 倍</text>

<line x1="30" y1="186" x2="650" y2="186" stroke="#e5e7eb" stroke-width="1"/>

<text x="40" y="208" font-size="11" fill="#1a1a1a" font-weight="600">解法 A：先探路（counts 走一趟小消息）</text>
<rect x="40" y="218" width="132" height="30" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="106" y="237" font-size="10" fill="#1d4ed8" text-anchor="middle">all-to-all(counts) 本地 4 字节</text>
<rect x="188" y="218" width="150" height="30" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="263" y="237" font-size="10" fill="#047857" text-anchor="middle">按真实长度发数据，零浪费</text>
<text x="356" y="237" font-size="10" fill="#6b7280">代价：多一次同步，延迟 +1 RTT</text>

<text x="40" y="278" font-size="11" fill="#1a1a1a" font-weight="600">解法 B：预分配上限（固定 slot）</text>
<rect x="40" y="288" width="132" height="26" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="106" y="305" font-size="10" fill="#1d4ed8" text-anchor="middle">每专家固定 N 个 slot</text>
<rect x="188" y="288" width="150" height="26" rx="4" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="263" y="305" font-size="10" fill="#854f0b" text-anchor="middle">一次通信，延迟可预测</text>
<text x="356" y="305" font-size="10" fill="#6b7280">代价：空 slot 白占带宽，或超限丢 token</text>
</svg>
</div>

两条路没有绝对优劣，它们对应**两种不同的 SLA**：

- **Preill 偏高吞吐** → 走 A：数据量大，浪费 30% 带宽很贵，多一个 RTT 无所谓（本来就要算几百毫秒）；
- **Decode 偏高交互** → 走 B：每步数据量小，多一个 RTT 直接加到 TPOT 上，宁可浪费也不能多等。

### 3.2 DeepEP 的系统语义：两种模式对应两种 SLA

DeepEP 的设计基本就是上面这个结论的工程化。它提供两类 kernel，差别不在算法优雅度，而在**假设了哪种负载**：

| | **normal kernel** | **low-latency kernel** |
|---|---|---|
| 目标负载 | Prefill / 训练（大 batch、变长） | Decode（小 batch、每步都走一遍） |
| 跨节点路径 | 节点内 NVLink 转发 + 节点间 RDMA | 纯 RDMA 直发（IBGDA） |
| 缓冲策略 | 按需分配（走解法 A） | 固定 slot（走解法 B） |
| 能否进 CUDA Graph | 不能（形状不定） | 能（形状固定）——这就是站内 `aihot-20260912` 里 DeepEP v2 ElasticBuffer 让「跨节点 MoE decode 进图」的原因 |
| SM 占用 | 大量 SM 参与搬运 | 只占个位数百分比的 SM，剩下的给计算 |
| 重叠方式 | 通信与计算分块流水 | hook 式：先发出去，等需要结果时再 `hook()` |

两个细节值得单独说：

**第一，normal kernel 的「转发」不是绕路，是省路。** 如果每个 rank 直接把 token 用 RDMA 发给所有目标 rank，64 卡下就是 64×64 的连接，跨节点链路会被占满。DeepEP 的做法是：节点内先用 NVLink 把 token 汇聚到「负责目标节点的那张卡」，再由它走一次 RDMA。跨节点流量从 O(卡数²) 降到 O(节点数)，代价是节点内多一跳 NVLink——而 NVLink 比 RDMA 快一个量级，这个置换划算。

**第二，low-latency 模式存在的唯一理由是「不能等」。** Decode 每步只有几十到几百个 token，通信量小到几乎不占带宽，但**延迟是纯增加在关键路径上的**。所以这时候要的不是带宽效率，是确定性和低占用：固定 buffer 保证形状不变（能进图）、少量 SM 保证不抢计算资源、hook 保证在等通信的时候还能干别的活。

### 3.3 FP8 dispatch：传之前先把位宽砍掉

一个容易被低估的优化：dispatch 传的是 hidden state，如果权重已经是 FP8，那么 token 也可以先量化再发，通信量直接减半。

代价是要带 scale，而且要**按 token 带**（per-token scale），否则精度掉得厉害。这又反过来约束了 Grouped GEMM 的实现——它必须能接受"每个 group 一份 scale"这种布局，不能只吃全局一个 scale。这就是下一节要讲的接口耦合。

## 4. Grouped GEMM：把 E 次启动变成 1 次，把 E 个尾巴摊成 1 个

### 4.1 朴素做法错在哪

MoE 层的 FFN 本质是 E 个独立的小矩阵乘：专家 e 拿自己的 $M_e$ 个 token，乘自己的权重 $W_e$。最直白的实现就是循环：

```python
for e in range(num_experts):
    out[e] = x_e @ W[e]   # E 次 kernel launch
```

这个写法有两个致命问题，而且第二个比第一个严重得多：

<div class="fig">
<svg viewBox="0 0 680 290" width="100%" font-size="13" font-weight="400" role="img" aria-label="循环调用 GEMM 与 Grouped GEMM 的对比">
<title>图 3：不是 launch 开销的问题，是每个 expert 都得单独吃一次尾部效应</title>

<text x="40" y="26" font-size="11" fill="#b91c1c" font-weight="600">循环调用：E 次 launch，每次都有自己的填充 + 排空</text>

<g font-size="9">
<text x="20" y="56" fill="#6b7280">e0</text>
<text x="20" y="86" fill="#6b7280">e1</text>
<text x="20" y="116" fill="#6b7280">e2</text>
<text x="20" y="146" fill="#6b7280">e3</text>
</g>

<rect x="44" y="44" width="70" height="18" rx="2" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="116" y="44" width="10" height="18" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>
<text x="140" y="57" font-size="9" fill="#b91c1c">M 小 → 算不满</text>

<rect x="44" y="74" width="128" height="18" rx="2" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="174" y="74" width="10" height="18" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>

<rect x="44" y="104" width="30" height="18" rx="2" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="76" y="104" width="10" height="18" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>
<text x="100" y="117" font-size="9" fill="#b91c1c">只有 30 行，tile 都铺不满 → 空转</text>

<rect x="44" y="134" width="96" height="18" rx="2" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<rect x="142" y="134" width="10" height="18" rx="2" fill="#fca5a5" stroke="#ef4444" stroke-width="0.5"/>

<text x="220" y="60" font-size="10" fill="#6b7280">每个红块是一段「启动 + 排空」</text>
<text x="220" y="78" font-size="10" fill="#6b7280">E 个专家 = E 段，谁也填不满谁的空</text>
<text x="220" y="112" font-size="10" fill="#b91c1c">总时间 ≈ Σ max(计算, 启动开销)</text>
<text x="220" y="130" font-size="10" fill="#b91c1c">E 越大越糟：128 个专家就是 128 次白扔</text>

<line x1="30" y1="172" x2="650" y2="172" stroke="#e5e7eb" stroke-width="1"/>

<text x="40" y="196" font-size="11" fill="#047857" font-weight="600">Grouped GEMM：1 次 launch，所有 expert 的 tile 进同一个队列</text>

<rect x="44" y="210" width="180" height="18" rx="2" fill="#86efac" stroke="#10b981" stroke-width="0.5"/>
<text x="134" y="223" font-size="9" fill="#064e3b" text-anchor="middle">e0 的 tile</text>
<rect x="226" y="210" width="240" height="18" rx="2" fill="#6ee7b7" stroke="#10b981" stroke-width="0.5"/>
<text x="346" y="223" font-size="9" fill="#064e3b" text-anchor="middle">e1 的 tile</text>
<rect x="468" y="210" width="110" height="18" rx="2" fill="#86efac" stroke="#10b981" stroke-width="0.5"/>
<text x="523" y="223" font-size="9" fill="#064e3b" text-anchor="middle">e2 + e3</text>

<text x="44" y="244" font-size="10" fill="#047857">一段连续执行，只在结尾排空一次</text>
<text x="44" y="262" font-size="10" fill="#6b7280">小的专家（e2/e3 只有 30 行）被后面的专家接着填——这是分组调度，不是「依次算」</text>
</svg>
</div>

- 表面问题：E 次 launch，每次都有固定开销。128 个专家就是 128 次。
- **真问题：每个专家单独算，就要单独吃一次尾部效应。** 一个只有 30 行的专家和一个 800 行的专家，在 GPU 上都是「启动 → 铺 tile → 排空」。30 行那次，SM 利用率可能只有 10%，但这个空转无法被别的专家填补——因为它们在另一次 launch 里。

Grouped GEMM 把这两件事一起解决：**所有专家的所有 tile 进同一个全局队列，谁来取就谁算**。小的专家自然被后面的工作填满，只在整批结束时排空一次。尾部效应从 E 次降到 1 次。

顺带的好处：launch 从 E 次降到 1 次，CPU 侧不再成为瓶颈（尤其在图模式下，一次 launch 才能被捕获）。

### 4.2 接口长什么样：offsets 是全部约定

Grouped GEMM 的接口看上去很奇怪——不传 batch size，传一个 offsets 数组：

```python
# counts: 每个专家收到的 token 数（e0=3, e1=0, e2=5, e3=2）
counts  = torch.tensor([3, 0, 5, 2])
offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])   # [0, 3, 3, 8, 10]
```

`offsets[e]:offsets[e+1]` 就是专家 e 在**已经排好序的 token 张量**里的行区间。注意 e1 的 `offsets` 是 `[3, 3]`——**空组，长度 0**，合法且必须支持，因为路由完全可能冷落某个专家。

这解释了两件事：

1. **为什么 dispatch 必须顺手做 permute。** Grouped GEMM 要求同一专家的 token 连续存放，而 dispatch 本来就要把 token 打包发送——打包的顺序天然可以按专家排。如果这两步分开做，就多一次全量 gather。
2. **为什么这个接口天然适配变长。** 没有 capacity factor、没有 padding、没有 drop。每个专家只算自己真的收到的 token，一点不浪费。

一个可验证的等价实现（`torch._grouped_mm` 需要 PyTorch 2.9+，签名以你的版本为准；老版本可用 `grouped_gemm` 第三方包）：

```python
import torch

counts  = torch.tensor([3, 0, 5, 2], device='cuda')
offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
T, H, I = int(counts.sum()), 1024, 512

x = torch.randn(T, H, dtype=torch.bfloat16, device='cuda')      # 已按专家排序
w = torch.randn(4, I, H, dtype=torch.bfloat16, device='cuda')   # 4 个专家的权重

# grouped：一次调用，内部按 offsets 分段
out = torch._grouped_mm(x, w.transpose(1, 2), offs=offsets)

# 参考实现：循环版本，语义必须完全一致
ref = torch.empty_like(out)
for e in range(4):
    s, t = int(offsets[e]), int(offsets[e + 1])
    if t > s:                       # 空组直接跳过，不要算
        ref[s:t] = x[s:t] @ w[e].T

torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
```

这段代码里有个细节值得抄：**空组必须显式跳过**。`t > s` 那个判断不是防御性编程，是真的会发生——你跑 10 万步一定会遇到某个专家零 token。

### 4.3 masked 还是 grouped：capacity factor 的遗产

早期 MoE（GShard / Switch）用 **capacity factor** 给每个专家分配固定容量的 buffer：算得出、能预分配、能定长通信。代价是超限的 token 被**丢弃**（drop），不足的**填零**（pad）。

MegaBlocks 那篇（MLSys 2023）的核心论点就是：这两者都是浪费，MoE 应该直接写成**块稀疏 GEMM**，根本不设容量上限。站内 `aihot-20260925` 引的那组数字——比 Tutel 最优配置快 40%、比 Megatron-LM 稠密快 2.4×、零 token 丢弃——说的就是这件事。

| | masked / 定容 GEMM | grouped / 变长 GEMM |
|---|---|---|
| 容量 | 固定 capacity factor | 无上限，按实际 counts |
| 超限 token | **丢弃**（质量损失） | 全部计算 |
| 空位 | 填零，仍占算力 | 不存在 |
| 通信 | 定长，简单 | 变长，需要 counts 交换 |
| 推理侧适用性 | 低（质量不可接受） | 高 |

**推理几乎不应该用 capacity factor。** 训练时丢掉 5% 的 token 还可以靠下一批补回来，线上服务丢掉的是用户的真实输出。所以现代推理栈一律走变长 grouped 路线——代价就是要处理 3.1 里那套变长通信。

### 4.4 量化的耦合：per-group scale

前面提到 FP8 dispatch 要带 per-token scale。到了 Grouped GEMM 这一侧，量化又提出第二个要求：**每个专家需要自己的激活 scale**。

原因很实在：384 个专家的激活分布不一样。金融文本的专家和代码的专家，激活范围可能差一个量级。如果全模型共用一个 scale，小范围的专家会被截断得很惨。所以 FP8 grouped GEMM 的实际接口长这样：

```text
out = grouped_gemm(x_fp8, w_fp8,
                   x_scale[group],     # 每组一份激活 scale
                   w_scale[group],     # 每组一份权重 scale
                   offsets)
```

这就是为什么 DeepSeek 的 DeepGEMM 从一开始就同时支持 contiguous grouped 和 masked 两种变体，并且把 per-group scaling 做进 kernel——**量化方案和 GEMM 接口是绑在一起设计的，不能分开选**。

## 5. Combine：逆操作，但延迟账不一样

combine 把各专家的输出按原路发回，然后每张卡按 top-k 的 gate 权重加权求和。形式上它是 dispatch 的逆，但三个地方不对称：

| | dispatch | combine |
|---|---|---|
| 方向 | token → 专家所在卡 | 结果 → token 原卡 |
| 时机 | Grouped GEMM **之前** | Grouped GEMM **之后** |
| 权重用在哪 | 不用（路由结果只决定地址） | **必须用**（Σ gateᵢ·yᵢ） |
| 能否被掩盖 | 可以（先发，等需要用之前再等） | **基本不能**——收齐才能进下一层 |
| 能否合并 | — | 可与下一层的 reduce-scatter 合并 |

最后一行是个实用的优化点：如果紧接着有 TP 的 all-reduce（FFN 输出要按 TP 归约），那么 combine 的 all-to-all 和它方向一致，可以合并成一次通信，省一趟。反之如果分开做，就是两次全量往返。

## 6. EPLB：训练靠 loss，推理靠搬运

### 6.1 为什么推理侧没有 loss 可用

训练 MoE 时有 auxiliary loss 或 bias 项，在梯度里直接把路由往均衡方向推。**推理时没有梯度、没有 loss、不能改模型**——路由是权重算出来的既定结果，热点专家就是热点专家。

于是只剩两条路：

1. **路由侧**：加 bias 强行改 top-k 结果。DeepSeek 的 aux-loss-free 偏置就是这条路，但它是**训练期**埋进去的，部署时改不动。
2. **部署侧**：不碰路由，改专家的**物理分布**——给热点专家复制冗余副本，把副本放到空闲的卡上。

EPLB 走的是第 2 条。它和 aux-loss-free 偏置是两道保险，一个管训练期、一个管部署期，互补而不替代。

### 6.2 moe_lb 怎么读

先定一个指标：

```text
moe_lb = 最慢 rank 的负载 / 所有 rank 的平均负载
```

- 完全均衡时 = 1.0；
- 1.09 意味着**最忙的那张卡只比平均水平多干 9% 的活**。

这个数字看起来不大，但在 EP64 下意义很大：因为每个 step 都要等最慢的卡，所以**有效吞吐 ≈ 平均吞吐 / moe_lb**。1.09 已经吃掉了 9% 的算力；如果不做均衡，热点专家能把 moe_lb 推到 2.0 以上，直接腰斩。

站内 `fw10` 记了 DeepInfer 的实测：Prefill 阶段 1.074 / 1.045，Decode 阶段 1.090。

### 6.3 冗余复制 + minimax 贪心

<div class="fig">
<svg viewBox="0 0 680 330" width="100%" font-size="13" font-weight="400" role="img" aria-label="EPLB 的冗余专家复制与装箱">
<title>图 4：热点专家复制出冗余副本，再装箱到不同卡上</title>

<text x="40" y="26" font-size="11" fill="#b91c1c" font-weight="600">复制之前：e7 被 40% 的 token 命中，其余专家闲着</text>

<g font-size="10">
<text x="20" y="56" fill="#6b7280">卡 0</text>
<text x="20" y="86" fill="#6b7280">卡 1</text>
<text x="20" y="116" fill="#6b7280">卡 2</text>
<text x="20" y="146" fill="#6b7280">卡 3</text>
</g>

<rect x="56" y="44" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="57" font-size="9" fill="#1d4ed8" text-anchor="middle">e0</text>
<rect x="132" y="44" width="70" height="18" rx="3" fill="#fca5a5" stroke="#ef4444" stroke-width="0.8"/>
<text x="167" y="57" font-size="9" fill="#7f1d1d" text-anchor="middle">e7 ← 热点</text>

<rect x="56" y="74" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="87" font-size="9" fill="#1d4ed8" text-anchor="middle">e1</text>
<rect x="132" y="74" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="167" y="87" font-size="9" fill="#1d4ed8" text-anchor="middle">e2</text>

<rect x="56" y="104" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="117" font-size="9" fill="#1d4ed8" text-anchor="middle">e3</text>
<rect x="132" y="104" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="167" y="117" font-size="9" fill="#1d4ed8" text-anchor="middle">e4</text>

<rect x="56" y="134" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="147" font-size="9" fill="#1d4ed8" text-anchor="middle">e5</text>
<rect x="132" y="134" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="167" y="147" font-size="9" fill="#1d4ed8" text-anchor="middle">e6</text>

<text x="230" y="60" font-size="10" fill="#b91c1c">卡 0 扛 double 负载</text>
<text x="230" y="78" font-size="10" fill="#6b7280">其余 7 张卡在等它</text>

<line x1="30" y1="172" x2="650" y2="172" stroke="#e5e7eb" stroke-width="1"/>

<text x="40" y="196" font-size="11" fill="#047857" font-weight="600">复制之后：e7 出 3 个副本，按负载分流</text>

<rect x="56" y="212" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="225" font-size="9" fill="#1d4ed8" text-anchor="middle">e0</text>
<rect x="132" y="212" width="70" height="18" rx="3" fill="#6ee7b7" stroke="#10b981" stroke-width="0.8"/>
<text x="167" y="225" font-size="9" fill="#064e3b" text-anchor="middle">e7 ①</text>

<rect x="56" y="242" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="255" font-size="9" fill="#1d4ed8" text-anchor="middle">e1</text>
<rect x="132" y="242" width="70" height="18" rx="3" fill="#6ee7b7" stroke="#10b981" stroke-width="0.8"/>
<text x="167" y="255" font-size="9" fill="#064e3b" text-anchor="middle">e7 ②</text>

<rect x="56" y="272" width="70" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="0.5"/>
<text x="91" y="285" font-size="9" fill="#1d4ed8" text-anchor="middle">e3</text>
<rect x="132" y="272" width="70" height="18" rx="3" fill="#6ee7b7" stroke="#10b981" stroke-width="0.8"/>
<text x="167" y="285" font-size="9" fill="#064e3b" text-anchor="middle">e7 ③</text>

<text x="230" y="228" font-size="10" fill="#047857">三张卡各接 1/3，最慢的那张不再是瓶颈</text>
<text x="230" y="246" font-size="10" fill="#6b7280">选谁复制：让 max 负载下降最多的那个</text>
<text x="230" y="264" font-size="10" fill="#6b7280">不是简单地复制「最热的」</text>
<text x="230" y="288" font-size="10" fill="#b91c1c">代价：3 份 e7 权重吃 3 份显存</text>
</svg>
</div>

算法本身不复杂：按历史 `tokens_per_expert` 给每个专家算一个分数，每次贪心地选「复制之后 max 负载降得最多」的那一个，重复到冗余预算用完。目标是 **minimax**，不是平均。

这里有个容易搞错的点：**minimax 贪心不等于复制最热的专家。** 如果最热的那个复制一份后仍然是最热的，第二份复制的收益就很小了；反而复制第二热的专家，可能让新的最大值降得更多。所以选择依据是「复制后新的 max」而不是「当前的值」。

### 6.4 重排周期为什么这么长

搬专家是要付代价的：一个专家的权重（比如 hidden 2048 × intermediate 512 × 2 字节）是 MB 级，几百个专家重排一次是 GB 级的卡间搬运。而且搬运期间这些专家不能用，得等。

所以 EPLB 是**周期性**动作，不是每步动作。DeepInfer 公开的配置：

| 阶段 | 每卡冗余专家起点 | 统计窗口 | 重排周期 |
|---|---|---|---|
| Prefill | 4 个 | 100 步 | 1000 步 |
| Decode | 1 个 | 1000 步 | 10000 步 |

三个数字的选择逻辑不同：

- **冗余起点**：Prefill 的 batch 大、单步就要跑完，热点效应剧烈，多给几个副本；Decode 的 batch 小、路由相对平滑，1 个够。
- **统计窗口**：要够长才能滤掉单步抖动，但太长会跟不上真实分布变化。Prefill 100 步（每步 token 多，样本够），Decode 1000 步（每步 token 少，要多攒）。
- **重排周期**：远大于统计窗口，因为搬运贵。1000 / 10000 这个比例说明——**统计是廉价的、搬运是昂贵的，所以要「多次观察、少次动手」**。

### 6.5 那 0.0001 的扰动

一个值得学的工程细节：EPLB 给每个专家评分时，额外加一个极小偏移：

```text
score[e] = tokens_per_expert[e] + e * 1e-4
```

作用是让所有 score **严格互不相等**，避免 argmax 出现并列时不同层、不同进程返回不同结果。

这 0.0001 换来的是**可复现的调度**。在一个 384 专家、64 卡的集群里，如果两个专家分数完全相等，不同 rank 的 argmax 实现细节（甚至同一次运行的不同时刻）可能选出不同专家，导致每次重排结果不一样。在调试一个"偶发变慢"的问题时，这种不确定性会把排查成本放大十倍。

**为了确定性牺牲一点点最优性，在生产系统里比多 1% 吞吐重要得多。**

### 6.6 代价：故障域和显存

EPLB 不是免费的：

- **显存**：冗余副本是实打实的权重。Decode 每卡 6 个专家 + 1 个冗余 = 7 个，多出约 17% 的专家权重。
- **故障域**：EP64 意味着 64 张卡构成**单一通信域**，挂一张卡整实例不可用。恢复策略不是快速重启，而是重启时「从所有当前健康的 GPU 重新确定 EP size」，拿剩下的卡降级继续跑。
- **所以 Wide-EP 的安全网是冗余容量，不是恢复速度。** 这一点在做容量规划时必须算进去。

## 7. 串起来：decode 一步的时间账本

把四个组件放回同一条时间轴上，才能看清各自的权重：

| 段落 | 性质 | 能不能藏 | Decode 下的相对权重 |
|---|---|---|---|
| Route（gate + top-k） | 本地计算 | 否 | 小但不可忽略（每步都算） |
| Dispatch | **纯通信** | 部分（hook 可先发后等） | **高**——每步都付，纯加 TPOT |
| Grouped GEMM | 计算 | — | 中（decode 时 M 很小，是访存受限） |
| Combine | **纯通信** | 基本不能 | **高**——收齐才能进下一层 |
| 加权还原 | 本地计算 | 否 | 小 |

两个结论：

1. **Decode 阶段，通信占的比例远高于 Prefill。** Prefill 一次算几千个 token，Grouped GEMM 是绝对主角；Decode 一次算几百个 token，dispatch + combine 两次往返的固定延迟就变得显著。**这就是 DeepEP 要有 low-latency 模式的全部理由。**
2. **EPLB 不在这条时间轴上。** 它作用于「很多步之间」，收益体现在**多步累计**的吞吐上，而不是单步延迟。所以：**追求 TTFT / TPOT 的调优看 dispatch 和 combine，追求集群吞吐的调优看 EPLB 和 EP size**，两组旋钮解决不同的问题，混着调会互相抵消。

## 8. 参数决策表：四个旋钮怎么定

把本篇的结论压成一张可查的表：

| 旋钮 | 取小 | 取大 | 决定依据 |
|---|---|---|---|
| **EP size**（专家切几张卡） | 每卡专家多、通信域小、故障域小 | 每卡专家少、显存压力小、高交互区间纵深更好 | **看负载 SLA**：低并发长输出 → EP 开大只是烧通信；高交互（每用户 250+ tok/s）→ EP64 对 EP8 能有 10 倍差距 |
| **dispatch 模式** | — | — | **Prefill 走 normal，Decode 走 low-latency**。同一个模型两个阶段可以用不同 backend |
| **冗余专家数** | 省显存 | 热点扛得住 | **看路由偏斜程度**。先用 `moe_lb` 测，>1.15 就该加 |
| **重排周期** | 跟得快，但搬运开销大 | 稳，但跟不上分布漂移 | **看流量是否突变**。稳态流量周期放长，有明显昼夜/活动峰谷就缩短 |

关于第一行，`fw10` 里那组数据值得再贴一次（64×SM100、固定 3 个草稿 token、按 75% 逐 token 接受率折算）：

- 每用户 100 tok/s 时：EP64 约 2.66 万 tok/s/GPU，EP8 约 2.5 万——**峰值几乎一样**；
- 每用户 250 tok/s 时：EP64 仍有约 1.55 万，EP8 掉到约 1700。

**Wide-EP 买的不是峰值，是高交互性区间的纵深。** 这句话在容量规划时最容易被忽略——如果你算的是"满 batch 吞吐"，会得出"EP 开大没用"的错误结论。

## 9. 三个常见误判

**误判一：以为 Grouped GEMM 只是省 kernel launch。**
省 launch 是顺带的。真正的收益是把 E 个独立的尾部效应摊成 1 个——在专家数上百、每个专家 token 数很少的场景下，后者比前者大一个量级。反过来说，**如果你的专家数很少（比如 8 个）而且每个都吃得很满，Grouped GEMM 的收益很有限**，别指望它能救你。

**误判二：以为 EPLB 能解决路由偏斜。**
EPLB 改的是**专家的物理分布**，不改路由结果。如果某个专家真的被 40% 的 token 命中，复制 3 份能让 3 张卡各接 1/3，但总计算量一点没少。**它优化的是「等待」，不是「工作量」。** 想减少工作量得从路由侧重平衡（aux-loss-free bias）或从模型结构下手。

**误判三：以为 dispatch 的瓶颈是带宽。**
Prefill 阶段确实是带宽（数据量大），但 **Decode 阶段瓶颈是延迟**——每步通信量小到带宽根本用不满，真正吃时间的是那次往返的固定开销和同步点。所以在 Decode 上调 dispatch，优化方向是**降低延迟和占用**（固定 buffer、少 SM、hook 重叠），不是提高带宽利用率。

## 10. 与站内文章的衔接

- **`sys8`（并行切分策略全景）** 把 EP 放在五把刀里给了定性描述——"all-to-all，通信量随 top-k 和 batch 变化，需要专家负载均衡"。本篇就是把这三句话各自展开成机制：all-to-all 为什么要变长、通信量怎么变、均衡怎么做；
- **`sys12`（Mooncake）** 讲的是请求级的路由与 KV 搬运（跨实例），本篇讲的是**层内的 token 级路由**——两个不同粒度的"分发"，别混；
- **`fw10`（DeepInfer）** 的 §3 讲了 Wide-EP 的分阶段配比、§3.1 讲了 EPLB 的 minimax 贪心与 1e-4 扰动。本篇在它之上补了**另外两个组件**（dispatch 的变长语义、Grouped GEMM 的 offsets 接口），并把四者串成一条流水——`fw10` 是"DeepSeek 交出了什么"，本篇是"这四块各自为什么长这样"；
- **`fw7`（CUDA Graph）** 里的 dispatcher 是图模式分发器，与本篇的 token dispatcher 同名不同物。但两者有一个真实交点：**low-latency kernel 的固定 buffer 设计，正是为了让 MoE decode 能进图**（`aihot-20260912` 记的 DeepEP v2 ElasticBuffer 就是干这个）；
- **`op1`（MLA 算子）** 是单卡算子视角，本篇是跨卡数据流视角。MLA 压缩的是 KV，Grouped GEMM 优化的是专家计算——两者都是"让每步读写的数据变少"，但一个在注意力、一个在 FFN；
- **`fa4`（DeepSeek V4）/ `fa10`（MiniMax M3.1 Flash）** 讲了模型的专家数与稀疏度（384 专家 / top-8 这类配置）。**模型侧的专家数直接决定了 EP 的通信复杂度**——专家越多，单卡摊到的越少，dispatch 的目的地越分散。这篇的系统视角和它们的模型视角是一枚硬币的两面；
- **`aihot-20260925`（MegaBlocks 与 grouped GEMM）** 是本站唯一提到 grouped GEMM 的地方，一句话带过。本篇补上了它为什么重要、接口长什么样、和 capacity factor 的关系。

---

## 系列导航

**推理系统基础设施手记**

- [（十二）Mooncake——以 KVCache 为中心的 PD 分离架构](/blog/sys12-mooncake-kvcache-centric-serving)
- [（十一）SGLang HiCache L2 在昆仑芯 P800/P900 上的适配与调优实战](/blog/sys11-hicache-l2-kunlunxin-tuning)
- [（十）KV Cache 工程实现](/blog/sys10-kv-cache-engineering-sglang-vllm-attentionstore)
- [（九）KV Cache 全景](/blog/sys9-kv-cache-fundamentals-optimization)
- [（八）并行切分策略全景与 PD 分离](/blog/sys8-parallel-strategies-pd-disaggregation)

**相关**

- [（十）DeepInfer 深度解析](/blog/fw10-deepinfer)——Wide-EP 配比与 EPLB 实测数据的来源
- [（七）CUDA Graph](/blog/fw7-cuda-graph)——另一个同名 dispatcher 的出处
- [（一）MLA 算子](/blog/op1-mla-operator)——单卡算子视角的对照