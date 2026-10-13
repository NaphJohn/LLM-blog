---
title: '推理系统基础设施手记（十二）：Mooncake——以 KVCache 为中心的 PD 分离架构，以及它和 vLLM 的三种组合方式'
description: '一句话讲清 Mooncake 的核心优化原理：以 KVCache 为中心做存算分离。四张架构图看清 PD 分离到底发生在哪一层——Mooncake 的 Conductor 只调度、实例角色层才分离；对比 vLLM 的 Disagg Proxy + KV Connector 方案与非 PD 单体架构；最后看 Mooncake 如何通过 Connector 接口挂进 vLLM，成为外置 KV 层。'
pubDate: 2026-10-13
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys12-mooncake-kvcache-centric-serving
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

Mooncake（月之暗面 Kimi 的推理系统，FAST'25 论文《Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving》）的核心优化原理一句话：**以 KVCache 为中心做存算分离**——把「算 KV」（prefill）和「用 KV」（decode）拆成两类独立集群，用一个分层分布式 KV 池统一存储和复用前缀，让 KV 在硬件之间低成本流动，从而同时缓解算力闲置、显存容量和重复计算三个问题。

这篇用四张架构图把三件事讲清楚：Mooncake 自己长什么样、PD 分离到底发生在哪一层（一个流传很广的误解是「在 router 里」）、以及 Mooncake 如何以「外置 KV 层」的身份挂进 vLLM。

## 1. 核心原理：把系统围着 KVCache 转

传统推理系统围着**模型权重**转：请求来了，找一张卡，算完拉倒。Mooncake 把主次颠倒过来——KVCache 是一等公民，所有设计围着它转。为什么？

### 1.1 prefill 和 decode 的负载完全不对称

| 阶段 | 干什么 | 瓶颈 |
|------|--------|------|
| Prefill | 一次性算完整段 prompt 的全部 KV | **计算密集**，GPU 算力吃满 |
| Decode | 每生成一个 token，读一遍全部 KV | **访存密集**，卡在带宽不在算力 |

混在一个实例里，两类请求互相拖累：prefill 的算力峰值会把 decode 的延迟抬上去，decode 的空闲算力又补不上 prefill 的缺口。拆开后两类集群各自都能跑满自己的瓶颈资源，这就是 PD 分离（Prefill/Decode Disaggregation）的动机，sys8 已经讲过原理，这篇看工业级实现。

### 1.2 前缀复用是一座金矿

真实负载里前缀重复率高得惊人：system prompt、多轮对话历史、few-shot 模板、长文档的公共部分。同一段前缀的 KV 算一次存下来，后续请求直接复用，prefill 的计算量可以成倍下降，TTFT 随之塌掉。**能不能跨请求、跨实例地缓存和找到这些 KV，决定了系统的实际上限。**

### 1.3 一台机器的显存装不下所有 KV

KV Cache 随上下文变长线性膨胀（sys9 算过账），单卡 HBM 很快见底。自然的解法是分层：GPU HBM 放热数据，CPU DRAM 做主力，SSD 存冷前缀——代价是搬运，于是又需要一个足够快的传输层把分层藏起来。

一句话：**算 KV 的和用 KV 的拆开（PD 分离），KV 统一存起来（分布式池），用 RDMA 让它在层间和实例间流动（传输引擎），调度时优先去「已经有 KV」的地方（KV 感知路由）**——四块拼起来就是 Mooncake。

## 2. Mooncake 架构全景

<div class="fig">
<svg viewBox="0 0 680 580" width="100%" font-size="13" font-weight="400" role="img" aria-label="Mooncake PD 分离架构">
<title>图 1：Mooncake 的 PD 分离架构</title>
<defs>
<marker id="ar1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<rect x="240" y="40" width="200" height="44" rx="8" fill="#f6f8fa" stroke="#30363d" stroke-width="0.5"/>
<text x="340" y="62" font-size="13" fill="#1a1a1a" text-anchor="middle" dominant-baseline="central">请求 Request</text>
<path d="M340 84 L340 116" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)"/>

<rect x="190" y="120" width="300" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="140" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Conductor 全局调度器</text>
<text x="340" y="158" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">KV 感知路由 · 只指路，不搬 KV</text>
<path d="M270 176 L215 224" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)"/>
<path d="M410 176 L465 224" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)"/>

<rect x="110" y="230" width="200" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="210" y="250" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Prefill-only 实例</text>
<text x="210" y="268" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">启动指定角色 · 只算 KV</text>
<rect x="370" y="230" width="200" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="470" y="250" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Decode-only 实例</text>
<text x="470" y="268" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">启动指定角色 · 只生成</text>

<rect x="240" y="330" width="200" height="56" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="340" y="350" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Transfer Engine</text>
<text x="340" y="368" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">RDMA / NVLink 搬 KV</text>
<path d="M215 286 L295 324" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)"/>
<path d="M465 286 L385 324" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)"/>
<text x="228" y="302" font-size="11" fill="#6b7280" text-anchor="middle" dominant-baseline="central">KV 搬运</text>

<rect x="80" y="430" width="520" height="110" rx="16" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="340" y="450" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Mooncake Store · 分布式 KV 池</text>
<rect x="100" y="468" width="146" height="50" rx="8" fill="#f6f8fa" stroke="#10b981" stroke-width="0.5"/>
<text x="173" y="485" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">GPU HBM</text>
<text x="173" y="503" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">低延迟 · 容量小</text>
<rect x="266" y="468" width="146" height="50" rx="8" fill="#f6f8fa" stroke="#10b981" stroke-width="0.5"/>
<text x="339" y="485" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">CPU DRAM</text>
<text x="339" y="503" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">容量适中 · 主战场</text>
<rect x="432" y="468" width="146" height="50" rx="8" fill="#f6f8fa" stroke="#10b981" stroke-width="0.5"/>
<text x="505" y="485" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">SSD 存储</text>
<text x="505" y="503" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">大容量 · 冷前缀</text>
<path d="M340 386 L340 424" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)" marker-start="url(#ar1)"/>

<text x="340" y="562" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">PD 分离在实例角色层实现；Conductor 只负责调度</text>
</svg>
</div>

五个角色各管一件事：

| 角色 | 职责 | 关键点 |
|------|------|--------|
| **Conductor** | 全局调度器，接收请求、决定去哪个实例 | 按各节点缓存命中情况做 KV 感知路由；**只指路，不碰数据面** |
| **Prefill-only 实例** | 算 KV，算完即交棒 | 计算密集型，可把 batch 堆满追求算力利用率 |
| **Decode-only 实例** | 拿 KV 逐 token 生成 | 访存密集型，延迟敏感 |
| **Transfer Engine** | 在实例间、实例与 Store 间搬运 KV | 基于 RDMA 的通用传输库，支持批量收发 |
| **Mooncake Store** | 分布式 KV 池，提供 put/get 接口 | HBM / DRAM / SSD 三层，token 粒度寻址 |

一次请求的完整旅程：Conductor 先查 Store——**前缀命中就直接跳过对应段的 prefill**；未命中的部分调度到某个 P 实例计算，产出的 KV 经 Transfer Engine 写到 Store 并传给选定的 D 实例，D 边生成边把新 KV 异步回写。Decode 阶段新增的 KV 增量回写 DRAM/SSD，供后续多轮对话复用。

注意一个细节：Mooncake 的实例不只有 P/D 两种角色，还可以配成「两头都做」的普通实例，按集群负载灵活混布——PD 分离是**部署策略**，不是硬编码的拓扑。

## 3. PD 分离到底发生在哪一层

这是理解 Mooncake 和 vLLM 差异的关键，也直接回答一个常见误解：**「PD 分离是在 router 里做的吗？」——不是。**

### 3.1 Mooncake：拆在实例角色层，调度在 Conductor

图 1 里 Conductor 只是回答「这个请求该去哪个实例」，它不切分请求、不搬运 KV。真正的阶段拆分发生在**实例层**：进程启动时就被指定为 prefill-only 或 decode-only，两个阶段的边界由「实例只会干什么」天然划定；KV 的搬运完全在数据面由 Transfer Engine 完成。Router 指路，实例分工，传输层搬运——三件事分开。

### 3.2 vLLM：Proxy 协调 + 独立进程 + Connector 传输

vLLM 的 PD 分离是另一套组合拳，没有全局 Conductor：

<div class="fig">
<svg viewBox="0 0 680 525" width="100%" font-size="13" font-weight="400" role="img" aria-label="vLLM PD 分离架构">
<title>图 2：vLLM 的 PD 分离架构</title>
<defs>
<marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<rect x="240" y="40" width="200" height="44" rx="8" fill="#f6f8fa" stroke="#30363d" stroke-width="0.5"/>
<text x="340" y="62" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">客户端 Client</text>
<path d="M340 84 L340 116" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>

<rect x="200" y="120" width="280" height="44" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="142" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">API Server（OpenAI 兼容）</text>
<path d="M340 164 L340 196" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>

<rect x="200" y="200" width="280" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="220" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Disagg Proxy</text>
<text x="340" y="238" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">两阶段协调 · 先发 P 再发 D</text>
<path d="M270 256 L215 304" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>
<path d="M410 256 L465 304" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>

<rect x="110" y="310" width="200" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="210" y="330" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">P 实例 · Producer</text>
<text x="210" y="348" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">独立进程 · 只做 prefill</text>
<rect x="370" y="310" width="200" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="470" y="330" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">D 实例 · Consumer</text>
<text x="470" y="348" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">独立进程 · 只做 decode</text>

<rect x="110" y="420" width="460" height="56" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="340" y="440" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">KV Connector</text>
<text x="340" y="458" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">NIXL / Mooncake · 由 kv_transfer_config 指定</text>
<path d="M205 366 L250 414" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>
<path d="M475 366 L430 414" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>
<text x="222" y="396" font-size="11" fill="#6b7280" text-anchor="middle" dominant-baseline="central">KV 传输</text>

<text x="340" y="500" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">PD 分离 = Proxy 路由 + 独立 P/D 进程 + KV Connector 传输（响应流式经 Proxy 回传）</text>
</svg>
</div>

三个部件各司其职：启动时用 `--kv-transfer-config` 把进程声名为 `kv_producer`（P）或 `kv_consumer`（D）；**Disagg Proxy 承担两阶段流程协调**——先把请求发给 P 算 KV，等 KV 就绪再把请求发给 D 生成，响应流式经 Proxy 回传；KV 走哪条物理通道则由 KV Connector 决定（NIXL、Mooncake、LMCache 等）。同机 1P1D 可走 NVLink，跨机 1PnD 走 RDMA。和 Mooncake 的差别一目了然：**vLLM 的阶段边界由 proxy 的请求转发时序划定，Mooncake 由实例角色划定。**

### 3.3 参照系：非 PD 的单体架构

没有 PD 分离的世界长这样——Router 把请求分给某个 Engine 实例，实例内部靠 continuous batching + chunked prefill 把 prefill 和 decode 混在同一批次里跑，KV cache 全程留在本地 GPU（可 offload 到 CPU），**没有任何跨实例的 KV 搬运**：

<div class="fig">
<svg viewBox="0 0 680 430" width="100%" font-size="13" font-weight="400" role="img" aria-label="非 PD 传统单体架构">
<title>图 3：非 PD 单体架构</title>
<defs>
<marker id="ar3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<rect x="240" y="40" width="200" height="44" rx="8" fill="#f6f8fa" stroke="#30363d" stroke-width="0.5"/>
<text x="340" y="62" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">客户端 Client</text>
<path d="M340 84 L340 114" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar3)"/>

<rect x="200" y="118" width="280" height="44" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="140" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">API Server + Router</text>
<path d="M280 162 L140 206" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar3)"/>
<path d="M340 162 L340 206" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar3)"/>
<path d="M400 162 L540 206" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar3)"/>

<rect x="50" y="210" width="180" height="170" rx="16" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="140" y="230" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Engine 实例</text>
<text x="140" y="248" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">Continuous Batching</text>
<rect x="65" y="262" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="140" y="277" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Prefill + Decode</text>
<text x="140" y="295" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">同进程交替执行</text>
<rect x="65" y="314" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="140" y="329" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">KV Cache</text>
<text x="140" y="347" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">本地 GPU · 不跨实例</text>

<rect x="250" y="210" width="180" height="170" rx="16" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="230" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Engine 实例</text>
<text x="340" y="248" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">Continuous Batching</text>
<rect x="265" y="262" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="277" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Prefill + Decode</text>
<text x="340" y="295" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">同进程交替执行</text>
<rect x="265" y="314" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="329" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">KV Cache</text>
<text x="340" y="347" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">本地 GPU · 不跨实例</text>

<rect x="450" y="210" width="180" height="170" rx="16" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="540" y="230" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Engine 实例</text>
<text x="540" y="248" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">Continuous Batching</text>
<rect x="465" y="262" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="540" y="277" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Prefill + Decode</text>
<text x="540" y="295" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">同进程交替执行</text>
<rect x="465" y="314" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="540" y="329" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">KV Cache</text>
<text x="540" y="347" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">本地 GPU · 不跨实例</text>

<text x="340" y="405" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">非 PD：单实例内部完成 prefill 到 decode 全流程，KV 不跨实例流动</text>
</svg>
</div>

单体形态最简单、开箱即用，代价是两类负载互相挤压、前缀复用被锁在单机、长上下文受单卡显存限制——这正是 PD 分离和分布式缓存要解决的三件事。

## 4. Mooncake 如何起作用：挂进 vLLM 当「外置 KV 层」

前两节里 Mooncake 和 vLLM 像两个独立世界，实际上它们可以叠加。组合方式是分层的：**vLLM 的调度层（API Server / Router / Engine）完全不动，下层通过 vLLM 的 KV Connector 接口（`kv_transfer_config`）把 Mooncake 的 Transfer Engine + Store 挂进来**，替代或补充本地 KV 管理：

<div class="fig">
<svg viewBox="0 0 680 485" width="100%" font-size="13" font-weight="400" role="img" aria-label="Mooncake 与 vLLM 组合分层架构">
<title>图 4：Mooncake 与 vLLM 的分层组合</title>
<defs>
<marker id="ar4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<rect x="80" y="40" width="520" height="130" rx="16" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="62" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">vLLM 推理层 · 调度逻辑不变</text>
<rect x="100" y="80" width="140" height="50" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="170" y="98" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">API Server</text>
<text x="170" y="116" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">OpenAI 兼容</text>
<rect x="260" y="80" width="160" height="50" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="98" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Router / Proxy</text>
<text x="340" y="116" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">调度与路由</text>
<rect x="440" y="80" width="140" height="50" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="510" y="98" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Engine 实例</text>
<text x="510" y="116" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">P/D 或单体</text>
<path d="M340 170 L340 196" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar4)" marker-start="url(#ar4)"/>

<rect x="80" y="200" width="520" height="56" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="340" y="220" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">vLLM KV Connector 接口</text>
<text x="340" y="238" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">kv_transfer_config 配置入口</text>
<path d="M340 256 L340 282" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar4)" marker-start="url(#ar4)"/>

<rect x="80" y="286" width="520" height="150" rx="16" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="340" y="308" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Mooncake · 外置 KV 层</text>
<rect x="100" y="326" width="200" height="60" rx="8" fill="#f6f8fa" stroke="#10b981" stroke-width="0.5"/>
<text x="200" y="348" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Transfer Engine</text>
<text x="200" y="368" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">RDMA / NVLink 传输</text>
<rect x="320" y="326" width="260" height="60" rx="8" fill="#f6f8fa" stroke="#10b981" stroke-width="0.5"/>
<text x="450" y="348" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Mooncake Store</text>
<text x="450" y="368" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">DRAM / SSD 分布式 KV 池</text>

<text x="340" y="462" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">作用：① PD 间传 KV（可替代 NIXL）② 前缀缓存卸载复用 ③ 跨实例共享 KV</text>
</svg>
</div>

Mooncake 在这种组合下干三件事：

1. **PD 间的 KV 传输**：作为 KV Connector 的一种实现，承担 P 实例到 D 实例的 KV 搬运，和 NIXL 二选一或混用；
2. **前缀缓存卸载**：把本地放不下的前缀 KV offload 进 Store 的 DRAM/SSD 池，需要时再取回，突破单卡显存对缓存总量的限制；
3. **跨实例共享**：KV 出了单机的边界，任何实例命中即可复用，把 sys10 里 AttentionStore 式的「集群级全局索引」思路落到 vLLM 栈里。

分工很清晰：**vLLM 负责「怎么调度请求」，Mooncake 负责「KV 怎么存、怎么搬」**。这也是目前生产环境最实用的组合方式——不换调度栈，白得一套分布式 KV 层。

## 5. 三种形态对比

| 维度 | Mooncake（图 1） | vLLM PD 分离（图 2） | 非 PD 单体（图 3） |
|------|------|------|------|
| **阶段拆分位置** | 实例角色层（P-only / D-only 进程） | 独立 P/D 进程 | 无拆分，单进程内交替 |
| **路由/协调者** | Conductor（KV 感知全局调度） | Disagg Proxy（两阶段协调） | Router（只选实例） |
| **KV 搬运** | Transfer Engine（RDMA / NVLink） | KV Connector（NIXL / Mooncake） | 无，KV 不出本机 |
| **缓存复用** | 分布式 Store（HBM / DRAM / SSD 三层） | 本地 prefix cache + connector | 本地 prefix cache |
| **部署复杂度** | 高（自带全套调度与存储） | 中（配置 kv_transfer_config） | 低（开箱即用） |
| **适合场景** | 大规模集群、长上下文、前缀重的生产负载 | 灵活部署，官方 PD 方案 | 单机或小规模、快速起步 |

一句话区分：**Mooncake 把 KV 当作系统的一等公民重新设计了一切；vLLM 的 PD 分离是在既有引擎上加了一根 KV 专线；非 PD 单体则是「先跑起来再说」的默认形态。**

## 6. 与站内文章的衔接

- **sys8（PD 分离）** 讲清了为什么要拆、prefill:decode 怎么配比，本篇补上工业级实现：Mooncake 的实例角色方案与 vLLM 的 proxy + connector 方案；
- **sys9（KV Cache 全景）** 算过 KV 有多大、三层优化怎么分类，本篇的分布式 Store 正是「往容量和复用要收益」那条路的极端形态；
- **sys10（KV Cache 工程实现）** 讲单机怎么管（SGLang 两级池、vLLM 分页、AttentionStore 集群索引），本篇讲跨机怎么搬（Transfer Engine）和怎么被引擎用（Connector 接口）；
- **fw2 / fw3（vLLM / SGLang 内部）** 的 PagedAttention 与 Radix Cache 是本地缓存的基石，本篇的 Mooncake Store 可以看作把这块缓存「外置到集群」；
- **fa5（Mamba Hybrid + PD Cache）** 提醒过混合架构下缓存命中放错位置的静默错误风险——跨实例共享 KV 时，前缀哈希与调度路由的一致性同样是 Mooncake 这类系统要回答的问题。
