---
title: （三）SGLang 原理与结构：RadixAttention 与"程序视角"的推理
description: 从 LLM 应用真实结构出发，讲清基数树前缀复用、压缩状态机约束解码、零开销 Spec V2 与 UnifiedRadixTree。
pubDate: 2026-08-01
series: vLLM 与 SGLang 框架解码手记
lang: zh
altLang: en
altHref: /en/blog/fw3-sglang-internals
layout: ../../layouts/BlogPost.astro
---

## 1. 出发点不同：SGLang 看的是"程序"

vLLM 问的问题是"**显存怎么管才不浪费**"。SGLang 问的是另一个问题：

> 真实的 LLM 应用，请求之间到底长什么样？

答案是：**高度重复**。

- **多轮对话**：第 5 轮请求的前缀，是第 4 轮的全部内容
- **Agent / ReAct**：同一套 system prompt + 工具定义，每一步都重发一遍
- **Few-shot 批量任务**：几千条请求共享同一段几千 token 的示例
- **思维树 / 自洽性采样**：从同一个中间节点分叉出 N 条分支
- **批量评测**：同一道题目模板，只有末尾选项不同

如果每条请求都从头算 prefill，这些**完全相同的前缀会被重复计算几千遍**。

SGLang 的名字本身就说明了立场：**S**tructured **G**eneration **Lang**uage —— 它把 LLM 调用看成一个有结构的程序，而不是一串互不相干的独立请求。

## 2. RadixAttention：用基数树复用前缀 KV

### 2.1 机制

vLLM 也有前缀缓存，但早期主要是**精确前缀哈希匹配**（同一段完全相同的 block 才复用）。SGLang 做得更彻底：把所有活跃请求的 KV **组织成一棵基数树（Radix Tree）**。

- 树上每个节点存一段 token 序列对应的 KV block；
- 新请求进来时，沿树做**最长前缀匹配**，匹配到的部分直接复用，只 prefill 剩下的部分；
- 用 **LRU 淘汰**管理容量，热前缀自然留在树上；
- 分支天然共享——思维树的多条分支就是树上的多个子节点。

<div class="fig">
<svg viewBox="0 0 680 290" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <rect x="250" y="14" width="180" height="34" rx="6" fill="#d1fae5" stroke="#10b981"/>
  <text x="340" y="30" font-size="11" font-weight="700" fill="#065f46" text-anchor="middle">system prompt + 工具定义</text>
  <text x="340" y="43" font-size="10" fill="#047857" text-anchor="middle">1800 token · KV 只存一份</text>

  <path d="M300 48 L180 78" stroke="#10b981" stroke-width="1.5"/>
  <path d="M340 48 L340 78" stroke="#10b981" stroke-width="1.5"/>
  <path d="M380 48 L500 78" stroke="#10b981" stroke-width="1.5"/>

  <rect x="100" y="80" width="160" height="32" rx="6" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="180" y="100" font-size="10" fill="#1e3a8a" text-anchor="middle">对话 A · 历史 620 tok</text>
  <rect x="262" y="80" width="156" height="32" rx="6" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="340" y="100" font-size="10" fill="#1e3a8a" text-anchor="middle">对话 B · 历史 340 tok</text>
  <rect x="424" y="80" width="160" height="32" rx="6" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="504" y="100" font-size="10" fill="#1e3a8a" text-anchor="middle">批量评测 · 题干 900 tok</text>

  <path d="M150 112 L110 142" stroke="#3b82f6" stroke-width="1.2"/>
  <path d="M210 112 L250 142" stroke="#3b82f6" stroke-width="1.2"/>
  <path d="M340 112 L340 142" stroke="#3b82f6" stroke-width="1.2"/>
  <path d="M470 112 L430 142" stroke="#3b82f6" stroke-width="1.2"/>
  <path d="M504 112 L504 142" stroke="#3b82f6" stroke-width="1.2"/>
  <path d="M538 112 L578 142" stroke="#3b82f6" stroke-width="1.2"/>

  <g font-size="10" text-anchor="middle">
    <rect x="60" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="110" y="162" fill="#78350f">第 5 轮提问</text>
    <rect x="200" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="250" y="162" fill="#78350f">重试分支</text>
    <rect x="290" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="340" y="162" fill="#78350f">第 3 轮提问</text>
    <rect x="380" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="430" y="162" fill="#78350f">选项 A</text>
    <rect x="454" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="504" y="162" fill="#78350f">选项 B</text>
    <rect x="528" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="578" y="162" fill="#78350f">选项 C</text>
  </g>

  <rect x="60" y="192" width="270" height="60" rx="6" fill="#fef2f2" stroke="#fca5a5"/>
  <text x="195" y="212" font-size="11" font-weight="700" fill="#b91c1c" text-anchor="middle">无前缀复用（每条独立 prefill）</text>
  <text x="195" y="230" font-size="10" fill="#7f1d1d" text-anchor="middle">6 条请求 × (1800 + 历史 + 提问)</text>
  <text x="195" y="245" font-size="10" fill="#7f1d1d" text-anchor="middle">≈ 16,000 token 的重复计算</text>

  <rect x="350" y="192" width="270" height="60" rx="6" fill="#ecfdf5" stroke="#6ee7b7"/>
  <text x="485" y="212" font-size="11" font-weight="700" fill="#047857" text-anchor="middle">RadixAttention（树上共享）</text>
  <text x="485" y="230" font-size="10" fill="#065f46" text-anchor="middle">共享段只算一次，只 prefill 叶子增量</text>
  <text x="485" y="245" font-size="10" fill="#065f46" text-anchor="middle">≈ 2,000 token · TTFT 与显存双降</text>

  <text x="16" y="276" font-size="11" fill="#6b7280">图 1：基数树把重复前缀折叠成共享路径。共享前缀越长、分支越多，收益越大——这正是 Agent 与批量评测的典型形态。</text>
</svg>
</div>

### 2.2 收益边界

<div class="keybox">
RadixAttention 的收益<strong>完全取决于工作负载的前缀共享率</strong>。多轮 Agent、批量评测、思维树这类场景可以省掉 70%+ 的 prefill；但如果你的请求两两之间毫无共同前缀（比如纯文档摘要，每篇文章都不一样），它带来的只是一点点树维护开销。<strong>选型时先看你的流量长什么样。</strong>
</div>

**2026 年 7 月 25 日发布的 SGLang v0.5.16** 把这套机制推到了新版本：**UnifiedRadixTree 成为默认前缀缓存**——统一了此前分散的多种缓存路径（普通前缀缓存、分层缓存、PD 场景下的缓存），维护和命中逻辑收敛到一棵树上。

## 3. 结构化输出：压缩状态机

第二个差异化能力是**约束解码**。

当你要求模型输出严格的 JSON、或匹配某个正则、或遵循 EBNF 语法时，朴素做法是每生成一个 token 就检查一次约束、把非法 token 的 logits 置为 -inf。这个检查在 CPU 上做，**又是一个每步同步点**。

SGLang 的做法是**压缩有限状态机（compressed FSM）**：

- 提前把语法编译成状态机；
- 状态机上那些**只有唯一合法后继**的路径（比如 JSON 里 `{"name"` 后面必然跟 `:`），可以**一次性直接输出多个 token**，不用逐个走模型；
- 掩码计算尽量放在 GPU 上、与前向重叠。

对于返回固定 schema 的 Agent 工具调用，这个优化的实际效果非常明显——很多"结构性字符"是白送的。

<div class="warnbox">
这块也是 bug 高发区。<strong>SGLang PR #30747</strong> 修的就是 <strong>PP（流水线并行）+ 结构化输出同时开启时的崩溃</strong>（Issue #28424）。根因是调度逻辑和约束校验逻辑<strong>并发抢状态</strong>；修复方式是把约束校验与 micro-batch 边界对齐。<br/>
vLLM 侧同期也在补：<strong>0.26 让 grammar 失败不再让整个引擎崩溃</strong>，改为单请求报错。
</div>

## 4. 零开销 Spec V2：投机解码的调度革命

这是 SGLang 在 2026 年中最重要的性能工作，**v0.5.15（7/10）起成为默认**，端到端吞吐 **+约 11% TPS**。

### 4.1 传统投机解码的隐藏成本

投机解码本身的原理（草稿模型起草 → 大模型一次并行验证）可以看[推测解码手记](/LLM-blog/blog/ep1-speculative-decoding)。这里说的是**工程实现上的浪费**。

传统实现每一步都要：

```
GPU: 草稿模型生成 4 个候选
GPU: 目标模型并行验证
GPU → CPU (D2H): 把"接受了几个 token"拷回来      ← 同步点！
CPU: 根据接受数决定下一步的序列长度、KV 布局
CPU → GPU (H2D): 把新的元数据拷回去               ← 同步点！
```

那两次拷贝之间，**GPU 完全空闲**。而且因为接受数是运行时才知道的动态值，整个流程**无法被 CUDA Graph 捕获**——每步都得重新启动一堆小 kernel。

### 4.2 SGLang 的解法

<div class="fig">
<svg viewBox="0 0 680 230" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#b91c1c">传统投机解码：每步两次同步，无法整图捕获</text>
  <rect x="16" y="26" width="80" height="26" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="56" y="43" font-size="10" fill="#065f46" text-anchor="middle">draft ×4</text>
  <rect x="100" y="26" width="80" height="26" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="140" y="43" font-size="10" fill="#065f46" text-anchor="middle">verify</text>
  <rect x="184" y="26" width="50" height="26" rx="3" fill="#fecaca" stroke="#ef4444"/><text x="209" y="43" font-size="10" fill="#7f1d1d" text-anchor="middle">D2H</text>
  <rect x="238" y="26" width="86" height="26" rx="3" fill="#fed7aa" stroke="#f97316"/><text x="281" y="43" font-size="10" fill="#7c2d12" text-anchor="middle">CPU 定长度/KV</text>
  <rect x="328" y="26" width="50" height="26" rx="3" fill="#fecaca" stroke="#ef4444"/><text x="353" y="43" font-size="10" fill="#7f1d1d" text-anchor="middle">H2D</text>
  <rect x="382" y="26" width="80" height="26" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="422" y="43" font-size="10" fill="#065f46" text-anchor="middle">draft ×4</text>
  <text x="470" y="43" font-size="10" fill="#6b7280">…</text>
  <rect x="184" y="56" width="194" height="6" fill="#fca5a5"/>
  <text x="184" y="76" font-size="10" fill="#b91c1c">GPU 空转气泡（每一步都有）</text>

  <line x1="16" y1="92" x2="664" y2="92" stroke="#e5e7eb"/>

  <text x="16" y="114" font-size="12" font-weight="700" fill="#047857">Spec V2：draft-extend 可被 CUDA Graph 捕获，分支逻辑上 GPU</text>
  <rect x="16" y="122" width="200" height="26" rx="3" fill="#86efac" stroke="#10b981"/>
  <text x="116" y="139" font-size="10" fill="#065f46" text-anchor="middle">整步图：draft + verify + 元数据更新</text>
  <rect x="220" y="122" width="200" height="26" rx="3" fill="#86efac" stroke="#10b981"/>
  <text x="320" y="139" font-size="10" fill="#065f46" text-anchor="middle">整步图（下一步）</text>
  <rect x="424" y="122" width="200" height="26" rx="3" fill="#86efac" stroke="#10b981"/>
  <text x="524" y="139" font-size="10" fill="#065f46" text-anchor="middle">整步图（再下一步）</text>
  <text x="16" y="166" font-size="10" fill="#047857">CPU 提前一整步装配下一步 → GPU 无气泡，端到端 +约 11% TPS</text>

  <rect x="16" y="180" width="648" height="34" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="28" y="194" font-size="10" fill="#1e3a8a">同脉络的后续 PR：#31468 DFlash 去掉每步 host 同步（CPU 领先一整步）· #31487 减少 prefill CUDA graph 填充</text>
  <text x="28" y="208" font-size="10" fill="#1e3a8a">#31986 DSpark 稠密草稿逐层 ctx KV 投影堆叠成单个大 GEMM · #31985 经 forward_embed 把草稿 embedding 折进草稿图</text>
</svg>
</div>

三个关键动作：

1. **把 draft-extend 做成 CUDA-graph 可捕获**：用固定上界的张量形状 + GPU 上的掩码，替代运行时可变长度；
2. **砍掉 D2H / H2D**：接受数留在 GPU 上，分支逻辑改写成 GPU 可执行的形式；
3. **融合元数据计算**：page table、序列长度这些元数据的更新也进图。

节省的**不是算力，而是加速器空转的时间**——所以**低并发、时延敏感的 Agent 场景收益最大**。

### 4.3 IndexShare MTP：长上下文的草稿降本

配合 GLM-5.2 的优化里还有一个亮点：**IndexShare MTP**。

MTP（Multi-Token Prediction）是模型自带草稿头的投机解码。在 DSA（稀疏注意力）模型上，草稿步骤本来也要自己算一遍 top-k 稀疏索引。IndexShare 让草稿步**直接复用目标模型已经算好的 top-k 索引**，长上下文场景下草稿成本降低约 **1.9×**。

再加上 **TopK-V2（Lightning-TopK）**——用"选择"算法替代"完整排序"，专门优化 80k 级超长输入序列。

三者叠加的实测结果：**GLM-5.2 NVFP4 在 Blackwell 上跑到 500+ tok/s/user**（lmsys 官方博客，7/13）。

## 5. 其他值得知道的能力

| 特性 | 说明 | 版本 |
|---|---|---|
| **Breakable CUDA Graph** | 图可被中断，兼顾图收益与调度灵活性；DP attention 下默认开 breakable prefill graph（#31682） | v0.5.15 起 |
| **MLA context parallel decoding** | MLA 架构（DeepSeek 系）的上下文并行解码 | v0.5.15 |
| **FlashInfer all-to-all MoE routing** | MoE 路由走 FlashInfer 的 all-to-all | v0.5.15 |
| **HPC-Ops attention 后端** | 扩充可选注意力算子矩阵（#30540） | 07-22 main |
| **原生 web search** | 内置检索工具调用 | v0.5.15 |
| **VLM 跨请求 ViT 编码批处理** | 多模态并发时把视觉编码批起来（#24013） | 07-18 main |
| **大规模 EP（专家并行）** | MoE 模型的专家并行部署 | 持续演进 |

<div class="warnbox">
<strong>安全提醒：</strong>SGLang 0.5.5–0.5.12 存在多模态路径穿越漏洞（GHSA-qwrp-wghp-94q2），0.5.15 及以后已修复。仍在跑旧版本的务必升级。
</div>

## 6. 两家的性格差异

跟踪了这么多次发版，能看出很清楚的性格：

| | vLLM | SGLang |
|---|---|---|
| **起点** | 显存管理 | 程序结构 / 前缀复用 |
| **优势** | 模型 + 硬件 + 量化的**广度** | 共享前缀、结构化输出、前沿吞吐的**深度** |
| **发版风格** | 稳，大版本换代后密集修 bug | 激进，主分支前沿优化落地极快 |
| **典型场景** | 通用生产底座、多硬件、多模型 | Agent / 多轮 / 批量评测 / 大规模 EP |
| **量化激进度** | 全格式覆盖 | NVFP4 / MXFP4 方向扩张更猛 |

一个很典型的观察：在 07-22 那个窗口，**SGLang 主分支全在做投机解码性能打磨**（#31986/#31985），而 **vLLM 全在修稳定性**（#48524/#49302/#48843/#49306）——同一个主题的不同相位。

## 7. 小结

- SGLang 的内核是**"LLM 应用是有结构的程序"**这个洞察，由此长出 RadixAttention 与压缩状态机约束解码；
- **零开销 Spec V2** 把投机解码的调度开销压到接近零，是 v0.5.15 的核心卖点（+11% TPS）；
- **UnifiedRadixTree**（v0.5.16）统一了前缀缓存路径；
- 收益强依赖工作负载：**前缀共享率高的场景，SGLang 优势明显**。

下一篇讲两家**共同的前沿战场**：为什么"消灭同步停顿"会成为 2026 年中的主线，以及 PD 分离这个正在成型的新范式。
