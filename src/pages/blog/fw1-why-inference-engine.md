---
title: （一）为什么需要 vLLM / SGLang：朴素推理到底差在哪
description: 从一段最朴素的 HF generate 代码出发，讲清显存碎片、静态批、GPU 空转三大瓶颈，以及推理引擎存在的意义与核心指标。
pubDate: 2026-08-01
series: vLLM 与 SGLang 框架解码手记
lang: zh
altLang: en
altHref: /en/blog/fw1-why-inference-engine
layout: ../../layouts/BlogPost.astro
---

## 0. 这个系列讲什么

这个系列把过去一段时间每天跟踪 vLLM / SGLang 社区的内容，重新整理成一条循序渐进的主线：

1. **为什么要用这两个框架**（本篇）
2. **vLLM 的原理与结构**：PagedAttention、Continuous Batching、V1 架构、Model Runner V2
3. **SGLang 的原理与结构**：RadixAttention、结构化输出、零开销 Spec V2
4. **两家共同的前沿战场**：投机解码、消灭同步停顿、PD 分离、低比特量化
5. **版本演进时间线**：0.25 / 0.5.15 那一轮换代到底改了什么
6. **支持的模型与选型指南**：什么场景该选谁

不假设你读过前面的日报，但每一篇都会把当时日报里的关键事实（版本号、PR 编号、性能数字）落到对应的原理位置上。

## 1. 先看一段"能跑但不能用"的代码

几乎所有人第一次跑大模型推理，写的都是这样一段：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="cuda")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

inputs = tok("解释一下什么是推测解码", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=512)
print(tok.decode(out[0]))
```

这段代码**功能上完全正确**。单人单请求，它能给你正确答案。

但只要你把它包成一个 HTTP 服务，让 50 个人同时用，它会以肉眼可见的速度垮掉：显存爆掉、延迟飙到几十秒、GPU 利用率却只有百分之十几。

问题不在模型，在**服务方式**。要理解 vLLM 和 SGLang 存在的意义，得先看清朴素方式到底浪费在哪。

## 2. 三个致命瓶颈

### 2.1 KV Cache 显存碎片：预留式分配的浪费

自回归生成时，每生成一个 token 都要用到之前所有 token 的 Key / Value 向量。为了避免重复计算，这些向量被缓存下来，就是 **KV Cache**。

KV Cache 有多大？粗略公式：

```
KV 显存 = 2(K和V) × 层数 × KV头数 × 头维度 × 序列长度 × batch × dtype字节数
```

以一个 32 层、8 个 KV 头（GQA）、头维 128、FP16 的 8B 模型为例，**每个 token 大约占 128 KB**。一条 8K 上下文的请求就是 1 GB 左右。跑 40 条并发，光 KV 就要 40 GB。

朴素实现的做法是：**按 `max_new_tokens` 预留一块连续显存**。你设 `max_new_tokens=2048`，它就先占满 2048 个 token 的空间——哪怕这条请求实际只生成了 30 个 token 就遇到了停止符。

于是显存里出现了大量"占着但没用"的空洞。业界测下来，朴素方式的 KV 显存**有效利用率常常只有 20%–40%**。剩下的 60% 以上，是纯粹的浪费。

<div class="fig">
<svg viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="20" font-size="13" font-weight="700" fill="#b91c1c">朴素方式：连续预留</text>
  <rect x="16" y="32" width="290" height="30" rx="4" fill="#fef2f2" stroke="#ef4444"/>
  <rect x="16" y="32" width="70" height="30" rx="4" fill="#fca5a5" stroke="#ef4444"/>
  <text x="30" y="52" font-size="11" fill="#7f1d1d">实用 30 tok</text>
  <text x="150" y="52" font-size="11" fill="#b91c1c">浪费（预留 2048 却没用到）</text>

  <rect x="16" y="70" width="290" height="30" rx="4" fill="#fef2f2" stroke="#ef4444"/>
  <rect x="16" y="70" width="120" height="30" rx="4" fill="#fca5a5" stroke="#ef4444"/>
  <text x="34" y="90" font-size="11" fill="#7f1d1d">实用 400 tok</text>
  <text x="180" y="90" font-size="11" fill="#b91c1c">浪费</text>

  <rect x="16" y="108" width="290" height="30" rx="4" fill="#f3f4f6" stroke="#9ca3af" stroke-dasharray="4 3"/>
  <text x="70" y="128" font-size="11" fill="#6b7280">第 3 条请求：显存不足，排队等待</text>
  <text x="16" y="162" font-size="12" fill="#b91c1c">有效利用率 ≈ 20%–40%</text>

  <line x1="336" y1="20" x2="336" y2="200" stroke="#e5e7eb" stroke-width="1"/>

  <text x="360" y="20" font-size="13" font-weight="700" fill="#047857">分页方式：按需分块</text>
  <g fill="#a7f3d0" stroke="#10b981">
    <rect x="360" y="32" width="28" height="30" rx="3"/>
    <rect x="392" y="32" width="28" height="30" rx="3"/>
  </g>
  <text x="430" y="52" font-size="11" fill="#047857">请求 A：只占 2 块</text>

  <g fill="#a7f3d0" stroke="#10b981">
    <rect x="360" y="70" width="28" height="30" rx="3"/>
    <rect x="392" y="70" width="28" height="30" rx="3"/>
    <rect x="424" y="70" width="28" height="30" rx="3"/>
    <rect x="456" y="70" width="28" height="30" rx="3"/>
    <rect x="488" y="70" width="28" height="30" rx="3"/>
  </g>
  <text x="526" y="90" font-size="11" fill="#047857">请求 B：占 5 块</text>

  <g fill="#bfdbfe" stroke="#3b82f6">
    <rect x="360" y="108" width="28" height="30" rx="3"/>
    <rect x="392" y="108" width="28" height="30" rx="3"/>
    <rect x="424" y="108" width="28" height="30" rx="3"/>
  </g>
  <text x="466" y="128" font-size="11" fill="#1d4ed8">请求 C：立刻能进来</text>
  <text x="360" y="162" font-size="12" fill="#047857">有效利用率 &gt; 90%，并发数翻数倍</text>

  <text x="16" y="196" font-size="11" fill="#6b7280">图 1：KV Cache 连续预留 vs 分页按需分配。右侧每个小方块是一个固定大小的 block（如 16 个 token）。</text>
</svg>
</div>

### 2.2 静态批处理：队头阻塞

第二个问题是**批（batch）的组织方式**。

朴素服务通常做**静态批（static batching）**：攒够 N 条请求 → 一起送进模型 → **等最长的那条跑完** → 整批返回 → 再攒下一批。

问题很直接：同一批里有的请求生成 20 个 token 就结束了，有的要生成 2000 个。短的那些跑完之后**并不会释放槽位**，它们的位置一直空转到最长那条结束。

这就是典型的**队头阻塞（head-of-line blocking）**。实际效果是：批越大，平均延迟越糟；而且新来的请求必须等整批结束才能进入。

### 2.3 GPU 空等：真正的隐形杀手

前两个问题至少是"显性"的。第三个更隐蔽：**GPU 在等 CPU**。

一次 decode step 里，GPU 真正做矩阵乘法的时间可能只有几百微秒，但 CPU 侧要做的事情不少：调度下一批请求、准备输入张量、拷贝元数据、决定采样结果、判断是否结束……这些都在 host（CPU）上，而且经常需要把结果从 GPU 拷回 CPU（D2H）再拷回去（H2D）。

每一次这样的同步，GPU 都在**干等**。单步几百微秒的计算，配上几百微秒的启动和同步开销——**一半的时间 GPU 在空转**。

<div class="keybox">
<strong>一句话总结：</strong>朴素推理的三大浪费是「显存空洞」「槽位空转」「GPU 空等」。推理引擎做的事，本质上就是把这三种"空"填满。
</div>

## 3. 推理引擎登场：它们各自解决了什么

<div class="fig">
<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <rect x="20" y="16" width="200" height="200" rx="10" fill="#fef2f2" stroke="#fca5a5"/>
  <text x="120" y="40" font-size="14" font-weight="700" fill="#b91c1c" text-anchor="middle">朴素 HF generate</text>
  <text x="36" y="68" font-size="12" fill="#7f1d1d">✗ KV 连续预留 → 碎片</text>
  <text x="36" y="94" font-size="12" fill="#7f1d1d">✗ 静态批 → 队头阻塞</text>
  <text x="36" y="120" font-size="12" fill="#7f1d1d">✗ 每步同步 → GPU 空等</text>
  <text x="36" y="146" font-size="12" fill="#7f1d1d">✗ 无前缀复用</text>
  <text x="36" y="172" font-size="12" fill="#7f1d1d">✗ 无量化/并行编排</text>
  <text x="120" y="200" font-size="12" font-weight="700" fill="#b91c1c" text-anchor="middle">利用率 ~15%</text>

  <path d="M232 116 L268 116" stroke="#6b7280" stroke-width="2" marker-end="url(#ar1)"/>
  <defs><marker id="ar1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6b7280"/></marker></defs>

  <rect x="280" y="16" width="180" height="200" rx="10" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="370" y="40" font-size="14" font-weight="700" fill="#1d4ed8" text-anchor="middle">vLLM</text>
  <text x="294" y="68" font-size="12" fill="#1e3a8a">✓ 分页 KV / 块表</text>
  <text x="294" y="94" font-size="12" fill="#1e3a8a">✓ 连续批处理</text>
  <text x="294" y="120" font-size="12" fill="#1e3a8a">✓ MRv2 零同步</text>
  <text x="294" y="146" font-size="12" fill="#1e3a8a">✓ 硬件/模型广度最大</text>
  <text x="294" y="172" font-size="12" fill="#1e3a8a">✓ 全 CUDA Graph</text>
  <text x="370" y="200" font-size="12" font-weight="700" fill="#1d4ed8" text-anchor="middle">通用生产底座</text>

  <rect x="476" y="16" width="184" height="200" rx="10" fill="#ecfdf5" stroke="#6ee7b7"/>
  <text x="568" y="40" font-size="14" font-weight="700" fill="#047857" text-anchor="middle">SGLang</text>
  <text x="490" y="68" font-size="12" fill="#065f46">✓ RadixAttention 前缀树</text>
  <text x="490" y="94" font-size="12" fill="#065f46">✓ 结构化输出快</text>
  <text x="490" y="120" font-size="12" fill="#065f46">✓ 零开销 Spec V2</text>
  <text x="490" y="146" font-size="12" fill="#065f46">✓ 大规模 EP / PD</text>
  <text x="490" y="172" font-size="12" fill="#065f46">✓ 前沿优化落地快</text>
  <text x="568" y="200" font-size="12" font-weight="700" fill="#047857" text-anchor="middle">高共享 / Agent 场景</text>

  <text x="20" y="238" font-size="11" fill="#6b7280">图 2：两个引擎的定位差异——vLLM 求"广"（模型与硬件覆盖），SGLang 求"专"（共享前缀与前沿吞吐优化）。</text>
</svg>
</div>

**vLLM** 的起点是显存：用操作系统虚拟内存的思路管理 KV Cache（分页 + 块表），配合连续批处理，把吞吐提上去。之后它一路往"通用生产底座"走——模型覆盖最全、硬件后端最多（CUDA / ROCm / XPU / TPU）、量化格式最杂。

**SGLang** 的起点是"程序结构"：它注意到真实的 LLM 应用（Agent、多轮对话、批量评测、思维树）里，**大量请求共享相同前缀**。于是用基数树（Radix Tree）把前缀 KV 组织起来跨请求复用，这就是 RadixAttention。此外它在结构化输出（JSON Schema / 正则约束）和前沿吞吐优化上推进得非常激进。

## 4. 衡量标准：你到底该优化哪个数字

选框架、调参数之前，先明确指标。推理服务只有三个真正重要的数字：

| 指标 | 含义 | 谁在乎 |
|---|---|---|
| **TTFT**（Time To First Token） | 从请求到吐出第一个字的时间，主要由 prefill 决定 | 聊天体验、Agent 首响 |
| **TPOT / ITL**（Time Per Output Token） | 后续每个 token 的间隔，由 decode 决定 | 流式阅读的"顺滑度" |
| **Throughput** | 整机每秒处理的总 token 数 | 成本（每百万 token 多少钱） |

这三者**互相拉扯**。批开得越大，吞吐越高、单请求的 TPOT 越差；PD 分离能让 TTFT 大降，但要额外传输 KV。所有的框架设计取舍，本质都在这个三角里选位置。

<div class="warnbox">
<strong>常见误区：</strong>只看"吞吐 tok/s"这一个数字选型。离线批量任务确实该看总吞吐，但线上 Agent 场景真正卡人的是 TTFT 和 P99 TPOT——这两个指标下，中小 batch 的表现才是关键，而这恰好是 2026 年这轮架构换代（vLLM MRv2 / SGLang Spec V2）收益最明显的区间。
</div>

## 5. 两个概念：Prefill 与 Decode

后面几篇会反复出现这两个词，这里先说清楚。一次生成请求分成两个阶段，它们的计算特征**完全相反**：

| | Prefill（预填充） | Decode（解码） |
|---|---|---|
| 做什么 | 一次性处理整个 prompt，算出全部 KV | 每次只处理 1 个新 token |
| 并行度 | 高（几千 token 同时算） | 极低（一次一个） |
| 瓶颈 | **计算受限**（compute-bound） | **访存受限**（memory-bound） |
| 决定 | TTFT | TPOT |

理解这个差异，你就能理解后面所有的优化：

- **为什么 decode 要靠大 batch？** 因为它访存受限，权重从显存搬一次可以服务很多请求，batch 越大摊得越薄。
- **为什么会有投机解码？** 因为 decode 一次只出一个 token，算力大量闲置，那就"顺带"多验证几个候选。
- **为什么要 PD 分离？** 因为两个阶段特征相反，混在一张卡上互相干扰——prefill 一来就把 decode 的延迟拖长。分开部署，各自用最合适的并行策略和硬件。

## 6. 小结

| 朴素方式的问题 | 引擎的解法 | 出自 |
|---|---|---|
| KV 显存碎片 | 分页 KV Cache + 块表 | vLLM PagedAttention |
| 队头阻塞 | 连续批处理（迭代级调度） | 两家都有 |
| 重复算相同前缀 | 基数树前缀复用 | SGLang RadixAttention |
| GPU 等 CPU | 零同步模型运行器 / CUDA Graph 全捕获 | vLLM MRv2 / SGLang Spec V2 |
| decode 算力闲置 | 投机解码（MTP / EAGLE / DFlash / DSpark） | 两家都在推 |
| P/D 互相干扰 | Prefill-Decode 分离 | Dynamo / Mooncake / NIXL |

下一篇进入 vLLM 内部：从 PagedAttention 的虚拟内存类比开始，一路讲到它在 0.25 版本里**把 PagedAttention 删掉**这件看似矛盾的事。

> 与本站其他系列的联系：这里说的"decode 算力闲置"，正是[推测解码手记](/LLM-blog/blog/ep1-speculative-decoding)整个系列的出发点；而模型侧的 MoE、滑窗注意力等结构设计，可以参考[多模态解码手记](/LLM-blog/blog/mm5-efficiency-frontier)。
