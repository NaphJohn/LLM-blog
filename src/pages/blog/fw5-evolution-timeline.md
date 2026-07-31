---
title: （五）版本演进时间线：2026 年 7 月的两轮换代
description: 从 vLLM 0.25.0 到 0.26.0、SGLang 0.5.15 到 0.5.16，按时间还原这轮架构换代的每一步，以及升级时踩过的坑。
pubDate: 2026-08-01
series: vLLM 与 SGLang 框架解码手记
lang: zh
altLang: en
altHref: /en/blog/fw5-evolution-timeline
layout: ../../layouts/BlogPost.astro
---

## 1. 全景时间线

2026 年 7 月是这两个引擎近年变化最密集的一个月：**两家几乎同步完成了一轮架构换代，又几乎同日发布了下一个版本**。

<div class="fig">
<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <line x1="60" y1="40" x2="60" y2="316" stroke="#d1d5db" stroke-width="2"/>

  <g font-size="11" fill="#6b7280" text-anchor="end">
    <text x="50" y="60">7/10</text>
    <text x="50" y="100">7/11-12</text>
    <text x="50" y="140">7/13-14</text>
    <text x="50" y="180">7/18-19</text>
    <text x="50" y="220">7/21-23</text>
    <text x="50" y="260">7/25</text>
    <text x="50" y="300">7/29-31</text>
  </g>

  <g>
    <circle cx="60" cy="56" r="5" fill="#10b981"/>
    <rect x="76" y="42" width="270" height="26" rx="5" fill="#ecfdf5" stroke="#6ee7b7"/>
    <text x="88" y="59" font-size="11" fill="#047857"><tspan font-weight="700">SGLang v0.5.15</tspan> · Spec V2 默认 +11% TPS</text>
  </g>

  <g>
    <circle cx="60" cy="96" r="5" fill="#3b82f6"/>
    <rect x="76" y="82" width="330" height="26" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="88" y="99" font-size="11" fill="#1d4ed8"><tspan font-weight="700">vLLM v0.25.0</tspan> · MRv2 默认 + 删 PagedAttention</text>
  </g>

  <g>
    <circle cx="60" cy="136" r="5" fill="#f59e0b"/>
    <rect x="76" y="122" width="470" height="26" rx="5" fill="#fffbeb" stroke="#fcd34d"/>
    <text x="88" y="139" font-size="11" fill="#92400e"><tspan font-weight="700">vLLM v0.25.1</tspan>（修 NVFP4 乱码）· <tspan font-weight="700">SGLang v0.5.15.post1</tspan> · GLM-5.2 500 TPS 博客</text>
  </g>

  <g>
    <circle cx="60" cy="176" r="5" fill="#10b981"/>
    <rect x="76" y="162" width="420" height="26" rx="5" fill="#ecfdf5" stroke="#6ee7b7"/>
    <text x="88" y="179" font-size="11" fill="#047857">SGLang main：#31468 DFlash 去 host 同步 · #31487 减 prefill 图填充</text>
  </g>

  <g>
    <circle cx="60" cy="216" r="5" fill="#8b5cf6"/>
    <rect x="76" y="202" width="500" height="26" rx="5" fill="#faf5ff" stroke="#c4b5fd"/>
    <text x="88" y="219" font-size="11" fill="#6d28d9">分头行动：SGLang 打磨 DSpark(#31986/#31985) · vLLM 密集修稳定性(#48524/#49302…)</text>
  </g>

  <g>
    <circle cx="60" cy="256" r="6" fill="#ef4444"/>
    <rect x="76" y="242" width="470" height="26" rx="5" fill="#fef2f2" stroke="#fca5a5"/>
    <text x="88" y="259" font-size="11" fill="#b91c1c"><tspan font-weight="700">同日发版：vLLM 0.26.0 + SGLang 0.5.16</tspan> · 同步停顿之战落幕</text>
  </g>

  <g>
    <circle cx="60" cy="296" r="5" fill="#6b7280"/>
    <rect x="76" y="282" width="450" height="26" rx="5" fill="#f9fafb" stroke="#d1d5db"/>
    <text x="88" y="299" font-size="11" fill="#374151">主轴转向：生产部署稳定性 · GLM-5.2 NVFP4+MTP+P/D 落地</text>
  </g>

  <text x="60" y="332" font-size="11" fill="#6b7280">图 1：两条产品线在 7 月的节奏几乎完全同步，说明它们面对的是同一批瓶颈。</text>
</svg>
</div>

## 2. vLLM 演进详解

### 2.1 v0.25.0（7/11–7/12）：架构换代

这是**跨度最大的一次升级**，四件大事：

| 变更 | PR | 含义 |
|---|---|---|
| **Model Runner V2 成所有稠密模型默认** | #39337 | async-first、零 CPU-GPU 同步、step N 与 N+1 重叠 |
| **移除 PagedAttention** | #47361 | 抽象下沉到注意力后端 kernel，分页机制本身保留 |
| **Transformers v4 弃用** | #40389 | 必须迁移到 v5 |
| **Transformers backend parity** | — | HF 有实现的新架构 Day-0 全速服务 |

配套还有：统一流式解析引擎（#46610）、DeepSeek V4 Pro 的 DSpark 投机解码原生支持（8×B300 约 250 tok/s，比 MTP 高 12–42%）、异构词表通用投机（#38174）、thinking-budget 感知投机（#34668）、编译要求提到 C++20。

<div class="keybox">
<strong>性能特征：</strong>MRv2 的收益在<strong>中小 batch</strong>最明显——也就是真实 Agent 流量所处的区间。如果你只跑大 batch 离线任务，感知会弱很多。
</div>

### 2.2 v0.25.1（7/14）：两个 commit 的必打补丁

发布后两天就出补丁，因为发现了一个**静默正确性 bug**：

- **#48330 混合 dtype 量化融合守卫**：FlashInfer 的 `allreduce + RMSNorm + static-quant` 三合一融合核，在激活为 BF16、RMSNorm 权重为 FP32 时 dtype 不一致，把 4-bit NVFP4 读成了错误位模式 → 隐状态损坏 → 输出退化成重复的 `!!!!!`。修复加 dtype 匹配守卫：不一致走安全路径，一致时保留融合。
- **#47888**：TorchCodec 缺 FFmpeg 时不再阻塞启动。

<div class="warnbox">
跑 <strong>NVFP4</strong>（Gemma4 / Qwen 系 / GLM-5.2 等）的服务，<strong>0.25.0 必须升到 0.25.1</strong>。这个 bug 不报错、不崩溃，只是悄悄输出乱码——线上很容易漏掉。
</div>

### 2.3 7/18–7/25 主分支：稳定性收尾

0.25 是大改动，随之而来的是一波边缘配置修复：

| PR | 修什么 |
|---|---|
| #48524 | DFlash 层尺寸在 `num_target_layers ≠ num_hidden_layers` 时算错 |
| #49302 | 可中断分段 CUDA Graph 下 DSA 崩溃 |
| #48843 | 全量 CUDA Graph 捕获前未设 `graph_pool_id` |
| #49306 | FA4 JIT 预热时 MLA 回退 |
| #48860 | KV Connector 延迟请求时前缀缓存指标重复计数 |
| #49292 | Qwen3-VL 在 Transformers 后端的 M-RoPE |
| #49190 | Cosmos3 Edge 视频模型 |
| #48816 | GPTQ 量化 Qwen3.5 开投机解码时 MTP 权重加载错误 |
| #42569 | FA4 在 SM100（Blackwell）新增 **FP8 KV cache** 支持 |
| #48683 | ROCm 升 AITER v0.1.16.post5 |
| #45991 | XPU 新增 DeepSeek-V4 `fuse_index_q` SYCL 路径 |
| #49244 | 移除旧 partial-prefill 参数 |
| #48914 / #49427 | FlashInfer 升 0.6.15；恢复 `dequant_cache` 越界守卫 |

<div class="warnbox">
vLLM 主分支 6 月日均合入约 <strong>64 个 commit</strong>。这个速度意味着：<strong>生产环境必须锁定 release tag，不要跟 main</strong>；升级前跑完整回归。
</div>

### 2.4 v0.26.0（7/25）

- **grammar 失败不再让引擎崩溃**：结构化输出解析失败改为单请求报错，不影响其他请求；
- **前缀缓存命中率报告**：可观测性补齐；
- **NVFP4 需要 FlashInfer**：依赖关系明确化；
- **部分启动 flag 更名**：升级需检查启动脚本。

### 2.5 7/29–7/31：转向生产部署

版本节奏放缓后，重心转到落地：

- **GLM-5.2 生产部署方案**：NVFP4 + MTP + P/D 组合。`IndexerCache` 提升 MTP 接受率，**PCP（上下文并行）把 prefill 吞吐从 20.1k 提到 27.3k**；
- **P/D 二级缓存 TieringManager**（PR #42285）；
- **异步 KV 加载前瞻修复**（PR #46694）；
- 第 27 周周报主题：PD 解耦走 ZMQ + NIXL、Rust 协议重构。

## 3. SGLang 演进详解

### 3.1 v0.5.15（7/10）：零开销 Spec V2

| 特性 | 说明 |
|---|---|
| **零开销 Spec V2 默认** | draft-extend 做成 CUDA-graph 可捕获，砍 D2H/H2D，融合元数据 → **端到端 +约 11% TPS** |
| **IndexShare MTP** | 草稿步复用目标模型的 DSA top-k 索引，长上下文草稿成本降 **~1.9×** |
| **Breakable CUDA Graph 默认** | 图可中断，兼顾图收益与调度灵活 |
| **MLA context parallel decoding** | DeepSeek 系 MLA 架构的上下文并行解码 |
| **FlashInfer all-to-all MoE routing** | MoE 路由走 FlashInfer |
| **原生 web search** | 内置检索工具 |

同时修复了 **0.5.5–0.5.12 的多模态路径穿越漏洞**（GHSA-qwrp-wghp-94q2）。

### 3.2 v0.5.15.post1（7/14）：GLM-5.2 调优

配合 lmsys 官方博客《GLM-5.2 NVFP4 500 TPS》发布：

- Spec V2 默认 **+11% TPS**
- **IndexShare MTP** 复用 top-k → 降本约 **1.9×**
- **TopK-V2（Lightning-TopK）**：用"选择"替代"完整排序"，优化 80k 级超长输入
- 结果：**Blackwell 上 GLM-5.2 NVFP4 跑到 500+ tok/s/user**

跑 GLM-5.2 的团队应锁 post1。

### 3.3 7/18–7/23 主分支：投机解码持续打磨

| PR | 内容 |
|---|---|
| #31468 | **DFlash 去除投机解码每步 host 同步**——CPU 领先 GPU 一整步（pipeline），分支逻辑改为可被 CUDA Graph 捕获 |
| #31487 | 减少 prefill CUDA graph 填充 |
| #31986 | **DSpark 稠密草稿每层 ctx KV 投影堆叠为单个大 GEMM** |
| #31985 | 经 `forward_embed` 把草稿 embedding 折叠进草稿图 |
| #31682 | DP attention 默认开 breakable prefill cuda graph |
| #31981 | DSA draft-extend 元数据内核跳过超 KV 长度的 page-table 列 |
| #30272 | SM120 上 DeepSeek V4 flashinfer_mxfp4 MoE + TP2（消费级 Blackwell） |
| #24013 | VLM 跨请求 ViT 编码批处理（多模态并发提效） |
| #31825 | ModelOpt FP4 路径支持 **NVFP4_AWQ** |
| #31762 | 修 marlin_nvfp4 的 `routed_scaling_factor` |
| #30924 | 统一 per-token-group 量化内核 |
| #30540 | 新增 **HPC-Ops attention 后端** |
| #31109 | 移除 QServe 与 FBGEMM FP8 量化（旧路径需迁 ModelOpt / compressed-tensors） |
| #32047 | 修过期 per-token-group 量化调用方；`evict_from_tree_cache` 改为只驱逐 KV 缺口 |
| #27894 / #31835 | NIXL 分离功能测试进 CI；PrefillDelayer 推迟到 KV-budget 准入后 |

<div class="keybox">
把 #31468、#31986、#31985 连起来看，是一条非常清晰的主线：<strong>把投机解码里所有"每步都要回 CPU 问一下"的地方逐个消灭，把碎 kernel 合并成大 kernel，最终让整步进 CUDA Graph。</strong>省的都是加速器空转，不是算力。
</div>

### 3.4 v0.5.16（7/25）：UnifiedRadixTree

**UnifiedRadixTree 成为默认前缀缓存**——统一此前分散的多种缓存路径（普通前缀缓存、分层缓存、PD 场景缓存），命中与淘汰逻辑收敛到一棵树。

### 3.5 7/29–7/31

- **PR #30747** 修 **PP（流水线并行）+ 结构化输出崩溃**（Issue #28424）。根因：调度与约束逻辑**并发抢状态**；修复：约束校验与 micro-batch 边界对齐。
- **PR #30440** grpc disaggregated generation。

## 4. 升级检查清单

从这一个月的踩坑里，能提炼出一份实用清单：

**vLLM 0.24.x → 0.25.x/0.26.x**

- [ ] 直接上 **0.25.1 或更高**，别停在 0.25.0（NVFP4 静默乱码）
- [ ] transformers 升到 **v5**
- [ ] 编译环境支持 **C++20**
- [ ] 检查是否用了自定义算子（可能回退 MRv1，拿不到收益）
- [ ] 检查 partial-prefill 相关旧参数是否被移除
- [ ] 0.26 起检查启动 flag 是否更名；NVFP4 确认装了 FlashInfer
- [ ] **跑一遍输出正确性回归**，不要只看吞吐

**SGLang 0.5.x**

- [ ] 低于 **0.5.15** 的版本必升（多模态路径穿越漏洞）
- [ ] 跑 GLM-5.2 锁 **v0.5.15.post1**
- [ ] 用过 QServe / FBGEMM FP8 的迁到 ModelOpt / compressed-tensors
- [ ] 想要 DSpark / DFlash 最新优化 → 需要 main 分支，**但生产等稳定 tag**
- [ ] 同时开 PP + 结构化输出的，确认包含 #30747

## 5. 从时间线里能读出什么

1. **两家节奏高度同步**：0.25.0（7/11）vs 0.5.15（7/10），0.26.0 vs 0.5.16（同为 7/25）。说明它们面对的是**同一批瓶颈**，而不是各自造轮子。
2. **大版本之后必有一波修复潮**：0.25.0 之后一周内出补丁，随后两周密集修边缘配置。**新版本发布后等 1–2 周再上生产**是理性选择。
3. **主分支和 release 的差距很大**：SGLang 的前沿优化（DSpark GEMM 堆叠等）都在 main，稳定 tag 会滞后。要性能就跟 main 并自己压测，要稳定就锁 tag。
4. **主轴在迁移**：7 月上旬是"消灭同步停顿"，7 月下旬同日发版后，主轴转向**投机解码 + 前缀缓存 + day-one 模型支持 + 生产部署**。

下一篇是最实用的一篇：**支持哪些模型、什么场景选谁**。
