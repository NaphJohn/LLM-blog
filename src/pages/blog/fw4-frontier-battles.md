---
title: （四）共同的前沿战场：同步停顿、投机解码、PD 分离与低比特量化
description: 2026 年中两大引擎在同一条战线上收敛——消灭 GPU 空等、把 decode 算力榨干、把 prefill 和 decode 拆开部署、把权重压到 4 bit。
pubDate: 2026-08-01
series: vLLM 与 SGLang 框架解码手记
lang: zh
altLang: en
altHref: /en/blog/fw4-frontier-battles
layout: ../../layouts/BlogPost.astro
---

## 1. 一句话概括这个阶段

> **2026 年中，推理吞吐的前沿不再是"更快的 kernel"，而是"别让 GPU 空等主机喂数据"。**

现代硬件的 matmul 已经接近饱和，paging、continuous batching、tree speculative decoding 这些机制都已成熟。剩下的吞吐藏在哪里？**藏在加速器空等主机递活的那些缝隙里。**

vLLM 从**模型运行器端**（MRv2）动手，SGLang 从**投机解码调度器端**（Spec V2）动手，**两家从两端拆掉的是同一个同步卡点**。

<div class="fig">
<svg viewBox="0 0 680 240" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="340" y="20" font-size="14" font-weight="700" fill="#374151" text-anchor="middle">同一个卡点，两端夹击</text>

  <rect x="30" y="40" width="200" height="90" rx="8" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="130" y="62" font-size="13" font-weight="700" fill="#1d4ed8" text-anchor="middle">vLLM · MRv2</text>
  <text x="130" y="82" font-size="10" fill="#1e3a8a" text-anchor="middle">从模型运行器端切入</text>
  <text x="130" y="100" font-size="10" fill="#1e3a8a" text-anchor="middle">async-first · 零 CPU-GPU 同步</text>
  <text x="130" y="118" font-size="10" fill="#1e3a8a" text-anchor="middle">step N 与 N+1 重叠</text>

  <rect x="450" y="40" width="200" height="90" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="550" y="62" font-size="13" font-weight="700" fill="#047857" text-anchor="middle">SGLang · Spec V2</text>
  <text x="550" y="82" font-size="10" fill="#065f46" text-anchor="middle">从投机解码调度器端切入</text>
  <text x="550" y="100" font-size="10" fill="#065f46" text-anchor="middle">draft-extend 整图捕获</text>
  <text x="550" y="118" font-size="10" fill="#065f46" text-anchor="middle">砍 D2H / H2D 同步</text>

  <rect x="256" y="58" width="168" height="54" rx="8" fill="#fef2f2" stroke="#ef4444" stroke-width="2"/>
  <text x="340" y="80" font-size="13" font-weight="700" fill="#b91c1c" text-anchor="middle">Sync Stall</text>
  <text x="340" y="98" font-size="10" fill="#7f1d1d" text-anchor="middle">GPU 空等主机的时间</text>

  <path d="M232 85 L252 85" stroke="#3b82f6" stroke-width="2.5" marker-end="url(#ax)"/>
  <path d="M448 85 L428 85" stroke="#10b981" stroke-width="2.5" marker-end="url(#ax2)"/>
  <defs>
    <marker id="ax" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#3b82f6"/></marker>
    <marker id="ax2" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#10b981"/></marker>
  </defs>

  <text x="340" y="156" font-size="11" fill="#6b7280" text-anchor="middle">省的是加速器空转，不是算力 → 低并发、时延敏感的 Agent 场景收益最大</text>

  <rect x="30" y="172" width="620" height="48" rx="6" fill="#f9fafb" stroke="#e5e7eb"/>
  <text x="44" y="190" font-size="10" fill="#374151">vLLM 侧：MRv2 默认(#39337) · 删 PagedAttention(#47361) · 整步 CUDA Graph（300µs → 5µs）</text>
  <text x="44" y="208" font-size="10" fill="#374151">SGLang 侧：Spec V2 默认(+11% TPS) · #31468 DFlash 去每步 host 同步 · #31487 减 prefill graph 填充 · #31986/#31985 DSpark GEMM 堆叠</text>
</svg>
</div>

## 2. 投机解码：把 decode 的闲置算力榨干

### 2.1 为什么 decode 一定要投机

decode 是**访存受限**的：一次前向要把整个模型权重从 HBM 搬进 SM，却只算 1 个 token，算力单元大量闲置。

投机解码的本质就是：**既然反正要搬一次权重，那就顺便多验证几个候选 token**。搬运成本不变，产出翻几倍。

### 2.2 各家草稿方案对照

| 方案 | 草稿来源 | 特点 | 落地 |
|---|---|---|---|
| **独立小模型** | 同族的小尺寸模型 | 通用但要额外显存、词表须一致 | 两家都支持；vLLM #38174 支持**异构词表** |
| **EAGLE / EAGLE-3** | 轻量草稿头，复用目标模型隐状态 | 接受率高、开销小，最主流 | 两家一等支持 |
| **MTP**（Multi-Token Prediction） | 模型自带的多 token 预测头 | 训练时就学好，一步出 ~4 token | DeepSeek / GLM / Step 系列自带 |
| **DFlash** | 块扩散起草 + KV 注入 | 一次并行出整块草稿 | SGLang #31468 去 host 同步；vLLM #48524 修层尺寸 |
| **DSpark** | 半自回归起草 + 马尔可夫头 + 置信度调度 | 按置信度动态决定草稿长度 | vLLM 原生支持 DeepSeek V4 Pro；SGLang #31986/#31985 优化 |

<div class="keybox">
实测数字：<strong>DeepSeek V4 Pro + DSpark 在 8×B300 上约 250 tok/s，比 MTP 高 12–42%</strong>（vLLM 0.25 原生支持）。<br/>
DFlash 与 DSpark 的原理差异，可以看本站另一个系列：<a href="/LLM-blog/blog/ep5-dflash-vs-dspark">同台对比：DFlash 与 DSpark 到底差在哪</a>。
</div>

### 2.3 值得关注的细节

- **thinking-budget 感知投机**（vLLM #34668）：推理模型（带思维链）的思考阶段和回答阶段特征不同，草稿策略也该不同——这个 PR 让投机解码感知思考预算，可安全提速推理模型。
- **异构词表通用投机**（vLLM #38174）：草稿模型和目标模型词表不一样也能配对，大幅扩展了可用草稿模型的范围。
- **量化 + 投机的组合坑**：vLLM #48816 修的就是 GPTQ 量化 Qwen3.5 开投机解码时 **MTP 权重加载错误**。量化和 spec decode 一起开的团队要特别留意这类组合 bug。

## 3. PD 分离：正在成型的新范式

### 3.1 为什么要拆

第一篇讲过，prefill 和 decode 的计算特征完全相反。它们混在同一张卡上会互相伤害：

- 一个长 prompt 的 prefill 进来，会**霸占整步计算**，让所有正在流式输出的请求卡顿（TPOT 尖刺）；
- decode 想开大 batch 提吞吐，prefill 却希望尽快单独跑完降 TTFT；
- 两者最优的并行策略也不同（prefill 适合 TP + 上下文并行，decode 适合大 EP + 大 batch）。

**PD 分离（Prefill-Decode Disaggregation）**：把两个阶段部署到不同的 GPU 池，prefill 算完把 KV Cache 传给 decode 节点。

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <rect x="20" y="16" width="110" height="40" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="75" y="34" font-size="11" font-weight="700" fill="#1d4ed8" text-anchor="middle">Router</text>
  <text x="75" y="48" font-size="9" fill="#1e3a8a" text-anchor="middle">Rust / K8s / Planner</text>

  <path d="M132 36 L176 36" stroke="#6b7280" stroke-width="1.5" marker-end="url(#a4)"/>
  <defs><marker id="a4" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6b7280"/></marker></defs>

  <rect x="180" y="10" width="200" height="118" rx="8" fill="#fff7ed" stroke="#fb923c"/>
  <text x="280" y="30" font-size="12" font-weight="700" fill="#9a3412" text-anchor="middle">Prefill 池</text>
  <text x="280" y="48" font-size="10" fill="#9a3412" text-anchor="middle">计算受限 · 决定 TTFT</text>
  <g font-size="9" text-anchor="middle" fill="#7c2d12">
    <rect x="196" y="58" width="52" height="26" rx="4" fill="#fed7aa" stroke="#f97316"/><text x="222" y="75">GPU</text>
    <rect x="254" y="58" width="52" height="26" rx="4" fill="#fed7aa" stroke="#f97316"/><text x="280" y="75">GPU</text>
    <rect x="312" y="58" width="52" height="26" rx="4" fill="#fed7aa" stroke="#f97316"/><text x="338" y="75">GPU</text>
  </g>
  <text x="280" y="104" font-size="9" fill="#9a3412" text-anchor="middle">TP + 上下文并行(PCP)</text>
  <text x="280" y="118" font-size="9" fill="#9a3412" text-anchor="middle">可用较低显存、较高算力卡</text>

  <path d="M382 68 L432 68" stroke="#8b5cf6" stroke-width="2" marker-end="url(#a5)"/>
  <defs><marker id="a5" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#8b5cf6"/></marker></defs>
  <text x="407" y="60" font-size="9" fill="#6d28d9" text-anchor="middle">KV 传输</text>
  <text x="407" y="84" font-size="8" fill="#6d28d9" text-anchor="middle">NIXL / Mooncake</text>
  <text x="407" y="96" font-size="8" fill="#6d28d9" text-anchor="middle">NVLink / RDMA</text>

  <rect x="436" y="10" width="216" height="118" rx="8" fill="#f0fdf4" stroke="#4ade80"/>
  <text x="544" y="30" font-size="12" font-weight="700" fill="#166534" text-anchor="middle">Decode 池</text>
  <text x="544" y="48" font-size="10" fill="#166534" text-anchor="middle">访存受限 · 决定 TPOT</text>
  <g font-size="9" text-anchor="middle" fill="#14532d">
    <rect x="452" y="58" width="44" height="26" rx="4" fill="#bbf7d0" stroke="#22c55e"/><text x="474" y="75">GPU</text>
    <rect x="502" y="58" width="44" height="26" rx="4" fill="#bbf7d0" stroke="#22c55e"/><text x="524" y="75">GPU</text>
    <rect x="552" y="58" width="44" height="26" rx="4" fill="#bbf7d0" stroke="#22c55e"/><text x="574" y="75">GPU</text>
    <rect x="602" y="58" width="44" height="26" rx="4" fill="#bbf7d0" stroke="#22c55e"/><text x="624" y="75">GPU</text>
  </g>
  <text x="544" y="104" font-size="9" fill="#166534" text-anchor="middle">大 EP（专家并行）+ 大 batch</text>
  <text x="544" y="118" font-size="9" fill="#166534" text-anchor="middle">吃显存带宽与容量</text>

  <rect x="20" y="146" width="632" height="62" rx="8" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="336" y="166" font-size="12" font-weight="700" fill="#6d28d9" text-anchor="middle">分层 KV Cache 管理（4 层）</text>
  <g font-size="9" text-anchor="middle">
    <rect x="40" y="174" width="132" height="24" rx="4" fill="#ede9fe" stroke="#a78bfa"/><text x="106" y="190" fill="#5b21b6">GPU HBM（最快）</text>
    <rect x="184" y="174" width="132" height="24" rx="4" fill="#ede9fe" stroke="#a78bfa"/><text x="250" y="190" fill="#5b21b6">NVLink 内存池</text>
    <rect x="328" y="174" width="132" height="24" rx="4" fill="#ede9fe" stroke="#a78bfa"/><text x="394" y="190" fill="#5b21b6">Host DRAM</text>
    <rect x="472" y="174" width="160" height="24" rx="4" fill="#ede9fe" stroke="#a78bfa"/><text x="552" y="190" fill="#5b21b6">SSD / 对象存储（Mooncake）</text>
  </g>

  <text x="20" y="230" font-size="11" fill="#6b7280">图 2：PD 分离架构。两个阶段各自用最合适的并行策略与硬件，中间由 NIXL / Mooncake 传输 KV。</text>
  <text x="20" y="250" font-size="11" fill="#374151" font-weight="700">代表性数字：</text>
  <text x="20" y="268" font-size="10" fill="#374151">· NVIDIA Dynamo 1.0（含 Planner + 4 层 KV Manager）：DeepSeek-R1 671B 复合提升 30×，Llama-70B Hopper 干净 2×</text>
  <text x="20" y="284" font-size="10" fill="#374151">· DeepSeek V3 全量分离栈 ~545 tok/s/GPU（北极星指标）· Together CPD 长上下文 B200 提 35–40% · Mooncake 覆盖 75% 真实请求</text>
</svg>
</div>

### 3.2 生态玩家

| 项目 | 定位 | 特点 |
|---|---|---|
| **NVIDIA Dynamo 1.0** | 叠在 vLLM/SGLang 之上的编排层 | Planner + 4 层 KV Cache Manager；DeepSeek-R1 671B 复合 30× |
| **llm-d**（CNCF） | K8s 原生 PD 编排 | 云原生栈的标准路径 |
| **vLLM V1 Rust router** | 引擎自带路由 | 比 llm-d **高 25% RPS** 且无 K8s 依赖 |
| **SGLang PD router** | 引擎自带，多路由策略 | prefill 失败自动取消配对 decode |
| **Mooncake** | KV 存储与传输后端 | 多级缓存，覆盖 75% 真实请求 |
| **NIXL** | 统一传输层 | 两家都在收敛到它 |

### 3.3 几个有意思的变体

- **PPD（ICML 2026）**：对经典 PD 的批判——**KV 干脆留在 decode 节点，只 append 新增 token**。多轮对话第 2 轮起 TTFT 降 **68%**，且没有 TPOT 回归。这对 Agent 场景意义很大。
- **Together CPD**：把 prefill 再拆成 Pre-Prefill + Prefill 两段，长上下文在 B200 上提升 **35–40%**。
- **vLLM TieringManager（PR #42285）**：P/D 二级缓存管理，把 KV 在 HBM / DRAM / SSD 之间分层。
- **SGLang PrefillDelayer（#31835）**：把 prefill 协商推迟到 KV-budget 准入之后，避免算完发现 decode 侧放不下。

<div class="warnbox">
<strong>PD 分离不是免费午餐：</strong>它引入了 KV 网络传输（长上下文时可能是几个 GB）、跨节点调度复杂度、故障域扩大。<strong>只有在规模足够大、长 prompt 占比高、且有高速互联（NVLink / RDMA）时才划算。</strong>单机 8 卡跑中小模型，老老实实用混合部署即可。
</div>

## 4. 低比特量化：显存与带宽的直接解法

decode 是访存受限的，那**把权重压小**就是最直接的提速手段。

| 格式 | 位宽 | 说明 | 状态 |
|---|---|---|---|
| **FP8** | 8 | Hopper 起原生支持，精度损失极小 | 生产默认之一；vLLM #42569 让 FA4 在 SM100 支持 **FP8 KV cache** |
| **INT4 / AWQ / GPTQ** | 4 | 经典权重量化 | 成熟；注意 Step 系列的 Int4 权重 vLLM 暂不支持 |
| **NVFP4** | 4 | Blackwell 原生 4-bit 浮点，精度好于 INT4 | 2026 年主推；需 FlashInfer（vLLM 0.26 起） |
| **MXFP4** | 4 | 微缩放 4-bit，AMD 侧推进（SGLang #28291） | 扩张中 |
| **NVFP4_AWQ** | 4 | NVFP4 + AWQ 校准（SGLang #31825） | 前沿组合 |

**观察**：SGLang 在 FP4 方向扩张明显更激进（NVFP4_AWQ、marlin_nvfp4 修复、per-token-group 量化内核统一 #30924），vLLM 则胜在格式覆盖全。

<div class="warnbox">
量化是 bug 高发区，几个真实案例：<br/>
· vLLM <strong>#48330</strong>：混合 dtype 融合核把 NVFP4 读成错误位模式 → 静默输出 <code>!!!!!</code>（0.25.1 修）<br/>
· vLLM <strong>#48816</strong>：GPTQ + 投机解码时 MTP 权重加载错误<br/>
· SGLang <strong>#31762</strong>：marlin_nvfp4 的 routed_scaling_factor 错误<br/>
<strong>结论：量化 + 投机解码 + 新硬件三者叠加时，务必跑完整正确性回归，不要只看吞吐数字。</strong>
</div>

## 5. 一条支线：要不要脱离 transformers

这是社区里一个持续的分歧。

- **vLLM 的选择**：**双轨并行**。原生实现求性能，Transformers 后端求覆盖，而且 0.25 起让 Transformers 后端**速度追平原生**——新模型 Day-0 全速可用。代价是必须跟进 transformers v5（v4 已弃用 #40389）。
- **模型方的选择**：越来越多国产模型走 `trust_remote_code` 自研建模（如 Step 3.7 Flash，transformers 5.0+ 仅用于 debug）。
- **权衡**：自研纯 PyTorch 建模 → 可控算子、低开销、易定制 attention 与采样；代价是要自己实现 tokenizer、权重加载、并行切分。

一个观察：在 07-22 那个窗口，vLLM 侧完全没有"脱 transformers"的 PR，反而在做**双后端收敛**（#49292 让 Qwen3-VL 在 Transformers 后端也支持 M-RoPE、#47298 Ovis2_5 适配 transformers v5 special tokens）。**方向是"原生 MRv2 + transformers v5 双轨对齐"，而不是二选一。**

## 6. 小结：四条战线的关系

```
                 吞吐/延迟的四个来源
                          │
   ┌──────────┬───────────┼───────────┬──────────┐
   │          │           │           │          │
消灭同步    投机解码     PD 分离     低比特量化
   │          │           │           │
GPU 不空等  一次前向    两阶段互不   权重搬得
           出多 token   干扰、各自   更少
                        用最优并行
   │          │           │           │
 MRv2      EAGLE/MTP   Dynamo      FP8/NVFP4
 Spec V2   DFlash      NIXL         MXFP4
           DSpark      Mooncake     AWQ
```

它们**互相叠加**：MRv2 的零同步让投机解码能整图捕获；PD 分离让 decode 池能开更大 batch，而大 batch 又让投机解码的验证更划算；量化省下的显存可以拿去装更多 KV block。

下一篇把这些拼成时间线，看 2026 年 7 月那两轮版本到底改了什么、按什么顺序发生。
