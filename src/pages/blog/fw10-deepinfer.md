---
title: （十）DeepInfer 深度解析：DeepSeek 把生产级推理栈交出来之后，到底给了什么、留了什么
description: 拆解 DeepInfer 的四层架构（算子层 / 并行层 / 解码层 / 工程层）、Attention DP + Wide-EP 与 EPLB 贪心冗余复制、DSpark 置信度变长验证（5.0→3.2~4.4 个草稿 token）、SWA Bounded Replay 与 128 对齐的缓存命中规则、零开销批调度器与 DeferredRelease，以及它刻意没开源的 KV 管理 / PD 分离 / 跨实例负载均衡三块。
pubDate: 2026-10-27
series: vLLM 与 SGLang 框架解码手记
lang: zh
altLang: en
altHref: /en/blog/fw10-deepinfer
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么值得单独写一篇

2026 年 10 月 20 日，DeepSeek 开源了 DeepInfer——支撑 DeepSeek-V4.1-Flash 大规模线上服务的那套推理引擎，同日还放出了 `deepseek-recipe`（生产级 Token 编解码组件）。

这件事的份量不在于「又开源了一个框架」，而在于它开源的是**中间层**：大模型的能力边界由模型本身划定，但能不能以合理成本服务超大规模用户，取决于推理框架这一层。DeepSeek 此前已经交出了模型层（DeepSeek 系列）和算子层（FlashMLA / DeepGEMM / DeepEP / TileLang），这次补的是最后一块。

配套公开的还有两组数字，罕见地把账本翻开了：**单集群每天处理 155 万亿 Token，按谷价折算每卡每小时收入 38 美元**。注意这是收入不是利润，且 155 万亿是官方集群吞吐口径，材料没有拆出输入 / 输出 / 缓存命中的比例，因此不能据此还原真实新增计算量。

但站内此前的 DeepInfer 相关内容都散在 `aihot` 日报里（10-22 到 10-26 五期），视角是「社区一周内怎么追平它」。这一篇换个角度：**把 DeepInfer 自己的设计拆开**，看清哪些是真功夫、哪些是企业基础设施的必然耦合、哪些是别人抄不走的模型结构红利。

一句话结论先给：**DeepInfer 最值得学的不是某个 kernel，而是「模型结构与缓存系统的协同设计」——128 这个数字在模型配置、缓存块大小、重放窗口、命中规则里是同一个数，因为它们是一起被选出来的。**

## 1. 开源了什么，以及它在推理栈里的位置

先分清三层，否则很容易得出「开源不完整」这种过于粗糙的判断。

<div class="fig">
<svg viewBox="0 0 680 330" width="100%" font-size="13" font-weight="400" role="img" aria-label="DeepInfer 在推理栈中的位置">
<title>图 1：三层推理栈，DeepInfer 交付的是中间层</title>
<defs>
<marker id="a1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<rect x="60" y="40" width="560" height="60" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="340" y="62" font-size="13" fill="#047857" font-weight="600" text-anchor="middle" dominant-baseline="central">第三层 · 集群服务（部分公开设计，代码未全放）</text>
<text x="340" y="82" font-size="11" fill="#1a1a1a" text-anchor="middle" dominant-baseline="central">请求路由 · 跨实例负载均衡 · KV 存放位置 · 失效改路由 · 扩缩容</text>

<rect x="60" y="120" width="560" height="70" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.2"/>
<text x="340" y="144" font-size="13" fill="#1d4ed8" font-weight="600" text-anchor="middle" dominant-baseline="central">第二层 · DeepInfer 引擎 ← 本次开源主体</text>
<text x="340" y="164" font-size="11" fill="#1a1a1a" text-anchor="middle" dominant-baseline="central">接请求 → 组织批次 → 管在途序列与缓存块 → 调模型 → 回结果</text>
<text x="340" y="180" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">把局部算子串成一条可持续执行的链路</text>

<rect x="60" y="210" width="560" height="60" rx="8" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="340" y="232" font-size="13" fill="#1a1a1a" font-weight="600" text-anchor="middle" dominant-baseline="central">第一层 · 算子与通信（此前散装开源，本次打包调通）</text>
<text x="340" y="252" font-size="11" fill="#6b7280" text-anchor="middle" dominant-baseline="central">FlashMLA（注意力）· DeepGEMM（矩阵乘）· DeepEP（专家通信）· TileLang 算子</text>

<rect x="60" y="292" width="270" height="30" rx="6" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="195" y="307" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">开源：离线推理核心 LLM.generate()</text>
<rect x="350" y="292" width="270" height="30" rx="6" fill="#fef2f2" stroke="#ef4444" stroke-width="0.5"/>
<text x="485" y="307" font-size="11" fill="#b91c1c" text-anchor="middle" dominant-baseline="central">未开源：完整 KV 管理 / PD 分离 / 副本 LB</text>
</svg>
</div>

关键区分点在于：**「没有交付完整 KV cache 系统」不等于引擎没有缓存**。引擎内部当然有缓存块的分配与回收（否则连单实例都跑不了），缺的是跨实例的 KV 存放决策与搬运；同理，「公开了专家负载均衡（EPLB）」也不等于跨实例请求路由一并交付了。它们在不同层解决不同问题。

技术渊源官方写得很老实：DeepInfer 最初主要基于 vLLM 搭建，同时借鉴了 SGLang 与 TensorRT-LLM，入口与引擎文件仍保留派生标记。但当前实现里有 C++ 调度与资源管理、专门的模型执行路径、以及与 DeepSeek 模型结构配合的优化。**说它「只是套壳」会漏掉这些真正承担执行工作的部分，说它「与既有框架无关」也不符合源码。**

`deepseek-recipe` 则是另一半，被普遍低估：不是 tokenizer 包装，而是一组 Rust 库 + Python 绑定的 **API 兼容前门**，把 OpenAI Chat Completions / OpenAI Responses / Anthropic Messages 三种协议统一成内部 Conversation 结构，再按模型版本渲染 prompt、编码、把输出流式反解回目标格式。它还从编码阶段就区分普通文本与真正的控制 token——用户输入里字面出现的对话起止符仍按普通文字处理。模型推理、工具执行、HTTP 传输由调用方自备。

## 2. 四层架构：一张表看清它优化了什么

<div class="fig">
<svg viewBox="0 0 680 300" width="100%" font-size="13" font-weight="400" role="img" aria-label="DeepInfer 四层架构">
<title>图 2：DeepInfer 自底向上的四层设计</title>
<rect x="40" y="36" width="600" height="54" rx="8" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="60" y="56" font-size="12" fill="#1a1a1a" font-weight="600">算子层</text>
<text x="60" y="76" font-size="11" fill="#6b7280">FlashMLA · DeepGEMM · DeepEP · TileKernels · DeepSelect + 自定义 TileLang 算子</text>

<rect x="40" y="98" width="600" height="54" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="60" y="118" font-size="12" fill="#1a1a1a" font-weight="600">并行层</text>
<text x="60" y="138" font-size="11" fill="#2563eb">Attention DP / 均衡 CP + MoE EP + EPLB（专家并行负载均衡）</text>

<rect x="40" y="160" width="600" height="54" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="60" y="180" font-size="12" fill="#1a1a1a" font-weight="600">解码层</text>
<text x="60" y="200" font-size="11" fill="#047857">DSpark 推测解码（置信度调度验证）+ SWA Bounded Replay（滑动窗口有界重放）</text>

<rect x="40" y="222" width="600" height="54" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="60" y="242" font-size="12" fill="#1a1a1a" font-weight="600">工程层</text>
<text x="60" y="262" font-size="11" fill="#854f0b">异步调度（zero-overhead batch scheduler）+ 结构化输出（Regex / JSON Schema / EBNF）+ 可复现开关</text>
</svg>
</div>

这四层里，**只有解码层是真正的 DeepSeek 特色**，其余三层是把已有工程做到极致。下面按「抄得走的」和「抄不走的」分开讲。

## 3. 并行层：Prefill 要算力，Decode 要带宽，所以配比完全不同

V4.1-Flash 是 MoE 模型，内部有 384 个路由专家。两个阶段资源特征截然不同：**Prefill 一次处理数万 token，拼算力；Decode 一次只出一个 token，大部分时间在读权重和缓存，拼显存带宽。**

因此 DeepSeek 对两个阶段用了完全不同量级的并行：

| 阶段 | 并行策略 | 取值 | 理由 |
|---|---|---|---|
| Prefill | Attention DP + 均衡 CP + MoE EP | EP2–EP8 / DP2–DP8 | 输入块大，单批即可喂饱算力，无需过宽并行 |
| Decode | Attention DP + MoE Wide-EP | EP64–EP128 / DP64–DP128 | 专家摊得越开，每步 MoE 要读的显存越少，可聚成超大 batch 摊薄开销 |
| 生产实践 | P:D 接近 1:1 | — | 最小高效单元 64 卡 Prefill + 64 卡 Decode |

**Wide-EP 的实质是把专家摊到更多卡上。** 以 64 卡 Decode 为例：384 个路由专家摊开，每卡 6 个，再加 1 个热门专家副本，共 7 个。每卡专家越少，每步要从显存读的参数越少，同样多的卡就能接住更多请求。

官方离线基准（64 张 SM100，同一 scale-up 通信域，固定 3 个草稿 token、按 75% 逐 token 接受率折算）给出的分岔很有意思：

- 每用户 100 tok/s 时，EP64 约 2.66 万 tok/s/GPU，EP8 约 2.5 万——**峰值几乎一样**；
- 每用户 250 tok/s 时，EP64 仍有约 1.55 万，EP8 已掉到约 1700。

**所以 Wide-EP 买的不是峰值，而是高交互性区间的纵深。** 这一点在做容量规划时特别容易被忽略：如果你的负载是低并发长输出，EP 开大只是白烧通信。

### 3.1 EPLB：用贪心把热点专家复制出去

MoE 推理最怕专家热点——某些专家被路由爆满，其他卡空等。训练时靠 auxiliary loss 做均衡，**推理时没有 loss 可用，只能靠统计加搬运**。

EPLB 的做法是：根据历史 Token 负载，为热点逻辑专家分配冗余副本，再把物理副本安排到不同卡上。优化目标是 **minimax**——复制哪个专家能让所有 step 的最坏情况降得最多，而不是简单复制最热的那个。

| 阶段 | 每卡冗余专家起点 | 统计窗口 | 重排周期 | 实测 moe_lb |
|---|---|---|---|---|
| Prefill | 4 个 | 100 步 | 1000 步 | 1.074（EP4DP4）/ 1.045（EP2DP2） |
| Decode | 1 个 | 1000 步 | 10000 步 | **1.090** |

`moe_lb` 定义为「最慢 rank 负载 / 平均负载」，越接近 1 越均衡。**1.090 意味着最忙的那张卡只比平均水平多干 9% 的活**——在 64 卡专家并行下这已经相当紧张。

一个值得学的工程细节：EPLB 用 `tokens_per_expert` 加一个极小偏移（`arange_logical × 1e-4`）给每个专家评分，让各专家 score 严格互不相等。**这 0.0001 的扰动换来的是可复现的调度**——避免 argmax 并列时不同层返回不同结果。这种「为了确定性牺牲一点点最优性」的取舍，在生产系统里比多 1% 吞吐重要得多。

代价也很实在：EP64 意味着 64 张卡构成单一通信域，**挂一张卡整实例不可用**。恢复策略不是快速重启，而是重启时「从所有当前健康的 GPU 重新确定 EP size」，拿剩下的卡降级接着跑，安全网靠冗余容量而非恢复速度。

## 4. 解码层之一：DSpark 变长验证——生成 5 个不等于验证 5 个

站内 `ep4` 已经把 DSpark 的草稿结构（半自回归起草 + 马尔可夫头）讲透了，这里补 DeepInfer 里**调度侧的落地形态**。

推测解码的思路是先让轻量草稿网络提出多个候选 token，再由主模型并行验证。固定验证长度好实现，但不同请求、不同位置的草稿质量差异很大——**后半段候选若很可能被拒绝，仍送去验证就是纯浪费算力。**

DSpark 给草稿加了 Confidence Head，估计各位置被接受的概率。调度时结合实际硬件的吞吐曲线，决定给哪些请求验证多长的草稿前缀，目标是提高**有效输出吞吐**，而不是把验证批次堆大。

关键逻辑：**不能只给每个位置设独立门槛。** 后面位置的草稿，只有在前面的候选都被接受时才有用。所以调度器先把预测转成**前缀的累积收益**，再估计不同验证总量需要的执行时间，选「预期有效产出 / 预计耗时」较高的预算。

实测收益高度依赖并发——这是最容易误读的一组数字：

| 每 rank 并发序列 | 固定验证 5 个（tok/s） | DSpark 变长（tok/s） | 相对收益 |
|---|---|---|---|
| 16 | 4,891 | 4,816 | **-1.5%** |
| 32 | 8,964 | 9,111 | +1.6% |
| 48 | 12,032 | 12,467 | +3.6% |
| **60（生产水位）** | 13,730 | 14,466 | **+5.4%** |
| 72 | 14,962 | 15,961 | +6.7% |
| 96 | 16,817 | 18,660 | +11.0% |

**低负载时收益小甚至为负，随流量增大而增长。** 官方在生产水位（约 60 序列/rank）估算 +5.4%，整体线上生产负载 +4%~7%（官方亦表述为约 6%）。机制上是：每步需验证的草稿 token 从 5.0 降到 3.2~4.4（**降约 30%**），而平均接受长度基本不变（3.29 vs 3.20）。

> 注意这组数字来自 EP64/DP64 特定配置的官方估算，不是 DSpark 的通用加速比。另外它是按 75% 逐 token 接受率折算的**执行效率**测量，不代表真实任务上的草稿质量。

两个实现细节值得记：

1. 最大草稿长度默认 5，但**变长验证默认关闭**，需显式 `DEEPINFER_ENABLE_MTP_VARLEN=1`；没有可用性能曲线时，调度器退回验证全部草稿。
2. 异步调度下 CPU 可能还没收到最新置信信息，所以**粗粒度预算放在 CPU 侧，GPU 再用更新信息做细粒度分配**。

最后一句必须强调：**Confidence Head 只用来安排验证资源，绝不能凭它直接接受未经目标模型验证的 Token。** 这是无损性的底线。

## 5. 解码层之二：SWA Bounded Replay——本篇最值得读的部分

这块是全文技术密度最高、也最能体现「协同设计」的地方。先把名词钉死：

- **Prefill / Decode** 是两个**运行阶段**（处理输入 / 继续生成）；
- **CED 编码器—解码器** 是 V4.1-Flash 内部的**前后两段计算层**（Causal Encoder-Decoder）。

这两套叫法极易混淆。这里的「编码器」**不是**视觉编码器，「解码器」**也不等于** Decode 阶段。

### 5.1 问题：一个寿命三分钟的缓存，占了持久缓存的近一半

V4.1-Flash 除了主 KV Cache，还有独立的**滑动窗口注意力**（SWA）状态：每个解码器层都有一个只看最近 128 token 的局部窗口。处理输入时，前段（编码器）建立历史表示，后段（解码器）复用其中部分状态继续算——所以**一次 Prefill 也会经过解码器层**。

麻烦在于：解码器层的全局 K/V 可以从编码器末隐状态投影出来，**但 SWA 窗口状态不行**——它每层独立、不从任何东西投影。要精确重建一个已缓存前缀的 SWA 状态，得把最后 2560 个 token（20 层 × 128）重新跑过 20 个解码器层。

V4 时代的做法是老实持久化这 20 层每层 128 token 的状态。结果：**SWA 占掉了持久 KV Cache 近一半容量，而线上实测这块缓存 TTL 只有约 180 秒**（会话内分钟级复用，会话结束即作废）。

> 用能留几天的 SSD，去存寿命三分钟的数据——这是典型的「存储层级与数据生命周期错配」。

### 5.2 解法：不持久化，只重放一个有界后缀

<div class="fig">
<svg viewBox="0 0 680 340" width="100%" font-size="13" font-weight="400" role="img" aria-label="SWA Bounded Replay 与精确重建对比">
<title>图 3：精确重建 vs 有界重放</title>
<defs>
<marker id="a2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<text x="40" y="34" font-size="12" font-weight="600" fill="#b91c1c">V4：精确重建 = 2560 token × 20 层</text>
<rect x="40" y="46" width="520" height="26" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="0.5"/>
<text x="300" y="59" font-size="11" fill="#b91c1c" text-anchor="middle" dominant-baseline="central">要重算的历史（20 层 × 128 = 2560 token）</text>
<rect x="560" y="46" width="80" height="26" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="600" y="59" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">新 token</text>

<text x="40" y="130" font-size="12" font-weight="600" fill="#047857">V4.1：有界重放 = 只重算最近 128（默认保留 140）</text>
<rect x="40" y="142" width="520" height="26" rx="4" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="300" y="155" font-size="11" fill="#6b7280" text-anchor="middle" dominant-baseline="central">长历史仍走另一条缓存路径（不重算）</text>
<rect x="440" y="142" width="120" height="26" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1"/>
<text x="500" y="155" font-size="11" fill="#1d4ed8" text-anchor="middle" dominant-baseline="central">重放 128</text>
<rect x="560" y="142" width="80" height="26" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="600" y="155" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">新 token</text>

<rect x="40" y="196" width="600" height="52" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="60" y="216" font-size="12" fill="#854f0b" font-weight="600">重放时注意力范围截断在重放片段内</text>
<text x="60" y="236" font-size="11" fill="#1a1a1a">→ 得到的是「够用的近似」，不是逐位等价的完整历史恢复；成本变成与上下文长度无关的常数</text>

<rect x="40" y="262" width="600" height="56" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="60" y="282" font-size="12" fill="#047857" font-weight="600">收益</text>
<text x="60" y="302" font-size="11" fill="#1a1a1a">持久 KV 足迹 = V4-Flash 的 1/8（-87.5%）· 全局 KV 890 B/token = V1 的 1/437 · Decoder 侧 Prefill 计算量近乎减半</text>
</svg>
</div>

V4.1 的改法是**不持久化 + 有界重放**：SWA KV Cache 只留在产出它的那台机器内存里（TTL 约 180 秒，命中率 98%+）；一旦缺失，只重算最后 128 个 token 来重建近似状态，注意力窗口截在这 128 个里。

**这是近似，不是精确重建。** 因为层间感受野大于 128，重放结果与原路径存在细微偏差，DeepSeek 是在 post-training 阶段模拟 bounded replay 来对齐的。这一点非常重要：**你不能把这个技巧直接套到任何现成模型上**，窗口必须与训练一致，否则质量会掉。

配置上默认 `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`、`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`。**注意 140 不是把 128 的窗口改大**——保留的后缀必须覆盖 SWA 窗口加上最大回溯长度，配错会破坏语义。

`compact` 模式会真正删去后半段不需要重做的查询行（query 行），而不是给全量计算加个遮罩。但 `_store_yoco_full_kv()` 仍从前段共享给后段的完整历史表示建立所需缓存——**少算查询位置，不等于让模型忘掉之前的内容。**

### 5.3 128 这个数字为什么到处出现

这是全文我最想让你记住的一段。把常量摆在一起看：

| 常量 | 取值 | 来源 |
|---|---|---|
| 滑动窗口 | 128 token | 模型配置 |
| KV Cache 块大小 | 128 token | 缓存存储单元 |
| SWA 存储位置 | 128 的整数倍 | prefill 后存一次、decode 末尾再存一次 |
| 草稿块 | 5 token | DSpark，`DEEPINFER_USE_NEXTN_DECODE` |
| 推测回溯 | 6 token | 草稿块 + 1 |
| 最大命中长度 | L − 134 | 窗口 + 回溯 |
| 解码器重放 | 140 token | 必须 ≥ 窗口 + 回溯 |
| 每条序列 SWA 状态 | 20 层 × 134 × 528 B ≈ 1.4 MB | 仅编码器层，MXFP8 |
| 精确重建代价 | 2560 token × 20 层 | 有界重放要避免的 |

**模型里的 128 和存储系统里的 128 是同一个数，因为它们是一起被选出来的。5 和 6 和 134 和 140 也是。** 这就是从服务侧做协同设计的样子：模型有一个便宜到能存的窗口，缓存有一个刚好装下窗口的块大小。

### 5.4 命中规则：短追问为什么和长追问一样贵

规则本体：**命中长度上限为 L − 134，向下取整到 128 的块边界；若该边界没有存过 SWA 状态，再退一个整块。**

来看两个真实感的请求。同一段对话已累积 1460 token（1100 的 prompt + 360 的回复），上一轮在两个位置留下过 SWA 状态：1024（prefill 后）和 1408（decode 末尾）。

| 场景 | 新增 | L | 实际命中 | 编码器重算 | 解码器重放 |
|---|---|---|---|---|---|
| A：300 token 工具结果 | 300 | 1760 | 1408（有 SWA 状态） | 352 | 140 |
| B：40 token 用户消息 | 40 | 1500 | 1280 退化到底，**无 SWA 状态**，再退一块 → 1152 | 348 | 140 |

**第二个案例是最反直觉的：一个 40 token 的短追问，付了几乎和 300 token 长追问一样的 prefill。** 因为 L − 134 的上限把命中拉到了最后一个存过 SWA 状态的块边界之下，规则再退一整块。

换来的是什么？**系统永不需要存储解码器窗口状态、永不需要搬运它、prefill 时永不超过 140 token 走解码器。** 用一个可预测的、与上下文无关的常数，换掉了一整类跨机状态同步问题——对超大规模集群来说，这笔交易划算得离谱。

这也解释了为什么这块不能简单「优化」：任何试图把命中率拉高的改动，都会把 SWA 状态重新塞回缓存系统。

## 6. 工程层：零开销批调度器与 DeferredRelease

传统流程是串行的：CPU 选批次 → GPU 执行 → CPU 拿回结果后处理 → 再准备下一批。**GPU 算完当前批后，要等 CPU 把下一批排好，这段空隙不会自己消失。**

DeepInfer 的异步调度让两边重叠：GPU 算第 N 步时，CPU 已经在排第 N+1 步，并处理更早轮次的结果。`worker.py` 的执行顺序能说明这个安排——**Worker 先提交当前轮前向，再做上一轮后处理，最后提交当前轮采样。** 当前轮结果可能留到下一次调用才返回。

这带来两个必须处理的难点，也是这套设计真正的工程量所在：

1. **变长草稿验证要适应滞后时序**。准备下一轮时，上一轮的采样结果可能还没回到 CPU，`csrc/engine/zos.cpp` 与 Worker 的元数据修正路径在相邻批次间建立 remap，把本轮序列对应到上一轮设备结果，GPU 按真实结果修正输入和序列长度；需要回 CPU 的元数据走 pinned buffer + 事件衔接，而不是先把整轮结果同步回来。
2. **「请求结束」与「设备不再使用它的存储」不是同一时刻**。`DeferredRelease` 机制处理异步后的资源生命周期：序列元数据、输出对象、克隆时 pin 住的缓存块，都必须在安全位置回收。

**它不保证所有负载下设备都不空转**——低流量、通信或其他资源瓶颈照样影响利用率。它改善的只是执行链上的等待关系。收益在批次小、decode 步快的场景最明显，恰好是 Flash 类高吞吐模型所处的区间。

### 6.1 Batch-Invariant 与确定性采样

这对企业用户尤其重要——决定这套引擎能不能进生产（对账、审计、回归测试）：

- `DEEPINFER_BATCH_INVARIANT=1`：不管同批次合并了哪些请求，单条请求结果完全一致，避免浮点归约顺序不同导致的 logits 差异；
- 确定性采样：在上一项基础上，再设 `DEEPINFER_ENABLE_MTP_VARLEN=0` 并传入固定 `seq_seed`，相同输入 + 采样参数产出完全一致的输出。

> **限制：这两项当前仅 NVIDIA 平台支持，昇腾暂不支持。** 如果要在国产卡上做可复现回归，这是硬门槛。

## 7. 它刻意没给什么：三块缺口与各自的含义

这三块不是「没做完」，是**与内部基础设施强耦合，摘不出来**：

| 缺口 | 缺的是什么 | 自己补要做什么 |
|---|---|---|
| KV Cache 管理 | 完整跨实例缓存系统（3FS 侧） | 自建分页块缓存池：约 100 PiB SSD、峰值聚合读约 10 TiB/s、写约 850 GiB/s；Agent 流量留 3 天、聊天留 7 天，设计目标不活跃后存 72 小时以上 |
| PD 分离 | P/D 拆分 + 路由 + 交接 | README 说「所需引擎工作大部分已就位」，但拆分执行要自己搭；参考站内 `sys8` 的配比算法与 `sys12` 的 Mooncake 方案 |
| 副本负载均衡 | 多副本集群路由与流量均衡 | 与内部集群管理绑定，得接自己的 K8s / 网关层 |

这也解释了 `aihot` 日报里那条判断的底层原因：**vLLM / SGLang 一周内用同源内核追平了速度，真正的护城河差异其实在这三块工程组件上**——而它们恰好是开源双框架的强项。所以「别把 DeepInfer 当唯一答案」这个结论，不只是因为它慢了一步，而是因为**它交付的边界本来就不包含完整服务栈**。

至于 3FS 那组数字（100 PiB / 10 TiB/s），它说明另一件事：**DeepSeek 的缓存策略是建立在「SSD 容量近乎无限」这个前提上的**。中小规模部署照抄这套 TTL 策略没有意义，你的 SSD 没有 100 PiB，命中率和回本周期完全不同。

## 8. 抄得走与抄不走

| 维度 | 抄得走 | 抄不走 |
|---|---|---|
| 算子层 | FlashMLA / DeepGEMM / DeepEP 已开源，vLLM / SGLang 都已集成 | 与 V4.1-Flash 结构绑定的定制 kernel |
| 并行层 | Wide-EP 分阶段配比思想；EPLB 的 minimax 冗余复制 + 1e-4 扰动保确定性 | 384 专家 + 超节点硬件的具体取值（EP64–128 是它自己算出来的） |
| 解码层 | Confidence Head → 前缀累积收益 → 变长验证预算这条因果链 | SWA Bounded Replay 的近似重建（窗口必须与训练一致，现成模型硬套会掉质量） |
| 工程层 | 异步调度「先提交前向、再补上轮后处理、最后提交采样」的时序 | 3FS 级别的缓存假设；PD / 跨实例 LB 得自己搭 |

## 9. 与站内文章的衔接

- **`fw1`（为什么需要推理引擎）** 讲的三道墙——显存碎片、静态批、GPU 空转——DeepInfer 每题都给了新答案：分页块缓存、Wide-EP 超大 batch、零开销调度器；
- **`fw2` / `fw3`（vLLM / SGLang 内部）** 是本篇的血缘参照：DeepInfer 的 C++ Scheduler 与 ZeroMQ 式入口，追根到底是从 vLLM 那套 `LLMEngine + Scheduler + Worker` 长出来的；
- **`ep4`（DSpark 深度解析）** 讲了草稿侧机制，本篇补调度侧落地与那张「低并发收益为负」的表格——**这两个视角缺一个就会误判 DSpark 的价值**；
- **`ep7`（MTP 与置信度头）** 的置信度头在 DeepInfer 里被赋予了「分配验证预算」的明确职责，是这条链的生产版；
- **`sys9` / `sys10`（KV Cache 全景与工程实现）** 算过 KV 有多大、怎么管；本篇的 SWA 分层是个反向案例：**先问数据活多久，再决定存哪一层**；
- **`sys6`（HiSparse）** 与 SWA Bounded Replay 是同一思路的两种形态：都把「不需要精确的历史」换成常数成本；
- **`sys8`（PD 分离）/ `sys12`（Mooncake）** 正是 DeepInfer 没开源的那块，可以直接用来补齐；
- **`fa4`（DeepSeek V4）** 里的 CSA+HCA 压缩注意力，是 890 B/token 这个全局 KV 数字的模型侧来源——**模型侧压缩 + 系统侧重放，两条腿一起才把长上下文成本打下来**。
