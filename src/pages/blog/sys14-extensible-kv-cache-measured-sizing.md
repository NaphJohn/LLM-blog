---
title: '推理系统基础设施手记（十四）：Extensible KV Cache——KV 容量从「启动前估算」到「启动后实测」，以及虚拟地址如何把捕获与定容解耦'
description: '拆开 vLLM #56492 的三层机制：为什么 CUDA graph 捕获把 KV cache 的分配时间钉死（图里存的是指针，不是张量）、8% 的 gpu_memory_utilization 余量在 192 GiB 卡上等于 15 GiB 长期闲置、以及 cuMemAddressReserve + cuMemMap 的「地址先占、物理页后补」如何让定容延后到预热之后而不必重捕获。附定容公式的每一项为什么长这样、GB200 上八个模型的实测增益、DeepSeek-V4-Flash 为什么只涨 4.1%、H200 上 433k→672k 为什么不能当通用预期，以及 TPOT 上升 6–18% 的 capacity–latency 机制与回退的正确姿势（0.9 不等于回退）。'
pubDate: 2026-11-25
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys14-extensible-kv-cache-measured-sizing
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

老版本 vLLM 定 KV cache 大小的方式，本质上是**在信息最少的时候做一个不可逆的决定**：必须在 CUDA graph 捕获之前把 KV cache 建好（因为图里固化的是地址），而捕获和预热之后才会产生的那部分显存占用（graph 私有池、attention workspace、sampler/logprobs buffer、投机解码草稿），只能靠一次 profiling 猜 + 8% 的余量兜。

`#56492` 做的事很干脆：**把「地址确定」和「容量确定」拆成两个时刻**。地址在捕获前用虚拟内存预留占住，物理页等到预热跑完、真实空闲显存量出来之后再一次性补进去。地址没动，所以已捕获的图继续有效；容量是量出来的，所以那 8% 的余量可以还回去。

理解这个特性只需要抓住一句话：**它不是一个分配器优化，是一次时序重构**。

## 1. 约束从哪来

### 1.1 CUDA graph 里存的是指针，不是张量

捕获一个 CUDA graph，实质是把一串 kernel launch 的参数快照下来：grid/block 维度、以及**每个 tensor 参数的设备地址**。重放的时候不重新解析张量，直接按快照里的地址去访存。

所以顺序不能交换：

```text
KV cache 必须先存在（拿到固定 VA）
    ↓
才能捕获（把地址写进图）
    ↓
捕获和预热又会产生新的显存需求
    ↓
但这些需求发生时，KV 的容量早已定型
```

这就是老版本尴尬的地方——**KV cache 是显存里最大的那块，却是最早被决定的**。

### 1.2 老版本被迫做的三个猜测

profiling 那一次采样，能看到的只有模型前向本身。之后要发生的事它一概不知道，于是三块内容全靠推断：

| 需要预留的东西 | 老版本的处境 | 猜错的后果 |
|---|---|---|
| **CUDA graph 内存** | 要么忽略，要么跑一次「丢弃式捕获」专门量 | 忽略 → 运行 OOM；量一次 → 启动白花几十秒 |
| **预热期瞬时峰值**（attention workspace、sampler/logprobs、spec-decode draft） | profiling 根本没跑到这些路径 | 只能靠 slack 吸收 |
| **运行时峰值**（长 prefill、大 batch decode、prefix cache 命中的 chunked context） | 完全未知 | 同上 |

### 1.3 8% slack 的真实价格

`gpu_memory_utilization=0.92` 那个 0.08，落到 192 GiB 的卡上是 **约 15 GiB**。这块显存不是「设计上的冗余」，而是**估算能力的缺口**——你不知道要留多少，就按一个经验百分比留。

在 HBM 上，15 GiB 能换多少 token？对 70B 级模型、GQA、fp8 KV，粗算下来是**几十万 token 量级**。也就是说老版本每台机器都在长期闲置一大块本可以变成并发容量的显存，而代价已经付了好几年。

## 2. 破法：把地址和容量拆开

### 2.1 VMM 三件套

CUDA 驱动层的虚拟内存管理 API 允许把「保留地址」和「背后挂物理内存」分成两步：

```text
cuMemAddressReserve(&va, size, align, 0, 0)   // 只占地址，0 字节物理内存
cuMemCreate(&handle, size, &prop, 0)          // 申请一块物理显存
cuMemMap(va + offset, size, 0, handle, 0)     // 把物理块映射进预留的地址
cuMemSetAccess(va, size, &desc, 1)            // 设置读写权限
```

释放走 `cuMemUnmap` + `cuMemRelease`，地址段本身可以留着重用。ROCm 上对应 `hipMem*` 一族。

这套东西在 GPU 上一直都有（统一虚拟寻址的底层），只是从来没被用在 KV cache 上。原因也简单：**以前没人需要「先有地址、后有容量」这种语义**。

### 2.2 ExtensibleTensor：对上层完全伪装的张量

vLLM 把它封装成 `vllm/utils/extensible_tensor.py` 里的 `ExtensibleTensor`，通过 DLPack 暴露给 PyTorch。对上层来说它就是一个普通的 device tensor——shape、stride、data_ptr 都正常。

三个实现细节值得注意：

- **按段（segment）切分**。整块容量按 per-layer（或 per-tensor）切成若干段，每段自己独立 commit。这样每个 layer 的 view 落在自己那段连续 VA 里，kernel 侧一行代码都不用改。
- **commit 粒度是驱动 granule**（通常 2 MiB）。小段浪费一点，大段正好。
- **新提交的块必须清零**。物理页可能是从别的进程回收来的，不清零就读到脏数据——既是正确性问题，也是安全问题。

<div class="fig">
<svg viewBox="0 0 680 236" width="100%" font-size="12" role="img" aria-label="虚拟地址预留与按需提交的结构">
<title>图 1：一段预留的虚拟地址，背后只有被触碰的段才挂物理页</title>
<defs>
<marker id="m1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<text x="34" y="20" font-size="11" fill="#6b7280">预留的虚拟地址空间（容量上限，捕获前就确定）</text>
<rect x="34" y="28" width="612" height="40" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="0.8" stroke-dasharray="4 3"/>
<line x1="34" y1="48" x2="646" y2="48" stroke="#2563eb" stroke-width="0.5" stroke-dasharray="2 4" opacity="0.4"/>
<text x="340" y="60" font-size="10" fill="#1d4ed8" text-anchor="middle">layer 0 · layer 1 · layer 2 · …… · layer L−1　（每段 VA 连续，地址永不变动）</text>

<text x="34" y="92" font-size="11" fill="#6b7280">背后的物理页（预热结束时才决定挂多少）</text>

<rect x="34" y="100" width="150" height="34" rx="3" fill="#ecfdf5" stroke="#10b981" stroke-width="0.8"/>
<text x="109" y="121" font-size="10" fill="#047857" text-anchor="middle">已 commit</text>
<rect x="190" y="100" width="150" height="34" rx="3" fill="#ecfdf5" stroke="#10b981" stroke-width="0.8"/>
<text x="265" y="121" font-size="10" fill="#047857" text-anchor="middle">已 commit</text>
<rect x="346" y="100" width="150" height="34" rx="3" fill="#f9fafb" stroke="#9ca3af" stroke-width="0.8" stroke-dasharray="3 3"/>
<text x="421" y="121" font-size="10" fill="#6b7280" text-anchor="middle">未映射（无物理页）</text>
<rect x="502" y="100" width="144" height="34" rx="3" fill="#f9fafb" stroke="#9ca3af" stroke-width="0.8" stroke-dasharray="3 3"/>
<text x="574" y="121" font-size="10" fill="#6b7280" text-anchor="middle">未映射</text>

<line x1="109" y1="140" x2="109" y2="158" stroke="#10b981" stroke-width="1" marker-end="url(#m1)"/>
<line x1="265" y1="140" x2="265" y2="158" stroke="#10b981" stroke-width="1" marker-end="url(#m1)"/>

<rect x="34" y="162" width="274" height="26" rx="3" fill="#fff" stroke="#10b981" stroke-width="0.8"/>
<text x="171" y="179" font-size="10" fill="#047857" text-anchor="middle">warmup 真正触碰到的块 → 提交物理页</text>

<rect x="34" y="200" width="612" height="28" rx="4" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="48" y="218" font-size="10" fill="#854f0b">最后一步 extend_kv_cache：把量出来的剩余预算一次性映射进同一段 VA —— 指针值不变，已捕获的图无需重捕获</text>
</svg>
<div class="cap">图 1 · 预留与提交是两次操作。虚线部分全程零物理内存，所以「先占地址」几乎不花代价。</div>
</div>

### 2.3 为什么地址稳定是生死线

这是整个方案唯一不能妥协的点。如果最终容量和预留容量不一致，且**必须搬家**，那么：

- 已捕获的 CUDA graph 全部作废，得重新捕获（回到起点，启动时间优势归零）；
- 所有持有该张量指针的组件——attention backend 的 workspace 视图、KV connector 的注册信息、sleep mode 的快照——全部失效。

所以设计上预留的容量取的是**profiling 给出的上界**（保守、够大），最终只可能往里**补**物理页，绝不可能超出。这个方向性保证了「扩展」永远只是映射更多页，而不是重排。

同一套地址还顺带解决了另外三件事：

- **sleep mode**：睡眠时 `unmap` 释放物理页，唤醒时按原地址 `remap`，容量和内容位置都不变；
- **KV connector**：worker 侧在最终 extend 之后才创建，注册到的是最终内存；段边界对齐到 granule，保证 RDMA 注册不会跨物理块；
- **权重更新**：RL 场景下 `reload_weights` 需要瞬时峰值，warmup 量不到。默认流程会先清空 KV cache 再更新，此时可临时释放 KV 的物理页、更新完再映射回来。

## 3. 定容：公式里每一项都有理由

```python
budget = util × device_mem − weights
peak   = max(profiling_peak, warmup_peak)
margin = max(1 GiB, 0.75 × peak)
blocks = floor((budget − peak − margin) / bytes_per_block)
final  = min(blocks)   # over all workers × all DP engines
```

### 3.1 为什么 peak 取 max 而不是相加

profiling 峰值和预热峰值测的是**同一类东西在两个时刻的瞬时占用**，不是两个并存的开销。取 max 是对的，相加会双重计数、白扔掉一大块容量。

取 max 的含义是「保守地相信观测到的最大值」，也就是**宁可少算一点容量，也不要在长 prefill 时 OOM**。

### 3.2 margin 为什么是 `max(1 GiB, 0.75 × peak)`

两个下限分别管两种失败模式：

- **1 GiB 绝对值**：挡住小模型上「peak 很小、余量算出来只有几十 MB」的情况，那里一次意外的分配就能撞墙；
- **0.75 × peak 相对值**：峰值越大，抖动幅度越大。peak 已经 15 GiB 的配置，1 GiB 余量显然不够，按比例放大才合理。

这个 0.75 是经验系数，不是推导出来的——它承担的是「warmup 没覆盖到的路径」的风险敞口。所以特性保留了自动回退：**只要配置落在不支持的分支，就退回老的估算路径**，而不是硬上。

### 3.3 预热新增的三个探针

量的准确性完全取决于 warmup 跑没跑到。为此预热多了三段：

| 新增步骤 | 量的是什么 | 不加会怎样 |
|---|---|---|
| 满预算 prefill | 长上下文 prefill 的 attention workspace 峰值 | 长 prompt 首次到来时 OOM |
| prompt-logprobs 分块 | logprobs 相关的临时 buffer | 请求 logprobs 时才炸 |
| MLA 模型的 chunked context prefill | MLA 的 context 上投影（prefix cache 命中路径） | 命中缓存反而更吃显存 |

第三项尤其值得注意：**MLA 的 context 上投影只在「有 cached context」时才走**，普通预热跑不到。这正是老版本靠 slack 蒙、蒙错就 OOM 的典型路径。

### 3.4 为什么要取全局最小

TP 和 DP 的各 rank 是**步调一致**的：每一步必须处理同样多的 block，否则 collective 就死等。所以容量必须收敛到所有 worker 的最小值，再走一次 `extend_kv_cache` RPC 统一下发。

这也解释了为什么「每个 rank 各自测各自的空闲显存」不够——测得快的 rank 会多算容量，最后还是要砍到最慢的那个。

<div class="fig">
<svg viewBox="0 0 680 214" width="100%" font-size="12" role="img" aria-label="显存预算的分配对比">
<title>图 2：旧路径的 8% slack 在新路径里变成实测峰值加按公式算出的余量</title>

<text x="34" y="18" font-size="11" fill="#6b7280">旧 · gpu_memory_utilization = 0.92（KV 由估算决定）</text>
<rect x="34" y="26" width="43" height="34" fill="#f3f4f6" stroke="#d1d5db" stroke-width="0.5"/>
<rect x="77" y="26" width="435" height="34" fill="#d1d5db"/>
<text x="294" y="48" font-size="10" fill="#374151" text-anchor="middle">KV cache（估算）</text>
<rect x="512" y="26" width="86" height="34" fill="#9ca3af"/>
<rect x="598" y="26" width="48" height="34" fill="#fecaca"/>
<text x="622" y="48" font-size="9" fill="#b91c1c" text-anchor="middle">slack</text>
<text x="34" y="74" font-size="9.5" fill="#9ca3af">权重</text>
<text x="516" y="74" font-size="9.5" fill="#9ca3af">graph/workspace 估算</text>
<text x="600" y="74" font-size="9.5" fill="#b91c1c">8% 空转 ≈15 GiB</text>

<text x="34" y="108" font-size="11" fill="#6b7280">新 · utilization 未设时按 1.0（KV 由实测决定）</text>
<rect x="34" y="116" width="43" height="34" fill="#f3f4f6" stroke="#d1d5db" stroke-width="0.5"/>
<rect x="77" y="116" width="470" height="34" fill="#93c5fd"/>
<text x="312" y="138" font-size="10" fill="#1e3a8a" text-anchor="middle">KV cache（实测）</text>
<rect x="547" y="116" width="61" height="34" fill="#a7f3d0"/>
<rect x="608" y="116" width="38" height="34" fill="#ddd6fe"/>
<text x="34" y="164" font-size="9.5" fill="#9ca3af">权重</text>
<text x="551" y="164" font-size="9.5" fill="#047857">实测瞬时峰值</text>
<text x="600" y="164" font-size="9.5" fill="#6d28d9">margin</text>

<rect x="34" y="178" width="612" height="26" rx="4" fill="#f6f8fa" stroke="#e5e7eb" stroke-width="0.5"/>
<text x="48" y="195" font-size="10" fill="#374151">省下来的不是「算法更聪明」，而是不再需要用百分比去覆盖未知；margin 是可计算的，slack 不是</text>
</svg>
<div class="cap">图 2 · 示意比例取 GB200 上 Qwen3-8B 的方向（150.2 → 161.6 GiB），不是逐卡数据。</div>
</div>

## 4. 收益：数字，以及它的边界

GB200、每卡 189 GiB、其余默认设置，八个模型全部有增益：

| 模型 | TP | 旧 sizing | 新默认 | Token 增益 | init（不含权重） | 端到端 |
|---|---|---|---|---|---|---|
| Qwen3-8B | 1 | 150.2 GiB | 161.6 GiB | +7.5% | −29% | −18% |
| Qwen3-8B + EAGLE3 | 1 | 147.3 GiB | 157.4 GiB | +6.8% | −32% | −14% |
| Qwen3-32B | 1 | 103.1 GiB | 113.0 GiB | +9.5% | −30% | −12% |
| gpt-oss-20b | 1 | 150.5 GiB | 161.0 GiB | +7.0% | −28% | −11% |
| Qwen3.6-35B-A3B | 1 | 97.6 GiB | 106.2 GiB | +8.8% | −21% | −9% |
| MiniMax-M2.7 | 4 | 109.7 GiB | 118.4 GiB | +7.9% | −31% | −6% |
| Qwen3-235B-A22B-FP8 | 4 | 108.1 GiB | 117.1 GiB | +8.2% | −34% | −17% |
| DeepSeek-V4-Flash | 4 | 119.9 GiB | 124.9 GiB | +4.1% | −30% | −9% |

两个观察：

**启动收益不是来自捕获变快，而是来自少捕一次。** 老版本为了猜 graph 内存要跑一次丢弃式捕获，现在直接跳过——真正的捕获在两个版本里耗时一样。

**DeepSeek-V4-Flash 只涨 4.1%，这一行最值得看。** 它捕获后确实需要约 15 GiB（graph + workspace），也就是说**旧的那 8% slack 本来就在被吃掉**，没有多少余量可回收。这给出了判断自己能不能受益的方法：

> 收益大小 ≈ 你的模型在旧版本里被高估了多少。
> 如果 graph 私有池和 workspace 很大（大 batch、多图档位、投机解码、MLA），旧 slack 早已吃满，别期待 10%。

**关于 H200 上 MiniMax-M3 从 433k 涨到 672k（+55%）**：这是官方 release notes 的头条数字，但它是另一块卡、另一个模型，且没有给分解。合理解释是那套配置的旧估算偏差远大于 GB200 上的样本。**当作「上限存在的证据」，不要当作预期值。**

## 5. 代价：容量和延迟是同一个旋钮

### 5.1 机制

KV 容量变大，调度器就不再需要为了省显存而 preempt / 重算被换出的序列。它会把新增容量**换算成更多同时运行的序列**。而 decode 阶段每步的访存量是 `Σ 各序列上下文长度`，**KV 读带宽占了大头**——batch 变大，每 token 时间就上升。

这是教科书式的 capacity–latency 权衡：**你并没有「凭空得到显存」，你是把显存换成了并发容量**。吞吐和 TTFT 受益，TPOT 受损。

官方口径：部分 H200 模型在**并发 128 时 TPOT 上升 6–18%**。

### 5.2 一个需要标注的疑点

按上面的机制，TPOT 上升应该主要发生在「并发被抬高」的时候。但报告里**并发固定在 128 也涨**——在 `max_num_seqs` 打满的情况下 batch 不会更大，说明还有别的因素。可能的解释是更少的抢占让 decode batch 更「纯」（混入的 prefill 变少），而纯 decode 的 128 路本身就是最吃带宽的形态；也可能是 block 分布更分散导致 attention kernel 效率下降。

官方只给了现象没给分解，这一段属于推断。**实操上不必纠结归因，直接测自己的 P99 就行。**

### 5.3 谁该关

| 场景 | 建议 |
|---|---|
| 离线批处理、吞吐优先、embedding/rerank 批量 | 默认开，显式删掉 `--gpu-memory-utilization` |
| 交互式 chat / voice，TPOT 敏感 | 先测 P50/P99；不达标就显式设 0.9 |
| 一卡多引擎，或卡上还有别的进程 | 必须显式设——这是该参数现在的主要用途 |
| DGX Spark 这类统一内存 | 必须留出给 OS 和 CPU 侧的空间 |
| 显存面板告警 | 忽略。引擎刻意把可用显存全部吃进 KV，贴近 100% 是设计 |

## 6. 兼容性，以及「回退」的正确姿势

不支持的配置会自动退回老的估算路径（不会报错），只有**显式**传 `--enable-extensible-kv-cache` 才会在某些分支直接报错：

| 配置 | 行为 |
|---|---|
| 非 CUDA / ROCm（如 XPU、CPU） | 关。VMM 原语缺失 |
| V1 model runner | 关（V1 已移除，无意义） |
| `kv_cache_memory_bytes` / `num_gpu_blocks_override` | 关。你已显式给了尺寸，实测无意义 |
| HiSparse | 关。它自管 KV cache，需改造为走扩展分配器 |
| Elastic EP | 关。弹性扩容的 rank 跳过 warmup |
| LMCache MP connector | 关。走 legacy CUDA IPC，无法导出 VMM 内存 |
| Mooncake NVLink / BAREX（`custom_mem_pool`） | 关。cache 必须来自连接器自己的池 |
| ROCm + NIXL / Mooncake / MoRIIO | 关。UCX/ROCr 尚不能注册 HIP VMM 内存 |
| 没有 KV cache 的模型 | 关。没有可定容的东西 |

<div class="fig">
<svg viewBox="0 0 680 106" width="100%" font-size="12" role="img" aria-label="显式设置 utilization 与真正回退的区别">
<title>图 3：0.9 只是上限，不是回退</title>
<rect x="34" y="14" width="612" height="34" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="1"/>
<text x="48" y="30" font-size="11" fill="#b91c1c" font-weight="600">--gpu-memory-utilization 0.9 ≠ 回退</text>
<text x="48" y="43" font-size="10" fill="#7f1d1d">它只是上限。实测仍会从 90% 里扣掉量到的 graph + 预热内存，结果比旧版小 0.6–3.7%。</text>
<rect x="34" y="56" width="612" height="34" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="1"/>
<text x="48" y="72" font-size="11" fill="#047857" font-weight="600">--no-enable-extensible-kv-cache 才是回退</text>
<text x="48" y="85" font-size="10" fill="#065f46">回到 profiling 估算 + 8% slack 的老算法。此时 utilization 的默认值仍是 0.92。</text>
</svg>
<div class="cap">图 3 · 两个开关作用在不同的层：一个是预算上限，一个是定容算法本身。</div>
</div>

显式设置 0.8–1.0 之间、且因此压住超过 1 GiB 时，vLLM 会在启动时打一条警告，明确告诉你「有多少 GiB 被挡在 KV cache 之外」。

还有几个容易忽略的**副作用**：

- **显存监控长期贴近 100%**。这不是泄漏。
- **权重热更新**（RL 训练 loop）在没有清空 KV 的暂停下，需要 `gpu_memory_utilization` 留出余量；直接调 `reload_weights` 时如果连最大单层都放不下，会在加载前失败。
- **KV connector 的注册时机**变了：worker 侧 connector 在最终 extend 之后创建，注册的是最终内存。自研 connector 要确认自己不是拿的预留容量。

## 7. 上线检查清单

```text
[ ] 镜像版本 ≥ 0.34（含 #56492）
[ ] 删掉遗留的 --gpu-memory-utilization 0.9 / 0.92（吞吐场景）
[ ] 确认没设 --kv-cache-memory-bytes（否则整个特性不生效）
[ ] 确认没用 Mooncake NVLink/BAREX / LMCache MP / HiSparse / Elastic EP
[ ] 记录一次启动日志里的「block 数」与「init engine 耗时」作为基线
[ ] 交互式服务：压测 P50 / P99 TPOT，对比旧版本
[ ] 显存告警阈值上调，接受长期 ~100%
[ ] 回滚预案写的是 --no-enable-extensible-kv-cache，不是 utilization
```

## 8. 与站内文章的衔接

- **`sys9`（KV Cache 全景）** 给了「KV 到底多大、五个可量化旋钮」的静态视角。本篇补的是**动态视角**：这个旋钮在启动流程的哪个时刻被拧动，以及为什么它以前只能被估算；
- **`sys10`（KV Cache 工程实现）** 讲了 vLLM 的分页 Block 管理和跨层统一布局。Extensible KV cache 不改变分页语义，改的是**块数从哪来**——block 还是那些 block，只是 `num_gpu_blocks` 这个数从「猜」变成了「量」；
- **`fw7`（CUDA Graph）** 是理解本篇前提的必读：图里存的是地址快照，所以任何「重分配」方案都不成立。这也是为什么本篇的技术核心是「地址不动」而不是「分配更快」；
- **`sys12`（Mooncake）** 和 **`sys11`（HiCache L2）** 都在做 KV 的**跨层/跨机搬运**。Extensible KV cache 与它们的关系是**正交但有冲突**：它管 HBM 内部怎么定容，搬运管 HBM 之外怎么放；但 Mooncake 的 `custom_mem_pool` 和 LMCache MP 的 IPC 导出都与 VMM 内存不兼容，所以会回退；
- **`sys6`（HiSparse）** 是另一个「显存不够怎么办」的答案——把 HBM 当缓存、把 KV 分层。两者思路相反：HiSparse 是**承认装不下、分层换容量**，Extensible 是**先把装得下的算准**。实际部署里可以叠加（先算准，再分层），但目前代码路径上 HiSparse 会关掉 Extensible；
- **`sys13`（MoE 专家并行）** 里的 Grouped GEMM 和 EPLB 吃的是**计算侧**的显存与带宽；本篇吃的是**缓存侧**。两者在 decode 阶段争同一块 HBM 带宽——这解释了为什么 MoE 大模型上 TPOT 对 KV 容量变化更敏感。

---

## 系列导航

**推理系统基础设施手记**

- [（十三）MoE 专家并行的通信与计算](/blog/sys13-moe-expert-parallel-comms)
- [（十二）Mooncake——以 KVCache 为中心的 PD 分离架构](/blog/sys12-mooncake-kvcache-centric-serving)
- [（十一）SGLang HiCache L2 在昆仑芯 P800/P900 上的适配与调优实战](/blog/sys11-hicache-l2-kunlunxin-tuning)
- [（十）KV Cache 工程实现](/blog/sys10-kv-cache-engineering-sglang-vllm-attentionstore)
- [（九）KV Cache 全景](/blog/sys9-kv-cache-fundamentals-optimization)

**相关**

- [（七）CUDA Graph](/blog/fw7-cuda-graph)——为什么图里存的是地址
- [（七）HiSparse 分层 KV 缓存](/blog/sys6-hisparse-hierarchical-kv-cache)——相反方向的解法
- [（一）MLA 算子](/blog/op1-mla-operator)——context 上投影的显存来源
