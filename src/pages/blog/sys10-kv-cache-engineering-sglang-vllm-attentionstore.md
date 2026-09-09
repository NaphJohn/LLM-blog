---
title: '推理系统基础设施手记（十）：KV Cache 工程实现——SGLang 两级内存池、vLLM Block 管理与 AttentionStore 多级缓存'
description: '从原理走到代码：SGLang 的 req_to_token_pool → token_to_kv_pool 两级映射与 Radix Tree 前缀复用、HiCache 三层存储与预取重叠；vLLM 的 PagedAttention、显存 Profiling 定 block 数、跨层统一布局（PR 27743）、LRU Block 池与链式前缀哈希；以及百度 AttentionStore 在昆仑芯 P800 上的实测（64K 上下文 TTFT 降 6.2 倍、多轮吞吐 5.4 倍）。'
pubDate: 2026-09-09
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys10-kv-cache-engineering-sglang-vllm-attentionstore
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

原理篇（sys9）说了 KV Cache 该怎么削，这一篇看**两个主流引擎和一个工业级缓存系统到底怎么管它**：SGLang 用**两级内存池 + 基数树**把前缀复用做到 token 粒度；vLLM 用**分页 + LRU 块池**把显存碎片压到最低；百度 AttentionStore 则把视野从单卡扩到**集群**，用全局索引让调度"看得见缓存"。

## 1. SGLang：两级内存池

### 1.1 先看上下文：Scheduler 的事件循环

KV Cache 的管理散落在调度器的每一步里，先看清主循环：

```
Event Loop 无限循环:
  process_input_requests   → 收请求、分类、入 waiting_queue
  get_next_batch_to_run    → 组批（prefill 优先，否则更新 decode 批）
  run_batch                → 前向（生成 / idle / embedding）
  process_batch_result     → 处理输出、更新状态、释放或缓存 KV
```

内存不足时，prefill 请求会被**分块（chunked）**，decode 请求会被**撤回（retracted）**，重新排回 waiting_queue。

### 1.2 两级内存池

SGLang 把"请求 → token → 实际 KV 数据"拆成两级映射：

<div class="fig">
<svg viewBox="0 0 660 250" role="img" aria-label="SGLang 两级内存池映射关系">
  <title>图 1：SGLang 两级内存池与 tree_cache 的关系</title>
  <rect x="14" y="56" width="112" height="44" rx="6" fill="#f6f8fa" stroke="#30363d"/>
  <text x="70" y="76" font-size="12" fill="#6b7280" text-anchor="middle">Request</text>
  <text x="70" y="92" font-size="13" fill="#1a1a1a" text-anchor="middle">token 序列 ABC</text>
  <line x1="130" y1="78" x2="164" y2="78" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="164,78 156,74 156,82" fill="#2563eb"/>
  <rect x="170" y="42" width="196" height="72" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="268" y="62" font-size="12" fill="#2563eb" text-anchor="middle">req_to_token_pool</text>
  <text x="268" y="80" font-size="11" fill="#1a1a1a" text-anchor="middle">[req_pool_idx][pos]</text>
  <text x="268" y="98" font-size="11" fill="#1a1a1a" text-anchor="middle">→ out_cache_loc</text>
  <line x1="370" y1="78" x2="404" y2="78" stroke="#2563eb" stroke-width="1.5"/>
  <polygon points="404,78 396,74 396,82" fill="#2563eb"/>
  <rect x="410" y="30" width="236" height="96" rx="6" fill="#ecfdf5" stroke="#10b981"/>
  <text x="528" y="50" font-size="12" fill="#047857" text-anchor="middle">token_to_kv_pool</text>
  <text x="528" y="68" font-size="11" fill="#1a1a1a" text-anchor="middle">[layer][out_cache_loc]</text>
  <text x="528" y="84" font-size="11" fill="#1a1a1a" text-anchor="middle">[head][head_dim]</text>
  <text x="528" y="102" font-size="11" fill="#1a1a1a" text-anchor="middle">→ cache_k, cache_v</text>
  <text x="528" y="118" font-size="11" fill="#047857" text-anchor="middle">显存里真正的大块</text>
  <rect x="170" y="164" width="196" height="60" rx="6" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="268" y="184" font-size="12" fill="#854f0b" text-anchor="middle">tree_cache（RadixCache）</text>
  <text x="268" y="202" font-size="11" fill="#1a1a1a" text-anchor="middle">token id → out_cache_loc</text>
  <text x="268" y="218" font-size="11" fill="#1a1a1a" text-anchor="middle">跨请求复用前缀</text>
  <line x1="268" y1="164" x2="268" y2="118" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="14" y="234" font-size="11" fill="#6b7280">一级管"谁的第几个 token"，二级管"这块索引指向哪个 KV 数据"；tree_cache 决定哪些 token 可以直接复用。</text>
</svg>
<figcaption>图 1：一级池把请求映射到 KV 索引，二级池把索引映射到真实 KV 数据；tree_cache 负责跨请求前缀复用。</figcaption>
</div>

| 池 | 形状 | 索引方式 | 返回 |
|----|------|----------|------|
| **req_to_token_pool** | max_running_requests × context_len | `[req_pool_idx][pos]` | `out_cache_loc`（KV 索引） |
| **token_to_kv_pool** | layers × max_tokens × heads × head_dim | `[layer_id][out_cache_loc][head][dim]` | `cache_k`, `cache_v` |

前向传播时通常**一次取一整层的 KV**，因为该层要用到请求的所有历史 token。

### 1.3 tree_cache：跨请求复用

- 键是 **token id**（同一个 token 的 KV 与具体请求无关），值是 `out_cache_loc`。
- 实现有 `ChunkCache`（关闭 radix cache 时的简化路径）与 `PrefixCache`（含 **RadixCache**、**HiCache**）。

### 1.4 prefill 全流程（以请求 ABC 为例）

**① match_prefix**：在 radix tree 里找最长前缀。若树里已有节点 `AFG`，请求 `ABC` 会命中 `A`，并把 `AFG` 拆成 `A` 和 `FG`。

**② prepare_for_extend**：
- `req_to_token_pool`：分配 `req_pool_indices`，写入前缀部分
- `token_to_kv_pool`：只给**未命中的 token** 分配槽位 → 3 个输入 token − 1 个命中 = **只分配 2 个**（B、C）

**③ run_batch → forward_extend**：attention 后端把 B、C 的 K/V 写入刚分配的 `out_cache_loc`；计算时 Q 是 B、C，K/V 是通过索引取到的 A（命中）+ B、C（新增）。

**④ cache_unfinished_req**：把 `BC` 作为 `A` 的子节点挂回树，锁引用 +1。

### 1.5 decode 全流程

- `prepare_for_decode`：`token_to_kv_pool` 每步只分配 **batch_size × 1** 个槽位（每步只生成 1 个 token）
- `run_batch → forward_decode`：Q 是本次的 1 个 token，K/V 是全部历史
- 请求完成才调 `cache_finished_req`；**未完成的 decode 请求不对 cache 做额外操作**，只在最后把生成序列追加回树

### 1.6 HiCache：把显存扩展成三层

模块构成：

| 模块 | 职责 |
|------|------|
| **HiRadixTree** | GPU / CPU 双层前缀树，原生支持 KV 在两级间自动同步 |
| **Storage Backend** | 可插拔后端，已集成 3FS、Mooncake、NIXL；统一 `batch_get/set/exists`，支持零拷贝 |
| **Global KVManager** | 分布式文件系统元数据统一管理 |
| **3FS** | DeepSeek 开源高性能分布式文件系统（RDMA + NVMe SSD，TiB/s 级聚合读带宽） |

两个关键优化：

1. **预取与等待并行**：请求一入队就触发 `prefetch_from_storage`，利用排队这段时间把 KV 从存储搬到 Host 内存。调度策略有三种：
   - `best_effort`：调度到时若仍在预取就中断，直接进推理
   - `timeout`：超过阈值才中断，否则本轮跳过该请求
   - `wait_complete`：必须预取完才调度
2. **加载与计算重叠**：Host → GPU 通过独立 CUDA Stream **逐层加载**（`load_to_device_per_layer`），第 i 层的 KV 一就绪就可以开算，不必等全部层搬完。

效果是把阻塞 I/O 藏进"排队等待"和"GPU 计算"里，扩大有效缓存容量的同时尽量不伤 TTFT。

### 1.7 稀疏化的两种落地

**SWA（滑动窗口）**：每步只看最近 W 个 token，窗口外的 KV **真正释放**。实现靠"SWA 层放小池 + 滑窗即时回收 + tombstone 保留前缀命中能力"三件套：计算上用 window mask，显存上用双池 + 双 LRU + tombstone，做到"窗口外真回收、前缀仍可共享"。

**DeepSeek NSA**：**KV 全存，只读一部分**——和 SWA 思路相反。Indexer 给每个 page 打分选 Top-K，与 Compressed 全局摘要、Sliding 局部窗口三路并行融合。KV 不删，但 attention 的读量从 O(seq) 降到 O(K·page + W + seq/L)，长序列下算力和带宽同时下降。

## 2. vLLM：分页、块池与 Connector

### 2.1 PagedAttention

把操作系统虚拟内存那套搬进 KV Cache：

- 每个请求的 KV 切成固定大小的**逻辑块**（默认 `block_size = 16` token）
- **Block Table** 记录逻辑块 → 物理块的映射
- 物理块在显存里可以**不连续**，从而消除碎片

| 概念 | 操作系统类比 |
|------|--------------|
| Request | 进程 |
| Logical KV block | 虚拟页 |
| Block Table | 页表 |
| Physical KV block | 物理页帧 |

### 2.2 初始化：先算清楚能分多少块

1. **构造 dummy 数据**：按 `max_num_seqs` 和 `max_num_batched_tokens` 造出一批假请求（如 10 tokens / 3 seqs → 长度 4、3、3）
2. **模拟一次前向**测出峰值占用：
   `KV Cache 可用显存 = GPU 可用显存 −（模型权重 + 中间激活）− CUDA Graph 预留`
3. **算块数**：`总块数 = KV Cache 可用显存 / 所有层的物理块大小之和`，其中
   `per_layer_page_size = block_size × num_kv_heads × head_size × dtype_size × 2`
4. **预分配**一张 Empty Tensor 常驻显存（CPU 侧同理，默认 4 GiB）

### 2.3 跨层统一布局（PR 27743）

旧布局是"每层各自一块 + K/V 分开"，对计算无碍，但对 **KV offload 是毁灭性的**——有效块太小，传输效率极差。

PR 27743 把一整个逻辑块的所有层 KV 拼成连续内存：

```
旧: (2, num_blocks, block_size, num_kv_heads, head_size)
新: (num_layers, 2, num_blocks, block_size, num_kv_heads, head_size)
```

并按 attention 后端选择 stride 顺序（NHD 便于按 block 取到所有层连续地址，HND 便于按 head 批量访问）。**实测块大小变化**：

| 模型 | 旧块大小 | 新块大小 |
|------|----------|----------|
| Llama-3.1-8B | 32 KB | 2 MB |
| Qwen3-32B (TP=2) | 16 KB | 2 MB |
| DeepSeek-V2-Lite (bs=64) | 72 KB | 1.9 MB |
| Qwen3-8B | 28 KB | 1.97 MB |

Offloading Connector 吞吐因此**提升一个数量级**。

### 2.4 块管理组件

```
KVCacheManager            ← 顶层，对接 Scheduler
  └─ KVCacheCoordinator     ← 多 KV Cache Group 协调
       └─ SingleTypeKVCacheManager  ← 单类型块分配
            └─ BlockPool           ← 物理块池
                 └─ FreeKVCacheBlockQueue  ← LRU 双向链表
```

`KVCacheSpec` 按架构分：`FullAttentionSpec`（含 GQA，最常见）、`MLAAttentionSpec`（DeepSeek-V3，K/V 合并为 latent）、`SlidingWindowSpec`、`MambaSpec` 等。

每个物理块由 `KVCacheBlock` 表示（只存元数据，数据在 GPU Tensor 里）：

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int          # 物理块编号
    ref_cnt: int = 0       # 引用计数，0 表示空闲可回收
    _block_hash: ...       # 满块 + 开启 prefix caching 才有值
    prev_free_block / next_free_block   # LRU 双向链表指针
    is_null: bool = False  # block_id=0 的占位块
```

`BlockPool` 用 `FreeKVCacheBlockQueue` 维护 LRU：分配从**头部**弹出最旧块，释放追加到**尾部**，命中前缀时 `touch()` 以 O(1) 从链表中间摘除。

### 2.5 Prefix Caching 的链式哈希

块哈希**链式依赖前一块**：

```python
hash_block_tokens(hash_fn, parent_block_hash, curr_block_token_ids, extra_keys)
```

因此两个块哈希相同，当且仅当它们位置相同且**全部前缀 token 一致**。多模态、LoRA、`cache_salt` 通过 `extra_keys` 参与哈希，避免错误命中。

完整链路：`get_computed_blocks()` 查命中 → `touch()` 保护 → `allocate_slots()` 补新块 → 推理 → `cache_full_blocks()` 注册满块哈希 → 请求结束 `free_blocks()`（**保留哈希**，空闲不足时才驱逐）。

### 2.6 KV Connector

统一抽象层，负责实例间 KV 的保存、加载与传输，是 **PD 分离**与 **KV Offload** 的底座（如 LMCache、Mooncake 等实现）。

## 3. AttentionStore：从"迁移"升级为"系统"

### 3.1 三个必须解决的问题

光把 KV 卸载到 CPU/SSD 不够，生产落地会撞上三堵墙：

1. **调度盲区**：调度器看不见缓存分布，请求常被分到没缓存的节点，触发完整 prefill 重算，offload 的收益被完全抵消
2. **搬运低效**：HBM / DRAM / SSD 之间跨介质搬移缺乏针对性优化，传输延迟吃掉复用收益
3. **缓存随进程消亡**：缓存与推理进程强耦合，进程重启或升级，缓存立刻失效

### 3.2 全局索引 + 缓存感知调度

- **全局 KV 索引**：汇聚各节点的 KV Block 元数据（`BlockHash`、所在介质 HBM/DRAM/SSD），实时捕捉创建/销毁事件，形成 Host → Blocks 映射
- **调度决策升级**：从"只看资源健康"变成"资源 + 缓存协同打分"——先收敛到高命中率节点集合，再结合命中长度和介质读取效率综合打分

最终目标从"是否可用"变成"**是否最优**"。

### 3.3 多级缓存与传输优化

数据流转：请求先查显存 → 未命中则通过节点间池化从其他节点的主机内存迁移 → 仍未命中才现算；prefill 产出的 KV 传 decode 节点并异步回写 DRAM/SSD；decode 新增 KV 增量异步回写。

| 优化 | 做法 | 效果 |
|------|------|------|
| **昆仑芯原生适配** | 调 XPU 原生 API 优化数据搬运、缓存访问与执行调度；统一硬件抽象层 | 跨硬件平滑运行 |
| **读取加速** | 高速/低速介质**并行**发起传输（改串行为并行）；共享内存标大页；全生命周期锁页 | DRAM→HBM 通信效率较基线 **提升 4 倍** |
| **传输加速** | C++ SDK 把序列化/打包/跨节点传输从主进程剥离到异步线程池；回写与传输拆分并行 | KV 传输与模型计算形成流水线 |

此外，AttentionStore **以独立进程运行**，与推理引擎解耦——进程重启、故障恢复、版本升级时 KV 依然保留，自身重启后可通过本地索引表快速恢复。

### 3.4 实测数据（DeepSeek R1 671B / 昆仑芯 P800）

环境：2 台 prefill 节点，TP4 / DP4。

| 场景 | 收益 |
|------|------|
| 上下文 > 8K | TTFT 稳定优化 **50% ~ 80%** |
| 多轮对话 | 整体吞吐提升 **5.4 倍** |
| 64K 长上下文 | 相比默认 Chunk-Prefill，TTFT 降低 **6.2 倍** |

## 4. 三者对比

| 维度 | SGLang | vLLM | AttentionStore |
|------|--------|------|----------------|
| **核心抽象** | 两级内存池 + Radix Tree | 分页虚拟内存 + LRU 块池 | 集群级全局索引 + 多级介质 |
| **复用粒度** | token 级（前缀树） | block 级（链式哈希） | KV Block 全局可寻址 |
| **显存治理** | 池化 + 分层（HiCache） | 分页消除碎片 | HBM / DRAM / SSD 三级 |
| **解决的主要问题** | 前缀复用率、长上下文容量 | 显存碎片、调度效率 | 集群调度命中率、进程解耦 |
| **典型依赖** | 3FS / Mooncake / NIXL | KV Connector（LMCache 等） | 分布式 FS + RDMA |

一句话区分：**SGLang 和 vLLM 解决"单机怎么管好显存"，AttentionStore 解决"集群怎么让请求找到已有缓存"。**

## 5. 与已有文章的衔接

- **sys9** 是原理与分类，本篇是它的工程落地。
- **sys1（P900 + HiCache 的 NUMA/PCIe/NIC）** 讲硬件拓扑，正好解释本篇 HiCache 三层搬运为什么必须做并行与锁页优化。
- **sys8（PD 分离）** 里的 KV 传输代价（70B·4k token·FP16 ≈ 600MB），正是 KV Connector 与 AttentionStore 传输优化要解决的问题。
- **fw2（vLLM 内部）** 讲 PagedAttention 的概念层，本篇补到 Block 管理与跨层布局的实现层。
- **ag1** 指出 Agentic 负载把 KV Cache 从显存问题变成调度问题——AttentionStore 的全局索引正是这一判断的工业印证。
