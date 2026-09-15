---
title: 推理系统基础设施手记（十一）：SGLang HiCache L2 在昆仑芯 P800/P900 上的适配与调优实战
description: 把 SGLang HiCache 的 L2（Host DRAM 前缀缓存）真正跑快：从分层架构与净收益公式出发，落到昆仑芯 P800/P900 上的布局选择（page_first_direct → layer_first）、DMA 段数、异步搬运、NUMA 亲和、透明大页、PCIe IDO 保序、写入策略语义修复与 Ghost List、TP 组共享 Host KV，最后给出端到端 TTFT 从 28.83s 降到 18.45s（叠加 cache-aware + dp-aware 路由降到 14.47s）的实测收益。
pubDate: 2026-09-15
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys11-hicache-l2-kunlunxin-tuning
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

> **分层缓存收益能否成立，不但取决于命中率能做多高，也取决于跨层传输代价能压多低。**

HiCache 把前缀 KV 从显存（L1）扩展到 Host DRAM（L2）、再到外部存储（L3），本质是用一层更慢但大得多的存储换前缀复用率。但昆仑芯上这套机制默认是「负收益」——只要跨层搬运的额外开销盖过命中省下的计算，整体就变慢。本文沿着 sys1 的 NUMA × PCIe × NIC 拓扑，把 L2 在 P800/P900 上从「能跑」调到「跑得快」的全过程拆开。

## 1. 为什么需要分层前缀缓存

### 1.1 问题：L1 容量与前缀复用之间的矛盾

大模型推理的 Prefill 阶段存在大量重复计算。Agent、多轮对话这类流量里，system prompt、tool 定义、历史上下文在请求之间大量重叠，前缀重复度很高。Radix Cache 的作用就是复用这部分计算：把算过的前缀 KV 留在显存里，下次命中就跳过重算。

但 Radix Cache 驻留在 XPU HBM 上，容量被显存死死卡住——高并发下，缓存在几分钟内就被新请求挤占一空。显存不可能再扩，而显存之外还有两层被闲置的存储：**单机数百 GB 到 TB 级的 Host DRAM**，以及**集群级的外部存储**。HiCache 要解决的就是如何把前缀缓存扩展到这两层。

### 1.2 HiCache 的分层设计

HiCache 的核心思路是把「前缀的索引结构」和「KV 数据的存放位置」拆开：Radix 树只负责回答「这段 token 序列对应树上哪个节点」，节点的 KV 数据具体放在哪一层是另一件事。

<div class="fig">
<svg viewBox="0 0 680 432" width="100%" font-size="13" font-weight="400" role="img" aria-label="HiCache 三层前缀缓存架构">
<title>HiCache 三层前缀缓存架构</title>
<defs>
<marker id="ar1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<text x="340" y="34" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#1a1a1a">HiCache 三层前缀缓存架构</text>
<text x="340" y="54" text-anchor="middle" dominant-baseline="central" fill="#374151">Radix 树只管索引，KV 数据按层存放</text>
<rect x="200" y="78" width="280" height="60" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text x="340" y="102" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#26215C">L1 · XPU HBM</text>
<text x="340" y="122" text-anchor="middle" dominant-baseline="central" fill="#3C3489">10 GB 级 · 直接命中 · 无额外开销</text>
<rect x="200" y="168" width="280" height="60" rx="10" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="340" y="192" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#04342C">L2 · Host DRAM</text>
<text x="340" y="212" text-anchor="middle" dominant-baseline="central" fill="#085041">100 GB – TB 级 · PCIe 传输 (D2H / H2D)</text>
<rect x="200" y="258" width="280" height="60" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="340" y="282" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#042C53">L3 · 外部存储</text>
<text x="340" y="302" text-anchor="middle" dominant-baseline="central" fill="#0C447C">TB – PB 级 · 网络+IO · 经 L2 中转</text>
<path d="M312 138 L312 168" stroke="#5F5E5A" stroke-width="1.5" fill="none" marker-end="url(#ar1)"/>
<path d="M368 168 L368 138" stroke="#5F5E5A" stroke-width="1.5" fill="none" marker-end="url(#ar1)"/>
<text x="300" y="154" text-anchor="end" dominant-baseline="central" fill="#444441">淘汰下沉</text>
<text x="380" y="154" text-anchor="start" dominant-baseline="central" fill="#444441">命中加载</text>
<path d="M312 228 L312 258" stroke="#5F5E5A" stroke-width="1.5" fill="none" marker-end="url(#ar1)"/>
<path d="M368 258 L368 228" stroke="#5F5E5A" stroke-width="1.5" fill="none" marker-end="url(#ar1)"/>
<text x="300" y="244" text-anchor="end" dominant-baseline="central" fill="#444441">evict</text>
<text x="380" y="244" text-anchor="start" dominant-baseline="central" fill="#444441">load</text>
<rect x="120" y="346" width="440" height="64" rx="10" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="370" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#2C2C2A">净收益 = 命中率提升省下的计算 − 跨层搬运额外开销</text>
<text x="340" y="392" text-anchor="middle" dominant-baseline="central" fill="#444441">命中率↑ 与 传输代价↓ 两条主线共同决定收益正负</text>
</svg>
<p class="cap">图 1：L1/L2/L3 三层。一个节点被 L1 淘汰时，KV 可下沉到 L2 而非丢弃；下次前缀命中再从 L2 加载回 L1。L3 是 L2 的下一级兜底，且可让多实例共享同一份缓存。</p>
</div>

引入 HiCache 后，Radix 树本身不用改：token 序列仍对应树上一个确定节点，前缀匹配、引用计数、淘汰逻辑完全不变。变的只是节点上那份 KV 数据存在哪一层。这就决定了一个贯穿全文的评价标准：

```
净收益 = 命中率提升省下的计算时间 − 跨层数据移动带来的额外开销
```

后续所有变更，要么提升命中率（增大 L2 空间 / 提升 L2 利用效率 / 引入 L3），要么降低跨层传输开销——而跨层开销又可拆成四块：① 跨层传输带宽；② 布局转换开销；③ 算子下发调用耗时；④ 额外传输对推理主流程的串扰。

### 1.3 昆仑芯上的适配面

在昆仑芯 P800/P900 上落地 HiCache，主要处理三个层次的问题：

- **算子层**：上游跨层搬运 kernel 是 CUDA 实现的，需要在昆仑芯上补一套功能相同的算子。不适配就跑不起来。
- **框架层**：传输在什么线程提交、按什么粒度切分、哪些数据该写、写几份。不影响能否跑通，但直接决定净收益正负。
- **主机层**：NUMA 拓扑、页表规模、PCIe 保序策略。在 CUDA 生态里通常被驱动和硬件掩盖，在昆仑芯上会直接表现为性能问题。

## 2. L2 适配与调优

### 2.1 布局选择：page_first_direct → layer_first

HiCache L2 的 KV 布局主要有三种：`layer_first`、`page_first`、`page_first_direct`。初版实现选用了 `page_first_direct`——整页连续、单 layer 内 page_size 个 token 连续，适合 direct（DMA）式搬运。

快速验证的结果却令人沮丧：关闭 HiCache 时首条请求 5.95s，开启后变成 15.44s，且 Replay 请求（全命中）也略有提升。说明代价与写回的数据规模直接挂钩，而非一次性固定开销。

真正的问题在 **DMA 提交次数**。`page_first_direct` 的 dims 是 `(page_num, layer_num, page_size, 1, kv_dim)`，最外层是 page。Host 侧同 page 内不同 layer 目标地址相邻可合并，但不同 page 的同一 layer 隔着 `layer_num × page_size` 永远无法合并；device 侧不同 layer 是不同 buffer，天然不连续。两边限制叠加，段数固定在 `page_num × layer_num`。

换成 `layer_first`（dims `(layer, size, 1, kv_dim)`）后，同 layer 内不同 page 目标地址连续时可整段合并。以 32K tokens、page_size=64、78 层为例：512 page × 78 层 = 39,936 段；换成 layer_first 后理想情况是 **78 次连续搬运**，段数下降约两个数量级。每一段都要独立算地址并下发一次 DMA 描述符，段数本身就决定单层传输能不能跑到接近顺序带宽。

<div class="fig">
<svg viewBox="0 0 680 400" width="100%" font-size="13" font-weight="400" role="img" aria-label="KV 布局与 DMA 段数">
<title>KV 布局 vs DMA 段数</title>
<defs>
<marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<text x="340" y="34" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#1a1a1a">KV 布局决定 DMA 提交段数</text>
<text x="340" y="54" text-anchor="middle" dominant-baseline="central" fill="#374151">示例: 32K tokens · page_size 64 · 78 层 → 512 page</text>
<rect x="40" y="80" width="270" height="200" rx="10" fill="#FAECE7" stroke="#993C1D" stroke-width="0.5"/>
<text x="175" y="104" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#4A1B0C">page_first_direct</text>
<text x="175" y="124" text-anchor="middle" dominant-baseline="central" fill="#712B13">page 为最外维 · 页内跨 layer</text>
<rect x="70" y="142" width="48" height="120" rx="4" fill="#F0997B" stroke="#993C1D" stroke-width="0.5"/>
<rect x="124" y="142" width="48" height="120" rx="4" fill="#F0997B" stroke="#993C1D" stroke-width="0.5"/>
<rect x="178" y="142" width="48" height="120" rx="4" fill="#F0997B" stroke="#993C1D" stroke-width="0.5"/>
<rect x="232" y="142" width="48" height="120" rx="4" fill="#F0997B" stroke="#993C1D" stroke-width="0.5"/>
<text x="153" y="284" text-anchor="middle" dominant-baseline="central" fill="#712B13">每 page 切 layer_num 段</text>
<text x="175" y="318" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#4A1B0C">段数 = page_num × layer_num</text>
<text x="175" y="342" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#993C1D">512 × 78 = 39,936</text>
<path d="M330 190 L350 190" stroke="#5F5E5A" stroke-width="1.5" fill="none" marker-end="url(#ar2)"/>
<text x="340" y="176" text-anchor="middle" dominant-baseline="central" fill="#444441">↓ 约 2 个数量级</text>
<rect x="370" y="80" width="270" height="200" rx="10" fill="#EAF3DE" stroke="#3B6D11" stroke-width="0.5"/>
<text x="505" y="104" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#173404">layer_first</text>
<text x="505" y="124" text-anchor="middle" dominant-baseline="central" fill="#27500A">layer 为最外维 · page 合并</text>
<rect x="400" y="146" width="210" height="22" rx="4" fill="#97C459" stroke="#3B6D11" stroke-width="0.5"/>
<rect x="400" y="176" width="210" height="22" rx="4" fill="#97C459" stroke="#3B6D11" stroke-width="0.5"/>
<rect x="400" y="206" width="210" height="22" rx="4" fill="#97C459" stroke="#3B6D11" stroke-width="0.5"/>
<rect x="400" y="236" width="210" height="22" rx="4" fill="#97C459" stroke="#3B6D11" stroke-width="0.5"/>
<text x="505" y="284" text-anchor="middle" dominant-baseline="central" fill="#27500A">同 layer 内 page 连续合并</text>
<text x="505" y="318" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#173404">段数 = layer_num</text>
<text x="505" y="342" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#3B6D11">78 段</text>
<rect x="120" y="356" width="440" height="34" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="373" text-anchor="middle" dominant-baseline="central" fill="#2C2C2A">两端布局一致 → 无需维度交换 → 净收益转正 (冷启 10.7s → 7.9s)</text>
</svg>
<p class="cap">图 2：布局选择直接决定 DMA 提交段数。layer_first 让 host 与 device 两侧布局一致，算子内部不再做维度交换，段数从 page_num×layer_num 降到 layer_num。</p>
</div>

主路线由此定型：**layer_first + 双向异步**。Host 布局换成 layer_first 后，搬运算子换成同布局的 `transfer_kv_per_layer_mla`（H2D 逐层）/ `transfer_kv_all_layer_mla`（D2H 整体），冷启动首条请求从 10.74s 降到 7.91s。

### 2.2 把传输提交挪出调度主线程

初版实现里，H2D/D2H 搬运算子的 CPU 侧提交（Kernel Launch）是内联在调度主线程的 `cache_unfinished_req` 路径里的。以 D2H 为例，单个 `transfer_kv_all_layer_direct_lf_pf` 的 CPU 侧提交典型耗时 1.3s，且提交不返回调度循环就无法继续。一次请求要写回多个 chunk，足以解释上面多出来的 9 秒。

解法很自然：把 D2H 与 H2D 各起一个 daemon 线程消费一个队列。`start_writing()` 不再内联提交，而是把请求投递到 backup 队列，由 backup 线程在专用 stream 上提交；`start_loading()` 同理投递到 load 队列。异步化后主线程不再被提交阻塞，但实测非命中场景仍比不开 HiCache 高约 26%——说明代价不止在「谁来提交」。

### 2.3 真正的问题：DMA 提交次数

异步化把提交挪出主线程，但一次写回要搬运的数据量并没有减少。问题出在 DMA 提交次数上（见 2.1）。换成 layer_first 后段数下降约两个数量级，单层传输才能跑到接近顺序带宽。这一步是让 L2 主路线从「负收益」翻正的关键。

### 2.4 Host 内存的物理形态

异步化解决了「谁来提交」，但剩下两类代价来自 Host 内存本身的物理形态：

- **NUMA 亲和绑定**：P800 单机多卡分布在不同 NUMA Node 上。Host KV Pool 若落在远端 Node，每次 D2H/H2D 都跨 NUMA，并与本地 RDMA 注册内存争抢带宽。做法是指定每张卡对应的 NUMA Node，关掉框架自动探测。
- **透明大页（THP）**：一块 100GB 的 Host Buffer 若用 4KB 页，需要约 2600 万个页表项，同时加重 CPU 页表、IOMMU 和设备 DMA 三条地址翻译路径。改用 2MB 大页把映射条目降约 512 倍。实现上走匿名 mmap + `madvise(MADV_HUGEPAGE)`，且 `madvise` 必须在 `cudaHostRegister` 之前（注册会 Pin 住页面，之后 THP 无法再合并）。
- **空闲链表改 LIFO**：默认 FIFO 释放的 slot 排到队尾，分配指针在整个 pool 上循环扫描；改成 LIFO 让最近释放的 slot 优先被复用，写入集中在少数热页面上，带来纯内存局部性提升（不改变命中率与淘汰策略）。

实测验证：注册内存规模本身就是成本——80GB → 15GB，退化从 +20.6% 单调降到 +8.8%；把 Host KV pool 按 XPU 拓扑绑到本地 NUMA 后，退化基本消除（+11.5% → -2.1%）。

### 2.5 平台侧 PCIe 保序策略：IDO

海光平台有个硬件约束：当前 128 条高速 PCIe lanes 中，每颗 CPU 的 64 条固定用于 CPU 互联，剩余 64 条只能划出 4 条 PCIe x16。如果每张 P800 独占一个 x16，主网卡就没 PCIe 资源可用——因此**两颗 XPU 共享一个 PCIe x16 的上行带宽**。这带来一个直接后果：若严格按 P800 NUMA 拓扑一对一绑定，整台机器的 L2 只能用一半内存（典型 1.5T 机器约 750GB，8 卡均分再扣组件占用，每卡最多约 75GB），成为容量瓶颈。

若强制用到所有 NUMA 节点（如 H3C 拓扑 `1 1 3 3 5 5 7 7`，SGLang 设为 `0 1 2 3 4 5 6 7`），虽能用到全部内存，但 D2H 带宽严重下降。根因是：同一 PCIe Switch 下的两颗 XPU 对内存的写操作本应相互独立，但**严格保序策略让它们互相阻塞**。

<div class="fig">
<svg viewBox="0 0 680 380" width="100%" font-size="13" font-weight="400" role="img" aria-label="NUMA 拓扑与 PCIe 保序">
<title>NUMA 拓扑与 PCIe 保序</title>
<defs>
<marker id="ar3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<text x="340" y="34" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#1a1a1a">NUMA 拓扑与 PCIe 保序 (海光 P800)</text>
<text x="340" y="54" text-anchor="middle" dominant-baseline="central" fill="#374151">两颗 XPU 共享一个 Switch 的 x16 上行带宽</text>
<rect x="180" y="80" width="320" height="120" rx="10" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="100" text-anchor="middle" dominant-baseline="central" fill="#2C2C2A">PCIe Switch (SW0)</text>
<rect x="60" y="230" width="200" height="64" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text x="160" y="254" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#26215C">XPU0 → NUMA0</text>
<text x="160" y="274" text-anchor="middle" dominant-baseline="central" fill="#3C3489">D2H 写事务 ID=A</text>
<rect x="420" y="230" width="200" height="64" rx="10" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="520" y="254" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#04342C">XPU1 → NUMA1</text>
<text x="520" y="274" text-anchor="middle" dominant-baseline="central" fill="#085041">D2H 写事务 ID=B</text>
<rect x="280" y="230" width="120" height="64" rx="10" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="340" y="254" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#042C53">NIC1</text>
<text x="340" y="274" text-anchor="middle" dominant-baseline="central" fill="#0C447C">RDMA 争抢</text>
<path d="M160 294 L260 200" stroke="#534AB7" stroke-width="1.5" fill="none" marker-end="url(#ar3)"/>
<path d="M520 294 L420 200" stroke="#0F6E56" stroke-width="1.5" fill="none" marker-end="url(#ar3)"/>
<path d="M340 294 L340 200" stroke="#185FA5" stroke-width="1.5" fill="none" marker-end="url(#ar3)"/>
<rect x="60" y="316" width="560" height="44" rx="8" fill="#FAECE7" stroke="#993C1D" stroke-width="0.5"/>
<text x="340" y="334" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#4A1B0C">严格保序: ID=A 与 ID=B 互相阻塞 → D2H 带宽↓</text>
<text x="340" y="352" text-anchor="middle" dominant-baseline="central" fill="#712B13">改 IDO (ID-Based Ordering): 不同来源 ID 不再互等 → 带宽恢复</text>
</svg>
<p class="cap">图 3：两颗 XPU 共享同一 Switch 的 x16 上行。严格 PCIe 保序让两路 D2H 写事务互相阻塞；改为 ID-Based Ordering (IDO) 后，不同来源 ID 的写事务不再互相等待，D2H 带宽回升符合预期。</p>
</div>

解法是修改 CPU 寄存器，把 PCIe RC 端的通信策略调整为**基于 ID 的保序（IDO）**，让不同来源 ID 的写事务不再互相等待。适用于需要 NUMA 一对一绑定的场景，开启后性能回升符合预期。

### 2.6 写入策略语义修复 + Ghost List

L1 → L2 的写入策略默认有 `write_back` / `write_through` / `write_through_selective`（仅当 L1 命中次数 ≥ 2 才 backup，过滤一次性前缀）。但在 PD 分离的 Prefill 实例上，同一请求会插入 Radix 树两次（`cache_unfinished_req` 一次、`cache_finished_req` 一次），单个请求就把 `hit_count` 增长到 2，于是 `write_through_selective` 静默退化成 `write_through`，过滤效果完全消失——这是个语义非预期问题。

修复有两个机制：

1. **抑制同请求重复计数**：finished 阶段跳过 `hit_count` 累加，让 `hit_count` 重新表示「访问过该前缀的不同请求数」。
2. **Ghost List**：修复后引入新问题——若两次复用间隔超过该前缀在 L1 的驻留时间，第二次到来时节点已被 L1 淘汰、`hit_count` 归零，L2 命中率下降。Ghost List 用额外且容量可控的 Hash Map 存被 L1 淘汰的 page hash（粒度是 page 而非 node，因为不同请求 chunk 边界、部分命中长度都不同；page 边界哈希从 root 累积算到该边界，同时编码内容和位置）。重建时从后往前扫，找到的第一个命中即最长复用前缀，只把被证明复用的那段写到 L2。key 是 64-bit 整数，用 OrderedDict 实现 O(1) 增删查，每条约 120 字节。

<div class="fig">
<svg viewBox="0 0 680 420" width="100%" font-size="13" font-weight="400" role="img" aria-label="写入策略 Ghost List TP 组共享">
<title>写入策略 / Ghost List / TP 组共享</title>
<rect x="40" y="80" width="600" height="96" rx="10" fill="#FAEEDA" stroke="#854F0B" stroke-width="0.5"/>
<text x="60" y="104" text-anchor="start" dominant-baseline="central" font-weight="500" fill="#412402">1 · write_through_selective 语义修复</text>
<text x="60" y="128" text-anchor="start" dominant-baseline="central" fill="#633806">PD 分离下同一请求插树两次 → hit_count 虚高 → 静默退化为 write_through</text>
<text x="60" y="152" text-anchor="start" dominant-baseline="central" fill="#633806">修复: finished 阶段跳过累加 → hit_count 重新表示"不同请求数"</text>
<rect x="40" y="188" width="600" height="96" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text x="60" y="212" text-anchor="start" dominant-baseline="central" font-weight="500" fill="#26215C">2 · Ghost List</text>
<text x="60" y="236" text-anchor="start" dominant-baseline="central" fill="#3C3489">page 粒度哈希 (64-bit) · OrderedDict LRU · O(1) 增删查 · 每条约 120B</text>
<text x="60" y="260" text-anchor="start" dominant-baseline="central" fill="#3C3489">从后往前扫命中即最长复用前缀 · 只写被证明复用的那段 · 有界可配</text>
<rect x="40" y="296" width="600" height="96" rx="10" fill="#EAF3DE" stroke="#3B6D11" stroke-width="0.5"/>
<text x="60" y="320" text-anchor="start" dominant-baseline="central" font-weight="500" fill="#173404">3 · TP 组共享 Host KV</text>
<text x="60" y="344" text-anchor="start" dominant-baseline="central" fill="#27500A">MAP_SHARED buffer · 仅 tp0 执行 D2H · 所有 rank 同 indices 读取</text>
<text x="60" y="368" text-anchor="start" dominant-baseline="central" fill="#27500A">MLA/NSA replicated latent 逐 bit 相同 → L2 有效容量 ×8 · D2H 流量 ÷8</text>
</svg>
<p class="cap">图 4：三项软件优化。写入策略语义修复 + Ghost List 减少无效写回；TP 组共享 Host KV 把多份冗余合并成一份，放大有效容量并削减 D2H 流量。</p>
</div>

该策略默认关闭——收益是有条件的：节省的 D2H 带来的 TTFT 下降 与 命中率下降带来的 TTFT 上升 谁占优，取决于 L2 容量与数据集复用距离分布。L2 越紧张，过滤越有价值（挡住一次性前缀，让有限 L2 装「已被证明会复用」的高价值 KV）。

### 2.7 TP 组共享 Host KV

这是单项收益最大的一条。对 MLA/NSA 架构，Host 侧 KV 是一个 replicated latent，每个 TP rank 持有逐 bit 相同的同一份数据（即使开 dp attention，TP≠1 时仍有冗余）。于是 TP=8 意味着：8 份完全相同的 Host 内存副本（L2 有效容量只有名义值 1/8）+ 8 倍 D2H 流量（而 PCIe 带宽、IOMMU/TLB 预算、kernel launch 配额都是按设备或主机共享的）。

机制：一个 TP 组共用一份 `MAP_SHARED` 的 Host Buffer，只有组内 rank 0 执行 D2H 写入，所有 rank 用相同 indices 读取。介质约束决定走 `/dev/shm` + `MADV_HUGEPAGE`（XPU 驱动无法注册 hugetlbfs，但 THP-backed tmpfs 注册正常），注册期间临时切 `MPOL_INTERLEAVE` 覆盖全部在线 node。附带效果是共享模式下 NUMA 绑定从必选变可选（L2 内存本身已跨 node 交织）。正确性依赖隐式不变量：host pool 的 alloc/free 必须在各 rank 上 SPMD 确定一致，并提供可选自检开关做跨 rank 哈希一致性校验。

## 3. 收益汇总

GLM5 Workbuddy 24K–64K、前缀理论复用率 0.82、PD 分离、并发 10 的端到端测试：

| 阶段 | 关键改动 | TTFT | Prefix 命中率 | 相对 Base |
|---|---|---|---|---|
| Base | 关闭 HiCache，仅 L1 Radix | 28.83s | 46.28% | — |
| 主路线定型 | layer_first + kernel backend + 双向异步 | 22.08s | 69.23% | -23.42% |
| + Host 内存形态 | NUMA 亲和 / 大页 / LIFO | 20.86s | 69.23% | -27.66% |
| + TP 组共享 Host KV | 去 N 份冗余，L2 放大到 400GB | 19.12s | 72.79% | -33.69% |
| + L2 容量调优 | 回调到 200GB，命中率饱和 | 18.45s | 71.94% | -36.01% |

<div class="fig">
<svg viewBox="0 0 680 420" width="100%" font-size="13" font-weight="400" role="img" aria-label="端到端 TTFT 收益瀑布">
<title>端到端 TTFT 收益瀑布</title>
<defs>
<marker id="ar5" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<text x="340" y="34" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#1a1a1a">端到端 TTFT 收益瀑布</text>
<text x="340" y="54" text-anchor="middle" dominant-baseline="central" fill="#374151">GLM5 Workbuddy 24K–64K · 前缀复用率 0.82 · 并发 10</text>
<rect x="180" y="78" width="500" height="36" rx="6" fill="#F09595" stroke="#A32D2D" stroke-width="0.5"/>
<text x="188" y="96" text-anchor="start" dominant-baseline="central" fill="#501313">Base · 仅 L1 Radix</text>
<text x="672" y="96" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#501313">28.83s</text>
<rect x="180" y="124" width="383" height="36" rx="6" fill="#F7C1C1" stroke="#A32D2D" stroke-width="0.5"/>
<text x="188" y="142" text-anchor="start" dominant-baseline="central" fill="#501313">+ layer_first + 双向异步</text>
<text x="555" y="142" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#501313">22.08s · -23.4%</text>
<rect x="180" y="170" width="362" height="36" rx="6" fill="#FAC775" stroke="#854F0B" stroke-width="0.5"/>
<text x="188" y="188" text-anchor="start" dominant-baseline="central" fill="#412402">+ Host 内存形态 (NUMA/大页/LIFO)</text>
<text x="534" y="188" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#412402">20.86s · -27.7%</text>
<rect x="180" y="216" width="332" height="36" rx="6" fill="#C0DD97" stroke="#3B6D11" stroke-width="0.5"/>
<text x="188" y="234" text-anchor="start" dominant-baseline="central" fill="#173404">+ TP 组共享 Host KV</text>
<text x="504" y="234" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#173404">19.12s · -33.7%</text>
<rect x="180" y="262" width="320" height="36" rx="6" fill="#97C459" stroke="#3B6D11" stroke-width="0.5"/>
<text x="188" y="280" text-anchor="start" dominant-baseline="central" fill="#173404">+ L2 容量调优 (200GB)</text>
<text x="492" y="280" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#173404">18.45s · -36.0%</text>
<rect x="180" y="308" width="251" height="36" rx="6" fill="#639922" stroke="#3B6D11" stroke-width="0.5"/>
<text x="188" y="326" text-anchor="start" dominant-baseline="central" fill="#ffffff">+ cache-aware + dp-aware 路由</text>
<text x="423" y="326" text-anchor="end" dominant-baseline="central" font-weight="500" fill="#ffffff">14.47s · -42.6%</text>
<line x1="180" y1="64" x2="180" y2="360" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="174" y="96" text-anchor="end" dominant-baseline="central" fill="#444441">0</text>
<rect x="60" y="364" width="560" height="34" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text x="340" y="381" text-anchor="middle" dominant-baseline="central" fill="#2C2C2A">命中率 46%→72% 是主体收益；-23%→-36% 的 12.6pt 全来自传输路径优化</text>
</svg>
<p class="cap">图 5：端到端 TTFT 瀑布。从 28.83s 降到 18.45s；再叠加 cache-aware + dp-aware 路由，TTFT 进一步降到 14.47s。</p>
</div>

两点关键结论：

- **收益来自两方面**：命中率从 46.28% 提升到约 72%（+25pt）是主体，来自 L2 把有效缓存容量放大一个量级；而从 -23.42% 到 -36.01% 的 12.6pt，全部来自传输路径本身的优化——没有改任何模型计算，也没有提升命中率上限，纯粹是把「跨层数据移动的额外开销」压下去的结果。这正好印证开篇的净收益公式。
- **容量不是越大越好**：最后一行是容量下调带来的收益——命中率基本持平时，较小的 L2 反而更快。因为注册内存规模本身就是成本。合理做法是**从小到大逐档试**，观察命中率边际增益，进入平台期就停；超出平台期的容量只会增加注册内存规模，反过来拖慢 DMA 与 RDMA 路径。

L2 全命中的拟合公式：`T(n) = 779.1ms + 13.539us · n`，其中 779ms ≈ 735ms（两种模式共有，部署 floor） + 44ms（L2 路径独有）。H2D 实测带宽约 7.40 GB/s。平衡点约在 3K token：>3K token 的命中才取得收益。

## 4. 总结与展望

HiCache L2 在昆仑芯 P800/P900 上已具备实际应用价值，形成以**布局优化、异步传输、NUMA 亲和、TP 组共享**为核心的实现方案。但本文所有数值都是特定环境下的测量结果，不宜直接外推到其他平台或负载——换平台或换负载时，建议按同样的顺序重新归因：先确认主机侧（保序策略、NUMA、大页）没有留下明显单次传输代价，再按容量逐档试探命中率边际增益，最后才调整写入策略这类与数据集强相关的开关。

未来方向是 **L3**：单纯靠 L2 的 Host DRAM 空间扩展，在更长上下文与更高效率要求下仍不够。落地最快、成本最低的方案是把 PD 分离架构中 **D 节点的 Host DRAM 通过 Mooncake 池化**——D 节点本身承担 Decode，Host DRAM 长期低利用，纳入池化无需新增硬件即可为 Prefill 实例提供可观 L3 容量，且介质仍是 DRAM，跨层代价明显低于落盘；同时把单机私有缓存变为集群级共享缓存，命中率不再受单实例流量局部性限制。仍待验证的关键点是 RDMA 传输与 L2 写回的相互干扰，以及池化元数据在大规模实例下的一致性开销。

## 5. 系列导航

- **sys1（P900 + HiCache 的 NUMA × PCIe × NIC 数据链路）** 讲硬件拓扑，正是本篇 L2 搬运为什么必须做并行、锁页与 IDO 优化的背景。
- **sys9 / sys10（KV Cache 全景与工程实现）** 讲 KV Cache 的原理、分类与 SGLang / vLLM / AttentionStore 的实现，本篇是其昆仑芯 L2 适配的实战补完。
- 一句话串起来：**sys1 看清链路 → sys9/sys10 讲清原理与引擎 → sys11 把 L2 在昆仑芯上真正跑快。**
