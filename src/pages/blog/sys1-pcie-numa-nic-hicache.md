---
title: 推理系统基础设施（一）：P900 + HiCache 下的 NUMA × PCIe × NIC 数据链路
description: 把 PCIe、NUMA、NIC 放到同一条数据链路里理解。针对 P900 + HiCache 场景，用「拓扑图 + 五层优化图」两张图讲清加速卡怎么和 Host KV 传数据、Host KV 该放在哪个 CPU 内存节点、以及做 PD/Mooncake/RDMA 时 NIC 为什么要一起看。
pubDate: 2026-08-26
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys1-pcie-numa-nic-hicache
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么 P900 + HiCache 必须看 NUMA / PCIe / NIC

HiCache 把 KV Cache 放到 **Host DRAM** 上，当成一张「主机侧大 KV 池」。当某次请求命中 Host KV 时，它**并不是直接就在 P900 上被用掉**——通常还要走一层：

```
Host KV (Host DRAM)
     │
     │  H2D (Host → Device)
     ↓
   PCIe
     │
     ↓
   P900
     │
     ↓
   HBM (Device Memory)
```

也就是说，**Host KV 命中率高，不代表性能一定高**。因为命中之后还有 PCIe 搬运成本。如果 PCIe 带宽不够 / 延迟高 / 或者 KV 跨 NUMA 节点访问，就可能出现：

```
Host KV Hit ↑
     ↓
   PCIe 搬运
     ↓
TTFT ↑ / TPS ↓
```

所以在这个场景里，**PCIe、NUMA、NIC 必须放到同一条链路里一起看**。下面先用一张拓扑图把硬件关系画清楚，再用一张五层优化图把「怎么调」落到操作上。

## 1. 拓扑图：一台服务器里到底发生了什么

<div class="fig">
<svg viewBox="0 0 680 510" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="P900 HiCache 服务器 NUMA×PCIe×NIC 拓扑">
  <rect x="0" y="0" width="680" height="510" fill="none"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">P900 + HiCache 服务器拓扑：NUMA × PCIe × NIC</text>

  <!-- chassis -->
  <rect x="20" y="40" width="640" height="420" rx="14" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>

  <!-- NUMA 0 -->
  <rect x="36" y="64" width="276" height="372" rx="10" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5"/>
  <text x="174" y="86" font-size="13" font-weight="700" fill="#047857" text-anchor="middle">NUMA Node 0</text>

  <!-- NUMA 2 -->
  <rect x="368" y="64" width="276" height="372" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="506" y="86" font-size="13" font-weight="700" fill="#1d4ed8" text-anchor="middle">NUMA Node 2</text>

  <!-- ===== NUMA 0 components ===== -->
  <rect x="72" y="104" width="208" height="40" rx="6" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="176" y="129" font-size="12.5" fill="#b45309" text-anchor="middle">PCIe Root Complex</text>

  <rect x="72" y="160" width="128" height="54" rx="6" fill="#fef2f2" stroke="#ef4444"/>
  <text x="136" y="184" font-size="12.5" font-weight="700" fill="#b91c1c" text-anchor="middle">P900</text>
  <text x="136" y="202" font-size="11" fill="#b91c1c" text-anchor="middle">HBM</text>

  <rect x="212" y="160" width="68" height="54" rx="6" fill="#faf5ff" stroke="#a855f7"/>
  <text x="246" y="190" font-size="11.5" fill="#7e22ce" text-anchor="middle">NIC</text>
  <text x="246" y="206" font-size="10" fill="#7e22ce" text-anchor="middle">RDMA</text>

  <rect x="72" y="300" width="208" height="64" rx="6" fill="#f0fdf4" stroke="#22c55e"/>
  <text x="176" y="328" font-size="12.5" fill="#15803d" text-anchor="middle">Host DRAM</text>
  <text x="176" y="348" font-size="12" font-weight="700" fill="#15803d" text-anchor="middle">Host KV (HiCache)</text>

  <!-- NUMA 0 arrows (local, green) -->
  <line x1="176" y1="144" x2="138" y2="158" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>
  <line x1="176" y1="144" x2="244" y2="158" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>
  <line x1="140" y1="214" x2="172" y2="298" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>
  <line x1="246" y1="214" x2="180" y2="298" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>
  <text x="104" y="156" font-size="10" fill="#15803d">PCIe</text>
  <text x="252" y="156" font-size="10" fill="#15803d">PCIe</text>
  <text x="86" y="262" font-size="10.5" fill="#15803d">H2D / D2H</text>
  <text x="252" y="262" font-size="10.5" fill="#15803d">RDMA</text>

  <!-- ===== NUMA 2 components (shift +332) ===== -->
  <rect x="404" y="104" width="208" height="40" rx="6" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="508" y="129" font-size="12.5" fill="#b45309" text-anchor="middle">PCIe Root Complex</text>

  <rect x="404" y="160" width="128" height="54" rx="6" fill="#fef2f2" stroke="#ef4444"/>
  <text x="468" y="184" font-size="12.5" font-weight="700" fill="#b91c1c" text-anchor="middle">P900</text>
  <text x="468" y="202" font-size="11" fill="#b91c1c" text-anchor="middle">HBM</text>

  <rect x="544" y="160" width="68" height="54" rx="6" fill="#faf5ff" stroke="#a855f7"/>
  <text x="578" y="190" font-size="11.5" fill="#7e22ce" text-anchor="middle">NIC</text>
  <text x="578" y="206" font-size="10" fill="#7e22ce" text-anchor="middle">RDMA</text>

  <rect x="404" y="300" width="208" height="64" rx="6" fill="#f0fdf4" stroke="#22c55e"/>
  <text x="508" y="328" font-size="12.5" fill="#15803d" text-anchor="middle">Host DRAM</text>
  <text x="508" y="348" font-size="12" font-weight="700" fill="#15803d" text-anchor="middle">Host KV (HiCache)</text>

  <!-- NUMA 2 arrows (local, green) -->
  <line x1="508" y1="144" x2="470" y2="158" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>
  <line x1="508" y1="144" x2="576" y2="158" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>
  <line x1="472" y1="214" x2="504" y2="298" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>
  <line x1="578" y1="214" x2="512" y2="298" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>
  <text x="436" y="156" font-size="10" fill="#15803d">PCIe</text>
  <text x="584" y="156" font-size="10" fill="#15803d">PCIe</text>
  <text x="418" y="262" font-size="10.5" fill="#15803d">H2D / D2H</text>
  <text x="584" y="262" font-size="10.5" fill="#15803d">RDMA</text>

  <!-- NUMA interconnect (vertical red double arrow in gap) -->
  <line x1="340" y1="120" x2="340" y2="408" stroke="#dc2626" stroke-width="2.5" marker-start="url(#ar)" marker-end="url(#ar)"/>
  <text x="340" y="100" font-size="11" font-weight="700" fill="#dc2626" text-anchor="middle">NUMA</text>
  <text x="340" y="113" font-size="11" font-weight="700" fill="#dc2626" text-anchor="middle">interconnect</text>

  <!-- remote access (red dashed) -->
  <path d="M200,187 C 300,205 360,285 404,330" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#ar)"/>
  <text x="300" y="244" font-size="11" font-weight="700" fill="#dc2626" text-anchor="middle">remote NUMA 访问</text>
  <text x="300" y="260" font-size="10.5" fill="#dc2626" text-anchor="middle">+1 hop · ↑延迟 · ↓带宽</text>

  <!-- legend -->
  <line x1="40" y1="482" x2="74" y2="482" stroke="#16a34a" stroke-width="3"/>
  <text x="82" y="486" font-size="11.5" fill="#374151">本 NUMA 内访问（零额外跳，理想）</text>
  <line x1="360" y1="482" x2="394" y2="482" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4"/>
  <text x="402" y="486" font-size="11.5" fill="#374151">跨 NUMA 远程访问（+1 跳，延迟↑ 带宽↓）</text>

  <defs>
    <marker id="ag" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#16a34a"/></marker>
    <marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#dc2626"/></marker>
  </defs>
</svg>
  <p class="cap">图 1：P900 + HiCache 的服务器拓扑。绿色 = P900 访问本 NUMA 节点的 Host KV（理想路径）；红色虚线 = P900 跨 NUMA interconnect 去访问另一个节点的 Host KV（多一跳，延迟↑有效带宽↓）。</p>
</div>

三者的分工一句话记：

- **PCIe**：P900 ↔ Host / CPU 之间的数据通道（负责 DMA、带宽、延迟，以及 device discovery / MMIO / BAR / P2P 等控制面）。
- **NUMA**：Host KV 放在哪个 CPU 内存节点、以及 P900 访问它够不够近。
- **NIC**：做 PD 分离 / Mooncake / RDMA / KV Transfer 时，NIC 也要和 P900、Host KV 待在同一个 NUMA。

核心数据链路（一句话版）：

```
P900 HBM → PCIe → PCIe Root → NUMA Node → Host DRAM → Host KV
```

## 2. 五层优化图：从拓扑到绑定怎么落地

<div class="fig">
<svg viewBox="0 0 680 392" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="NUMA 五层优化">
  <rect x="0" y="0" width="680" height="392" fill="none"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">NUMA 优化五层：让「计算设备」和「它需要访问的内存」尽量靠近</text>

  <!-- Row 1 -->
  <rect x="20" y="48" width="6" height="46" rx="3" fill="#10b981"/>
  <circle cx="48" cy="71" r="14" fill="#10b981"/>
  <text x="48" y="76" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">1</text>
  <text x="74" y="68" font-size="13" font-weight="700" fill="#1a1a1a">识别硬件拓扑</text>
  <text x="74" y="88" font-size="11.5" fill="#374151">numactl -H / lscpu -e / lspci -tv / cat numa_node → 画出「P900 → PCIe Root → NUMA」拓扑表</text>

  <!-- Row 2 -->
  <rect x="20" y="108" width="6" height="46" rx="3" fill="#2563eb"/>
  <circle cx="48" cy="131" r="14" fill="#2563eb"/>
  <text x="48" y="136" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">2</text>
  <text x="74" y="128" font-size="13" font-weight="700" fill="#1a1a1a">CPU 绑到对应 NUMA</text>
  <text x="74" y="148" font-size="11.5" fill="#374151">rank0/1 → NUMA0，rank2/3 → NUMA2（即 --numa-node 0 0 2 2 背后的思想）</text>

  <!-- Row 3 -->
  <rect x="20" y="168" width="6" height="46" rx="3" fill="#f59e0b"/>
  <circle cx="48" cy="191" r="14" fill="#f59e0b"/>
  <text x="48" y="196" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">3</text>
  <text x="74" y="188" font-size="13" font-weight="700" fill="#1a1a1a">Host KV 也放 Local NUMA</text>
  <text x="74" y="208" font-size="11.5" fill="#374151">CPU 线程 · Host KV · PCIe Root · P900 四者 locality 一致，而不是只绑 rank</text>

  <!-- Row 4 -->
  <rect x="20" y="228" width="6" height="46" rx="3" fill="#a855f7"/>
  <circle cx="48" cy="251" r="14" fill="#a855f7"/>
  <text x="48" y="256" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">4</text>
  <text x="74" y="248" font-size="13" font-weight="700" fill="#1a1a1a">NIC 也要考虑 NUMA</text>
  <text x="74" y="268" font-size="11.5" fill="#374151">PD / Mooncake / RDMA 场景下，P900 · Host KV · NIC 三者尽量同 NUMA</text>

  <!-- Row 5 -->
  <rect x="20" y="288" width="6" height="46" rx="3" fill="#dc2626"/>
  <circle cx="48" cy="311" r="14" fill="#dc2626"/>
  <text x="48" y="316" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">5</text>
  <text x="74" y="308" font-size="13" font-weight="700" fill="#1a1a1a">避免过度绑定</text>
  <text x="74" y="328" font-size="11.5" fill="#374151">Locality ↔ 负载均衡找平衡；请求不均匀时，强绑死可能反而更差</text>

  <!-- bottom note -->
  <rect x="20" y="350" width="640" height="32" rx="8" fill="#f1f5f9" stroke="#e2e8f0"/>
  <text x="40" y="371" font-size="11.5" fill="#1a1a1a">实验矩阵：正确 NUMA &gt; 自动 NUMA &gt; Remote NUMA ⇒ 证明 HiCache 性能明显受 NUMA locality 影响</text>
</svg>
  <p class="cap">图 2：NUMA 优化五层。从「看清拓扑」到「绑 CPU、绑 KV、绑 NIC」，最后一步是别绑死——locality 和负载均衡之间找平衡。</p>
</div>

## 3. 你现在最该测的不是「NUMA 开 / 关」

真正有价值的实验是对比**不同 locality 配置**，而不是简单的开/关：

| 配置 | 目的 |
|---|---|
| 正确 topology binding | 看 locality 优化的收益上限 |
| 自动 NUMA（系统分配） | baseline 对照 |
| 错误 / remote binding | 验证 NUMA 惩罚有多大 |
| Shared KV + NUMA ON | Shared KV 的最佳 locality |
| Shared KV + NUMA OFF | Shared KV 对 NUMA 的敏感性 |

如果结果呈现：

```
正确 NUMA  >  自动 NUMA  >  Remote NUMA
```

基本就能证明：**HiCache 的性能明显受 NUMA locality 影响**——而这正是 `--numa-node 0 0 2 2` 这类参数存在的意义。

## 4. 一句话总结

> **PCIe 解决「P900 怎么和 Host / CPU 传数据」，NUMA 解决「这些 Host 数据放在哪个 CPU 内存节点、离 P900 够不够近」，NIC 在 PD / Mooncake 时也要就近。**

所以你最终优化的是一整条链路：

```
P900  ↔  PCIe Root Complex  ↔  NUMA Node  ↔  Host DRAM  ↔  Host KV
```

而不是单独一个 `--numa-node` 参数。
