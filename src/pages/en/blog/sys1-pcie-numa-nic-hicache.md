---
title: 'Inference Systems Infrastructure Notes (1): The NUMA x PCIe x NIC Data Path Under P900 + HiCache'
description: 'Understand PCIe, NUMA and NIC as one single data path. For the P900 plus HiCache scenario, two diagrams (a topology map and a five-layer optimization map) explain how the accelerator exchanges data with host KV, which CPU memory node the host KV should live on, and why the NIC has to be considered alongside them for PD, Mooncake and RDMA.'
pubDate: 2026-08-26
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys1-pcie-numa-nic-hicache
layout: ../../../layouts/BlogPost.astro
---

## 0. Why P900 + HiCache Forces You to Look at NUMA / PCIe / NIC

HiCache puts the KV cache in **host DRAM**, treating it as one large host-side KV pool. But when a request hits host KV, that data is **not consumed on the P900 right away** — it usually has to travel one more leg:

```text
Host KV (host DRAM)
     |
     |  H2D (host to device)
     v
   PCIe
     |
     v
   P900
     |
     v
   HBM (device memory)
```

In other words, **a high host KV hit rate does not guarantee high performance**, because the PCIe transfer cost comes after the hit. If PCIe bandwidth is insufficient, latency is high, or the KV lives across a NUMA boundary, you can end up with:

```text
Host KV hit rate up
     |
     v
   PCIe transfer
     |
     v
TTFT up / TPS down
```

So in this scenario **PCIe, NUMA and NIC must be read as one path**. First a topology diagram to make the hardware relationships concrete, then a five-layer optimization diagram that turns "how to tune it" into actionable steps.

## 1. Topology: What Actually Happens Inside One Server

<div class="fig">
<svg viewBox="0 0 680 510" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="P900 HiCache server topology with NUMA, PCIe and NIC">
  <rect x="0" y="0" width="680" height="510" fill="none"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">P900 + HiCache server topology: NUMA x PCIe x NIC</text>

  <rect x="20" y="40" width="640" height="420" rx="14" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>

  <rect x="36" y="64" width="276" height="372" rx="10" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5"/>
  <text x="174" y="86" font-size="13" font-weight="700" fill="#047857" text-anchor="middle">NUMA Node 0</text>

  <rect x="368" y="64" width="276" height="372" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="506" y="86" font-size="13" font-weight="700" fill="#1d4ed8" text-anchor="middle">NUMA Node 2</text>

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

  <line x1="176" y1="144" x2="138" y2="158" stroke="#16a34a" stroke-width="2" marker-end="url(#agEn)"/>
  <line x1="176" y1="144" x2="244" y2="158" stroke="#16a34a" stroke-width="2" marker-end="url(#agEn)"/>
  <line x1="140" y1="214" x2="172" y2="298" stroke="#16a34a" stroke-width="2" marker-end="url(#agEn)"/>
  <line x1="246" y1="214" x2="180" y2="298" stroke="#16a34a" stroke-width="2" marker-end="url(#agEn)"/>
  <text x="104" y="156" font-size="10" fill="#15803d">PCIe</text>
  <text x="252" y="156" font-size="10" fill="#15803d">PCIe</text>
  <text x="86" y="262" font-size="10.5" fill="#15803d">H2D / D2H</text>
  <text x="252" y="262" font-size="10.5" fill="#15803d">RDMA</text>

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

  <line x1="508" y1="144" x2="470" y2="158" stroke="#16a34a" stroke-width="2" marker-end="url(#agEn)"/>
  <line x1="508" y1="144" x2="576" y2="158" stroke="#16a34a" stroke-width="2" marker-end="url(#agEn)"/>
  <line x1="472" y1="214" x2="504" y2="298" stroke="#16a34a" stroke-width="2" marker-end="url(#agEn)"/>
  <line x1="578" y1="214" x2="512" y2="298" stroke="#16a34a" stroke-width="2" marker-end="url(#agEn)"/>
  <text x="436" y="156" font-size="10" fill="#15803d">PCIe</text>
  <text x="584" y="156" font-size="10" fill="#15803d">PCIe</text>
  <text x="418" y="262" font-size="10.5" fill="#15803d">H2D / D2H</text>
  <text x="584" y="262" font-size="10.5" fill="#15803d">RDMA</text>

  <line x1="340" y1="120" x2="340" y2="408" stroke="#dc2626" stroke-width="2.5" marker-start="url(#arEn)" marker-end="url(#arEn)"/>
  <text x="340" y="100" font-size="11" font-weight="700" fill="#dc2626" text-anchor="middle">NUMA</text>
  <text x="340" y="113" font-size="11" font-weight="700" fill="#dc2626" text-anchor="middle">interconnect</text>

  <path d="M200,187 C 300,205 360,285 404,330" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#arEn)"/>
  <text x="300" y="244" font-size="11" font-weight="700" fill="#dc2626" text-anchor="middle">remote NUMA access</text>
  <text x="300" y="260" font-size="10.5" fill="#dc2626" text-anchor="middle">+1 hop, latency up, bandwidth down</text>

  <line x1="40" y1="482" x2="74" y2="482" stroke="#16a34a" stroke-width="3"/>
  <text x="82" y="486" font-size="11.5" fill="#374151">Access inside the same NUMA node (zero extra hops, ideal)</text>
  <line x1="360" y1="482" x2="394" y2="482" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4"/>
  <text x="402" y="486" font-size="11.5" fill="#374151">Cross-NUMA remote access (+1 hop, latency up, bandwidth down)</text>

  <defs>
    <marker id="agEn" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#16a34a"/></marker>
    <marker id="arEn" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#dc2626"/></marker>
  </defs>
</svg>
<p class="cap">Figure 1: P900 + HiCache server topology. Green = the P900 accesses host KV in its own NUMA node (the ideal path); red dashed = the P900 crosses the NUMA interconnect to reach another node's host KV (one extra hop, higher latency, lower effective bandwidth).</p>
</div>

The division of labour, in one line each:

- **PCIe** — the data channel between the P900 and the host / CPU (DMA, bandwidth, latency, plus the control plane: device discovery, MMIO, BAR, P2P).
- **NUMA** — which CPU memory node the host KV sits in, and whether the P900 sits close enough to it.
- **NIC** — for PD disaggregation / Mooncake / RDMA / KV transfer, the NIC also needs to be on the same NUMA node as the P900 and the host KV.

The core data path, one-line version:

```text
P900 HBM -> PCIe -> PCIe Root -> NUMA Node -> Host DRAM -> Host KV
```

## 2. Five-Layer Optimization: From Topology to Actual Pinning

<div class="fig">
<svg viewBox="0 0 680 392" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Five layers of NUMA optimization">
  <rect x="0" y="0" width="680" height="392" fill="none"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Five layers of NUMA optimization: keep the compute device close to the memory it reads</text>

  <rect x="20" y="48" width="6" height="46" rx="3" fill="#10b981"/>
  <circle cx="48" cy="71" r="14" fill="#10b981"/>
  <text x="48" y="76" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">1</text>
  <text x="74" y="68" font-size="13" font-weight="700" fill="#1a1a1a">Identify the hardware topology</text>
  <text x="74" y="88" font-size="11.5" fill="#374151">numactl -H / lscpu -e / lspci -tv / cat numa_node, then draw the P900 to PCIe Root to NUMA table</text>

  <rect x="20" y="108" width="6" height="46" rx="3" fill="#2563eb"/>
  <circle cx="48" cy="131" r="14" fill="#2563eb"/>
  <text x="48" y="136" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">2</text>
  <text x="74" y="128" font-size="13" font-weight="700" fill="#1a1a1a">Pin CPUs to the matching NUMA node</text>
  <text x="74" y="148" font-size="11.5" fill="#374151">rank0/1 to NUMA0, rank2/3 to NUMA2 (the idea behind flags like --numa-node 0 0 2 2)</text>

  <rect x="20" y="168" width="6" height="46" rx="3" fill="#f59e0b"/>
  <circle cx="48" cy="191" r="14" fill="#f59e0b"/>
  <text x="48" y="196" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">3</text>
  <text x="74" y="188" font-size="13" font-weight="700" fill="#1a1a1a">Put host KV on the local NUMA node too</text>
  <text x="74" y="208" font-size="11.5" fill="#374151">CPU thread, host KV, PCIe Root and P900 should all agree on locality, not just the ranks</text>

  <rect x="20" y="228" width="6" height="46" rx="3" fill="#a855f7"/>
  <circle cx="48" cy="251" r="14" fill="#a855f7"/>
  <text x="48" y="256" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">4</text>
  <text x="74" y="248" font-size="13" font-weight="700" fill="#1a1a1a">Consider NUMA for the NIC as well</text>
  <text x="74" y="268" font-size="11.5" fill="#374151">Under PD / Mooncake / RDMA, keep P900, host KV and NIC on the same NUMA node</text>

  <rect x="20" y="288" width="6" height="46" rx="3" fill="#dc2626"/>
  <circle cx="48" cy="311" r="14" fill="#dc2626"/>
  <text x="48" y="316" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">5</text>
  <text x="74" y="308" font-size="13" font-weight="700" fill="#1a1a1a">Avoid over-pinning</text>
  <text x="74" y="328" font-size="11.5" fill="#374151">Balance locality against load; when requests are skewed, hard pinning can be worse</text>

  <rect x="20" y="350" width="640" height="32" rx="8" fill="#f1f5f9" stroke="#e2e8f0"/>
  <text x="40" y="371" font-size="11.5" fill="#1a1a1a">Experiment matrix: correct NUMA &gt; automatic NUMA &gt; remote NUMA, showing HiCache performance is clearly sensitive to NUMA locality</text>
</svg>
<p class="cap">Figure 2: Five layers of NUMA optimization. From "see the topology" through pinning CPUs, KV and NIC, to the final step of not over-pinning: find the balance between locality and load.</p>
</div>

## 3. What You Should Measure Is Not "NUMA On / Off"

The valuable experiment compares **different locality configurations**, not a simple on/off switch:

| Configuration | Purpose |
|---|---|
| Correct topology binding | See the upper bound of the locality win |
| Automatic NUMA (system assigned) | Baseline for comparison |
| Wrong / remote binding | Measure how large the NUMA penalty is |
| Shared KV + NUMA ON | Best-case locality for shared KV |
| Shared KV + NUMA OFF | How sensitive shared KV is to NUMA |

If the results come out as:

```text
correct NUMA  >  automatic NUMA  >  remote NUMA
```

that is essentially proof that **HiCache performance is clearly affected by NUMA locality** — which is exactly why parameters like `--numa-node 0 0 2 2` exist.

## 4. One-Line Summary

> **PCIe answers "how does the P900 exchange data with the host / CPU"; NUMA answers "which CPU memory node holds that host data and is it close enough to the P900"; and in PD / Mooncake setups the NIC also has to be nearby.**

So what you are ultimately optimizing is an entire path:

```text
P900  <->  PCIe Root Complex  <->  NUMA Node  <->  Host DRAM  <->  Host KV
```

not a single `--numa-node` flag.
