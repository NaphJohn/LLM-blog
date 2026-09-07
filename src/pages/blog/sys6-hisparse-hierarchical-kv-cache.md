---
title: '推理系统基础设施手记（六）：HiSparse —— 把 HBM 当缓存用，长上下文服务的容量墙怎么拆'
description: '稀疏注意力把长上下文解码的「算力」压到了 top-k，可服务系统仍把整份 KV 留在 GPU HBM 里，于是显存先于算力耗尽。本篇拆解 Stanford MAST 的 HiSparse（arXiv:2608.07009，已合入 SGLang 上游）：主机 DRAM 存权威全量 KV、GPU 只留固定 B 槽热缓存、一个 RESOLVE 融合 CUDA 核在 decode graph 内完成命中检测 / LRU 替换 / 主机到设备抓取，并用精确逐层预取隐藏约一半 IO。附实测数字、适用前提与局限。'
pubDate: 2026-09-07
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys6-hisparse-hierarchical-kv-cache
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

> **前三篇（sys3/sys5）讲的是「怎么让 KV 变小、让每次读得更少」；这一篇讲的是更小之后仍然装不下的那一半问题——把 GPU HBM 从「KV 的居住地」降级成「KV 的缓存」。**

HiSparse（*Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management*，Stanford MAST Lab，arXiv:2608.07009，2026-08）的核心主张只有一句：**既然 top-k 稀疏注意力每步只读 k 个 KV 条目，那 HBM 里凭什么要常驻 L_ctx 个？** 把全量 KV 挪到主机 DRAM，GPU 只留一个固定大小的热缓存，再配一个融合 CUDA 核把「命中检测 + LRU 替换 + 主机抓取」一次做完——就能在**输出逐 token 完全不变**的前提下，把长上下文的并发度拉上去。

它已经合入 **SGLang 上游**，在 H200 / B200 / GH200 上跑通 DSA、NSA、Quest 三类稀疏注意力，长上下文峰值生成吞吐最高 **4.7×**。

---

## 1. 容量墙：稀疏注意力省了算力，却没省显存

先把问题摆清楚。Top-k 稀疏注意力（NSA、DSA、Quest 等）的卖点是**计算便宜**：第 ℓ 层、第 t 步只挑出 k 个历史位置 S_t^(ℓ) 来做注意力，k 通常是几千，而不是整个上下文长度 L_ctx。

但服务系统有个隐含假设拖了后腿：

> **任意一个过去的 token，都可能在未来某一步被索引器选中。**

为了不漏，系统干脆把**整个 KV cache 常驻 GPU HBM**。于是显存账单依旧随 L_ctx 线性增长——**算力还没用完，显存先没了**。这就是论文说的 capacity wall。

数字很直白：

| 场景 | KV 显存占用 | 后果 |
|---|---|---|
| GLM-5.1，单请求 128K 上下文 | ≈ **13.09 GB** | 80GB 卡只能并发个位数请求 |
| 单请求 1M 上下文 | 超过整卡 HBM | **根本无法服务** |

注意这里的尴尬：**计算侧已经优化到只读 k 条了，内存侧却还按 L_ctx 全量付费**。HiSparse 要拆的就是这个错配——让 HBM 占用与 k 成比例，而不是与 L_ctx 成比例。

难点在于稀疏选择是**动态**的：S_t^(ℓ) 每一步、每一层都在变，你没法像 MLA 那样静态地"压缩完就固定"。

---

## 2. 两级内存层次：主机存权威副本，GPU 只留热缓存

HiSparse 的解法是不碰模型逻辑，**只改 KV 记录放在哪**：

<div class="arch-fig">
<svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="HiSparse 分层 KV 缓存架构：主机 DRAM 存全量权威副本，GPU HBM 只留 B 槽热缓存，RESOLVE 核负责抓取">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#8B4513"/>
    </marker>
    <style>
      .host{fill:#eff6ff;stroke:#1d4ed8;stroke-width:1.5}
      .gpu{fill:#fff7ed;stroke:#c2410c;stroke-width:1.5}
      .kern{fill:#fef9c3;stroke:#a16207;stroke-width:1.5}
      .lab{font:600 13px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .sub{font:11px -apple-system,'PingFang SC',sans-serif;fill:#475569}
      .tiny{font:10px -apple-system,'PingFang SC',sans-serif;fill:#6b7280}
      .warn{font:600 11px -apple-system,'PingFang SC',sans-serif;fill:#b91c1c}
    </style>
  </defs>

  <rect x="30" y="20" width="280" height="86" rx="8" class="host"/>
  <text x="46" y="44" class="lab">① 主机 KV 池（Host DRAM · pinned）</text>
  <text x="46" y="64" class="sub">每个请求的权威全量 KV 历史，prefill 时直接写入</text>
  <text x="46" y="82" class="sub">容量 ≈ 主机内存，与 L_ctx 同量级</text>
  <text x="46" y="99" class="tiny">GLM-5.1 128K 单请求 ≈ 13.09 GB，主机内存毫无压力</text>

  <rect x="370" y="20" width="280" height="86" rx="8" class="gpu"/>
  <text x="386" y="44" class="lab">② GPU 热缓存（HBM · 固定 B 槽）</text>
  <text x="386" y="64" class="sub">每个「请求 × 层」一份，B ≥ k，只放最近被选中的 KV</text>
  <text x="386" y="82" class="sub">HBM 占用 = N_ℓ × B × W_KV × s，与 L_ctx 解耦</text>
  <text x="386" y="99" class="tiny">128K GLM-5.1 实测最高省 30× 显存</text>

  <rect x="175" y="160" width="330" height="72" rx="8" class="kern"/>
  <text x="191" y="184" class="lab">③ RESOLVE 融合 CUDA 核（在 decode CUDA graph 内）</text>
  <text x="191" y="204" class="sub">暂存 → 标记 → 扫描 → 抓取 → 发布，一次 launch 做完</text>
  <text x="191" y="222" class="tiny">GPU 线程用 ld.global.nc.v2.b64 直接从 pinned 主机内存拉 KV</text>

  <line x1="150" y1="106" x2="230" y2="158" stroke="#8B4513" stroke-width="1.6" marker-end="url(#arrow)"/>
  <text x="140" y="140" class="sub">miss 抓取</text>
  <line x1="500" y1="160" x2="500" y2="110" stroke="#8B4513" stroke-width="1.6" marker-end="url(#arrow)"/>
  <text x="510" y="140" class="sub">写回 / 更新页表</text>

  <rect x="175" y="268" width="330" height="52" rx="8" class="gpu"/>
  <text x="191" y="290" class="lab">④ 稀疏注意力核</text>
  <text x="191" y="308" class="sub">拿到物理设备偏移后照常算，读到的内容与全量驻留完全一致</text>
  <line x1="340" y1="232" x2="340" y2="266" stroke="#8B4513" stroke-width="1.6" marker-end="url(#arrow)"/>

  <text x="30" y="352" class="lab">关键不变式</text>
  <text x="30" y="372" class="sub">只改 KV 的物理存放位置，不改模型结构、不改计算 → 输出逐 token 不变（exact）</text>
  <text x="30" y="390" class="warn">代价只有一个：host ↔ device 的 IO 带宽</text>
</svg>
<p class="cap">图 1：HiSparse 的两级层次。主机 DRAM 是「权威副本」，GPU HBM 退化成一个由 RESOLVE 核管理的热缓存。</p>
</div>

三个部件各司其职：

1. **主机 KV 池**：prefill 阶段生成的 KV 记录直接写进固定的主机内存，这是唯一权威副本。
2. **GPU 热缓存**：为每个「请求 × 层」保留 B 个槽位，装最近被选中的 KV 记录。为保证当前注意力一定算得动，约束 **B ≥ k**。
3. **元数据**：GPU 上维护紧凑页表，把逻辑 token 位置映射到物理槽位，或标记为「仅主机」；同时维护 LRU 所需的最近性信息。

于是单请求 HBM 消耗变成：

```
HBM 占用 = N_ℓ × B × W_KV × s
```

N_ℓ 是层数，W_KV 是每 token 的 KV 元素数，s 是每元素字节数。**L_ctx 消失了**——这就是"把解码吞吐与显存容量解耦"的实质。

### 2.1 槽位到底怎么分（DRAM 与 HBM 的对应关系）

<div class="arch-fig">
<svg viewBox="0 0 680 470" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="HiSparse 槽位分配：主机 DRAM 按 token 全量存放，GPU HBM 每个请求层一组 B 槽，页表做逻辑位置到物理槽的映射">
  <defs>
    <marker id="arAlloc" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#8B4513"/>
    </marker>
    <style>
      .host{fill:#eff6ff;stroke:#1d4ed8;stroke-width:1.5}
      .gpu{fill:#fff7ed;stroke:#c2410c;stroke-width:1.5}
      .pt{fill:#f8fafc;stroke:#64748b;stroke-width:1.5}
      .lab{font:600 13px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .sub{font:11px -apple-system,'PingFang SC',sans-serif;fill:#475569}
      .tiny{font:10px -apple-system,'PingFang SC',sans-serif;fill:#6b7280}
      .cell{font:600 11px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .victim{fill:#fee2e2;stroke:#dc2626;stroke-width:1.5}
      .hot{fill:#dcfce7;stroke:#16a34a;stroke-width:1.2}
      .slot{fill:#ffffff;stroke:#c2410c;stroke-width:1}
      .kv{fill:#dbeafe;stroke:#2563eb;stroke-width:0.8}
      .form{font:700 13px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
    </style>
  </defs>

  <text x="20" y="24" class="lab">槽位分配：DRAM 按 token 全量存，HBM 按「请求 × 层」各给 B 槽</text>

  <rect x="20" y="44" width="250" height="250" rx="10" class="host"/>
  <text x="34" y="68" class="lab">① 主机 DRAM（权威全量副本）</text>
  <text x="34" y="88" class="sub">request 0</text>
  <g>
    <rect x="95" y="76" width="13" height="18" class="kv"/><rect x="110" y="76" width="13" height="18" class="kv"/><rect x="125" y="76" width="13" height="18" class="kv"/><rect x="140" y="76" width="13" height="18" class="kv"/><rect x="155" y="76" width="13" height="18" class="kv"/><rect x="170" y="76" width="13" height="18" class="kv"/><rect x="185" y="76" width="13" height="18" class="kv"/><rect x="200" y="76" width="13" height="18" class="kv"/><rect x="215" y="76" width="13" height="18" class="kv"/><rect x="230" y="76" width="13" height="18" class="kv"/><rect x="245" y="76" width="13" height="18" class="kv"/><rect x="255" y="76" width="8" height="18" class="kv"/>
  </g>
  <text x="34" y="138" class="sub">request 1</text>
  <g>
    <rect x="95" y="126" width="13" height="18" class="kv"/><rect x="110" y="126" width="13" height="18" class="kv"/><rect x="125" y="126" width="13" height="18" class="kv"/><rect x="140" y="126" width="13" height="18" class="kv"/><rect x="155" y="126" width="13" height="18" class="kv"/><rect x="170" y="126" width="13" height="18" class="kv"/><rect x="185" y="126" width="13" height="18" class="kv"/><rect x="200" y="126" width="13" height="18" class="kv"/><rect x="215" y="126" width="13" height="18" class="kv"/><rect x="230" y="126" width="13" height="18" class="kv"/><rect x="245" y="126" width="13" height="18" class="kv"/><rect x="255" y="126" width="8" height="18" class="kv"/>
  </g>
  <text x="34" y="188" class="sub">request 2</text>
  <g>
    <rect x="95" y="176" width="13" height="18" class="kv"/><rect x="110" y="176" width="13" height="18" class="kv"/><rect x="125" y="176" width="13" height="18" class="kv"/><rect x="140" y="176" width="13" height="18" class="kv"/><rect x="155" y="176" width="13" height="18" class="kv"/><rect x="170" y="176" width="13" height="18" class="kv"/><rect x="185" y="176" width="13" height="18" class="kv"/><rect x="200" y="176" width="13" height="18" class="kv"/><rect x="215" y="176" width="13" height="18" class="kv"/><rect x="230" y="176" width="13" height="18" class="kv"/><rect x="245" y="176" width="13" height="18" class="kv"/><rect x="255" y="176" width="8" height="18" class="kv"/>
  </g>
  <text x="34" y="228" class="tiny">每个 (request, layer, pos) 的 K/V 全量保留</text>
  <text x="34" y="246" class="tiny">prefill 直接写入，永不淘汰、永不做 LRU</text>
  <text x="34" y="264" class="tiny">容量随主机内存走，128K 请求约 13 GB 也毫无压力</text>
  <text x="34" y="284" class="tiny">按逻辑位置连续寻址，是唯一权威副本</text>

  <rect x="290" y="44" width="90" height="250" rx="10" class="pt"/>
  <text x="335" y="68" class="lab" text-anchor="middle">② 页表</text>
  <text x="335" y="88" class="tiny" text-anchor="middle">逻辑位置 → 物理槽</text>
  <rect x="300" y="100" width="70" height="20" rx="4" class="slot"/><text x="335" y="114" class="cell" text-anchor="middle">pos→slot</text>
  <rect x="300" y="126" width="70" height="20" rx="4" class="slot"/><text x="335" y="140" class="cell" text-anchor="middle">pos→slot</text>
  <rect x="300" y="152" width="70" height="20" rx="4" class="slot"/><text x="335" y="166" class="cell" text-anchor="middle">pos→slot</text>
  <rect x="300" y="178" width="70" height="20" rx="4" class="slot"/><text x="335" y="192" class="cell" text-anchor="middle">pos→slot</text>
  <text x="335" y="224" class="tiny" text-anchor="middle">命中</text>
  <text x="335" y="240" class="tiny" text-anchor="middle">= 槽里有</text>
  <text x="335" y="260" class="tiny" text-anchor="middle">未命中</text>
  <text x="335" y="276" class="tiny" text-anchor="middle">= 仅主机</text>

  <rect x="400" y="44" width="260" height="250" rx="10" class="gpu"/>
  <text x="414" y="68" class="lab">③ GPU HBM（热缓存 · 固定 B 槽）</text>
  <text x="414" y="86" class="sub">每个「请求 × 层」一组，B ≥ k</text>
  <text x="402" y="115" class="tiny">L0</text>
  <rect x="440" y="100" width="32" height="22" rx="3" class="hot"/><text x="456" y="115" class="cell" text-anchor="middle">17</text>
  <rect x="476" y="100" width="32" height="22" rx="3" class="slot"/><text x="492" y="115" class="cell" text-anchor="middle">9</text>
  <rect x="512" y="100" width="32" height="22" rx="3" class="slot"/><text x="528" y="115" class="cell" text-anchor="middle">44</text>
  <rect x="548" y="100" width="32" height="22" rx="3" class="victim"/><text x="564" y="115" class="cell" text-anchor="middle">3</text>
  <rect x="584" y="100" width="32" height="22" rx="3" class="slot"/><text x="600" y="115" class="cell" text-anchor="middle">21</text>
  <rect x="620" y="100" width="32" height="22" rx="3" class="slot"/><text x="636" y="115" class="cell" text-anchor="middle">8</text>
  <text x="402" y="143" class="tiny">L1</text>
  <rect x="440" y="128" width="32" height="22" rx="3" class="slot"/><text x="456" y="143" class="cell" text-anchor="middle">31</text>
  <rect x="476" y="128" width="32" height="22" rx="3" class="hot"/><text x="492" y="143" class="cell" text-anchor="middle">17</text>
  <rect x="512" y="128" width="32" height="22" rx="3" class="slot"/><text x="528" y="143" class="cell" text-anchor="middle">6</text>
  <rect x="548" y="128" width="32" height="22" rx="3" class="slot"/><text x="564" y="143" class="cell" text-anchor="middle">52</text>
  <rect x="584" y="128" width="32" height="22" rx="3" class="victim"/><text x="600" y="143" class="cell" text-anchor="middle">12</text>
  <rect x="620" y="128" width="32" height="22" rx="3" class="slot"/><text x="636" y="143" class="cell" text-anchor="middle">40</text>
  <text x="402" y="171" class="tiny">L2</text>
  <rect x="440" y="156" width="32" height="22" rx="3" class="slot"/><text x="456" y="171" class="cell" text-anchor="middle">5</text>
  <rect x="476" y="156" width="32" height="22" rx="3" class="slot"/><text x="492" y="171" class="cell" text-anchor="middle">28</text>
  <rect x="512" y="156" width="32" height="22" rx="3" class="hot"/><text x="528" y="171" class="cell" text-anchor="middle">17</text>
  <rect x="548" y="156" width="32" height="22" rx="3" class="slot"/><text x="564" y="171" class="cell" text-anchor="middle">63</text>
  <rect x="584" y="156" width="32" height="22" rx="3" class="slot"/><text x="600" y="171" class="cell" text-anchor="middle">1</text>
  <rect x="620" y="156" width="32" height="22" rx="3" class="slot"/><text x="636" y="171" class="cell" text-anchor="middle">35</text>
  <text x="402" y="199" class="tiny">L3</text>
  <rect x="440" y="184" width="32" height="22" rx="3" class="slot"/><text x="456" y="199" class="cell" text-anchor="middle">22</text>
  <rect x="476" y="184" width="32" height="22" rx="3" class="slot"/><text x="492" y="199" class="cell" text-anchor="middle">47</text>
  <rect x="512" y="184" width="32" height="22" rx="3" class="slot"/><text x="528" y="199" class="cell" text-anchor="middle">14</text>
  <rect x="548" y="184" width="32" height="22" rx="3" class="hot"/><text x="564" y="199" class="cell" text-anchor="middle">17</text>
  <rect x="584" y="184" width="32" height="22" rx="3" class="slot"/><text x="600" y="199" class="cell" text-anchor="middle">59</text>
  <rect x="620" y="184" width="32" height="22" rx="3" class="slot"/><text x="636" y="199" class="cell" text-anchor="middle">30</text>
  <text x="414" y="226" class="tiny">图中 B = 6 仅示意；实际 B 为 k 的数倍（几千量级）</text>
  <text x="414" y="244" class="tiny">绿 = 本步命中的热槽，红 = LRU 选中的 victim（下一个 miss 顶掉它）</text>
  <text x="414" y="262" class="tiny">同一逻辑位置 17 可在多层各占一槽：分配粒度是「请求 × 层」</text>
  <text x="414" y="284" class="tiny">每槽 = 1 条 KV 记录 + 逻辑位置 + LRU 最近性位</text>

  <line x1="270" y1="150" x2="286" y2="150" stroke="#8B4513" stroke-width="1.6" marker-end="url(#arAlloc)"/>
  <line x1="380" y1="150" x2="396" y2="150" stroke="#8B4513" stroke-width="1.6" marker-end="url(#arAlloc)"/>
  <text x="271" y="140" class="tiny">查</text>

  <rect x="20" y="310" width="640" height="70" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="36" y="336" class="form">HBM 占用 = N_ℓ × B × W_KV × s —— 与 L_ctx 无关</text>
  <text x="36" y="358" class="sub">分配粒度是「请求 × 层」而不是「请求」：不同层的索引器选择不同，各自要有一组槽。</text>
  <text x="36" y="376" class="sub">约束 B ≥ k：保证当前步选出的 k 条一定放得下，注意力永远算得动。</text>

  <rect x="20" y="394" width="640" height="62" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="36" y="418" class="sub">算例：128K 上下文、k = 2048、取 B = 2k = 4096。</text>
  <text x="36" y="440" class="sub">单层常驻 4096 条 vs 全量 131072 条 ≈ 3.1%，约 <tspan font-weight="700">32× 节省</tspan>（与论文实测"最高 30×"吻合）。</text>
  <text x="36" y="458" class="sub">上下文再涨到 1M，B 仍是 4096，<tspan font-weight="700">HBM 占用不增</tspan>——这才叫解耦。</text>
</svg>
<p class="cap">图 3：DRAM 与 HBM 的分配对应关系。DRAM 侧按 (request, layer, pos) 连续全量存放；HBM 侧给每个「请求 × 层」分配一组固定 B 槽，靠页表做逻辑位置到物理槽的映射，LRU 决定谁被顶掉。</p>
</div>

三个要点值得单独记住：

1. **分配粒度是「请求 × 层」，不是「请求」**。不同层的索引器选择不同，所以每层都要有一组自己的槽——这也是为什么公式里要乘 N_ℓ。
2. **DRAM 与 HBM 解耦靠页表**。DRAM 侧按逻辑位置连续寻址；HBM 侧的槽位与逻辑位置无关，装的是"最近被选中的那几条"，映射关系全在页表里。
3. **B 是常数，不随上下文长**。这正是容量墙被拆掉的地方：上下文从 128K 涨到 1M，DRAM 侧线性增长（无所谓，主机内存大），HBM 侧一动不动。

---

## 3. RESOLVE 融合核：五个阶段一次 launch 做完

分层缓存的想法不新，难的是**怎么让 miss 的代价低到可以接受**。稀疏选择的访存是零散的（scattered），传统 CPU 侧分页搬运会直接把延迟暴露出来。

HiSparse 的做法是写一个叫 `RESOLVE` 的单一融合 CUDA 核，**每个稀疏层启动一次**，在 GPU 线程里并行做完五件事：

| 阶段 | 做什么 | 关键点 |
|---|---|---|
| **暂存 Stage** | 把索引器选出的逻辑位置载入共享内存哈希表 | 后续比对全在片上，不碰全局内存 |
| **标记 Mark** | 对照哈希表检查 GPU 缓存槽，识别"命中"和"可驱逐"槽 | 命中判定是纯元数据操作 |
| **扫描 Scan** | 并行扫描槽位、更新 LRU 元数据，为当前步的 miss 挑 victim | 并行前缀和式的槽位分配 |
| **抓取 Fetch** | miss 线程用**矢量化非一致性加载**（`ld.global.nc.v2.b64`）把 KV 从 pinned 主机内存拉进分配到的 GPU 槽 | **GPU-assisted IO：GPU 线程自己去拉** |
| **发布 Publish** | 更新页表，把物理设备偏移交给稀疏注意力核 | 下游核无感知 |

**"GPU 辅助 IO"是这篇论文真正的技术贡献**：让 GPU 线程直接从主机内存取数，即使是稀疏选择这种零散访存，也能吃满 PCIe / NVLink 带宽——绕开了"CPU 发现缺页 → 通知 → 再搬运"这条长路径。而且整个流程**在 decode CUDA Graph 内部完成**，不会因为动态控制流破坏图捕获。

---

## 4. 局部性：为什么一个小小的 B 就够用

任何缓存都靠局部性吃饭。HiSparse 的实测发现，稀疏注意力的选择序列有两个可利用的性质：

- **时间局部性**：连续解码步骤会**重复选中大量相同的 token**；
- **跨层相关性**：不同层往往关注上下文里相近的区域。

用 GLM-5.1（k = 2048）跑出来的 miss rate 很能说明问题：

<div class="arch-fig">
<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="GPU 缓存大小 B 相对 k 的倍数与 top-k miss rate 的关系">
  <style>
    .axis{stroke:#9ca3af;stroke-width:1.2}
    .bar{fill:#fb923c}
    .bar2{fill:#f97316}
    .bar3{fill:#c2410c}
    .lab{font:600 12px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
    .sub{font:11px -apple-system,'PingFang SC',sans-serif;fill:#475569}
    .val{font:600 12px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
  </style>
  <line x1="70" y1="200" x2="620" y2="200" class="axis"/>
  <line x1="70" y1="30" x2="70" y2="200" class="axis"/>

  <rect x="130" y="80" width="90" height="120" class="bar"/>
  <text x="152" y="70" class="val">30.0%</text>
  <text x="145" y="220" class="lab">B = k</text>

  <rect x="290" y="144" width="90" height="56" class="bar2"/>
  <text x="312" y="134" class="val">13.4%</text>
  <text x="305" y="220" class="lab">B = 2k</text>

  <rect x="450" y="172" width="90" height="28" class="bar3"/>
  <text x="472" y="162" class="val">6.7%</text>
  <text x="465" y="220" class="lab">B = 4k</text>

  <text x="70" y="22" class="lab">GLM-5.1 · k = 2048：top-k miss rate 随缓存放大快速下降</text>
  <text x="80" y="244" class="sub">缓存从 k 放大到 2k，miss 直接腰斩；放大到 4k 再腰斩一次。B 只需几倍 k，不必随上下文增长。</text>
</svg>
<p class="cap">图 2：LRU 管理下的局部性红利。缓存只要几倍于 k，就能把 miss 压到个位数百分比。</p>
</div>

### 精确逐层预取

还有一招更狠的。有些模型（如 GLM-5.2）在**多组层之间共享索引器输出**——一旦某个"锚点层"确定了它的选择，系统**立刻就能知道**后面那些共享层的选择是什么。

HiSparse 据此做 **exact layer-wise prefetching**：

1. 锚点层算出选择 → 同步推出后续共享层的 miss 计划；
2. 一个**后台纯复制核**重放这些未来层的 miss 计划；
3. 主机到设备的传输与当前层的计算**重叠**。

结果：**隐藏掉约一半的剩余 IO 延迟**。注意这里用的是"精确"（exact）——不是启发式猜，是确定性地知道，所以不会引入正确性风险。

---

## 5. 为什么它是「精确」且「与索引器无关」

这两点是 HiSparse 能直接进生产的关键：

- **精确（exact）**：只改变 KV 记录的**物理放置**，不碰模型结构、不碰注意力计算、不做近似召回。所以**模型输出逐 token 完全不变**——这是和"用近似检索换显存"那类方案的根本分野。
- **与索引器无关（indexer-agnostic）**：它不关心你是怎么挑出那 k 个位置的。论文在 **DSA、NSA、Quest** 三类稀疏注意力上都做了评估，都能吃。

换句话说，HiSparse 是**稀疏注意力的通用内存后端**——只要你的模型已经有稀疏索引器，它就能接上。

---

## 6. 实测效果

| 配置 | 结果 |
|---|---|
| DeepSeek-V4-Flash（NSA），双 B200，64 并发 | 生成吞吐 **2.1×** |
| Qwen3 + Quest，GH200，20 万输入长度 | 生成吞吐最高 **4.7×** |
| 高负载场景（prefill / decode 共享 GPU） | **TTFT 显著下降**（不再因 HBM 耗尽阻塞新请求） |
| 单 token 延迟 | 与基线**相当**（不是更快，但没变慢） |
| no-IO oracle 实验 | 解析机制本身**无可测量的 per-token 开销** |

最后一行很重要：它说明 **host-device IO 是这套方案唯一的、也是全部的代价**。所以这条链路的带宽直接决定收益：

> **链路敏感性**：GH200（NVLink-C2C）上的 KV 抓取耗时比 H200（PCIe Gen5）**少近 4 倍**，因而可以配更小的 GPU 缓存、跑更高的并发。

---

## 7. 什么时候不该用它（局限与前提）

这部分比效果更值得记：

1. **前提是主机 DRAM 远大于 GPU HBM**。在 **Grace 系的 GB200 / GB300** 上，CPU 与 GPU 共享统一内存、主机侧并无容量优势——这套"卸载换容量"的逻辑**不成立**。论文自己把这点列为根本性限制。
2. **依赖 GPU-assisted IO 对零散访存也能接近链路带宽**。这一点借自作者的 Strata 工作，本文**未独立 benchmark**。PCIe Gen5 上 miss 代价明显高于 NVLink-C2C。
3. **换来的是并发度与容量，不是单请求速度**。per-token latency 只是"可比"，想让单个请求更快它帮不上忙。
4. **对稠密注意力模型无效**。必须先有稀疏索引器；没有 top-k 选择，就没有"只需常驻 k 条"的立足点。
5. **收益随负载而变**：低并发、短上下文场景下，基线根本没撞到容量墙，分层只会平白添一层 IO。

---

## 8. 一句话串起来

把 sys3 / sys5 和本篇叠起来看，KV 这条主线就完整了：

> **MLA 是「每个 token 存得更小」，NSA / DSA / Quest 是「每次只读重要的 k 条」，CSA 是「先把 token 压小再做稀疏读取」，HCA 是「把很长的历史压成很短的摘要后全读」——而 HiCache 与 HiSparse 是在 KV 已经产生之后，解决「放在哪一层存储」的问题。**

再往前一步区分二者：

- **HiCache** 解决"KV 放 GPU、CPU 还是更低层存储"，是**存储层级**的问题；
- **HiSparse** 解决"稀疏解码下 GPU 里到底需要常驻多少"，是**容量账单与 k 而非 L_ctx 挂钩**的问题。

**减计算**（稀疏注意力）与**减容量**（压缩 + 分层）是两条正交的线，HiSparse 站在后者的下游，也是目前最接近生产落地的那一环（已在 SGLang 上游）。

---

## 9. 与已有文章的衔接

- **KV Cache 压缩方向全景（MHA → HiCache）** → 见 [`sys5-attention-evolution-kvcache`](../sys5-attention-evolution-kvcache)
- **MSA / CSA / HCA 三种注意力改造路线** → 见 [`sys3-attention-evolution`](../sys3-attention-evolution)
- **HiCache 与 NUMA × PCIe × NIC 数据链路** → 见 [`sys1-pcie-numa-nic-hicache`](../sys1-pcie-numa-nic-hicache)
- **SGLang 原理（前缀复用与吞吐优化）** → 见 [`fw3-sglang-internals`](../fw3-sglang-internals)
- **HiCache / Mooncake 的 KV 复用与组合正确性** → 见 [`tr2-vllm-sglang-20260807`](../tr2-vllm-sglang-20260807)

---

## 附：论文信息

```
HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management
Zhiqiang Xie, Zhangheng Huang, Tingwei Huang, Ziyi Xu, Ruiyang Ma, Christos Kozyrakis
Stanford MAST Lab · 2026-08-07 · arXiv:2608.07009
代码状态：已合并至上游 SGLang
```

> 本篇为论文精读与工程解读，不构成任何技术选型或投资建议。数据引自原论文与 alphaxiv 中文解读页（2026-09-07 检索）。
