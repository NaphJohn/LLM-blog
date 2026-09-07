---
title: 'Inference Systems Infrastructure Notes (6): HiSparse — Treating HBM as a Cache to Break the Long-Context Capacity Wall'
description: 'Sparse attention cuts the compute of long-context decoding down to top-k, yet serving stacks still keep the entire KV cache resident in GPU HBM, so memory runs out long before FLOPs do. This post dissects HiSparse from Stanford MAST Lab (arXiv:2608.07009, merged into upstream SGLang): an authoritative full KV copy in host DRAM, a small fixed-B GPU hot cache, a fused RESOLVE CUDA kernel that does hit detection, LRU replacement and host-to-device fetches inside the decode graph, plus exact layer-wise prefetching that hides about half of the remaining IO. Includes measured numbers, preconditions and limits.'
pubDate: 2026-09-07
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys6-hisparse-hierarchical-kv-cache
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

> **The earlier posts (sys3 / sys5) asked "how do we make KV smaller and read less of it"; this one tackles the half that is still too big even after shrinking — demoting GPU HBM from "where KV lives" to "a cache over KV."**

HiSparse (*Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management*, Stanford MAST Lab, arXiv:2608.07009, Aug 2026) makes a single argument: **if top-k sparse attention reads only k KV entries per step, why should HBM hold L_ctx of them?** Move the full KV history to host DRAM, keep a small fixed-size hot cache on the GPU, and add one fused CUDA kernel that does hit detection, LRU replacement and host-to-device fetch in a single launch — and you raise long-context concurrency while keeping the output **bit-for-bit identical, token by token**.

It is **merged into upstream SGLang** and evaluated across three sparse-attention families (DSA, NSA, Quest) on H200, B200 and GH200, with up to **4.7x** higher peak generation throughput on long-context workloads.

---

## 1. Capacity Wall: Sparse Attention Saved FLOPs, Not Memory

Start with the problem. Top-k sparse attention (NSA, DSA, Quest, and friends) sells **cheap compute**: at layer ℓ and step t it selects only k historical positions S_t^(ℓ) — typically a few thousand, not the full context length L_ctx.

But serving stacks carry a hidden assumption that drags everything back:

> **Any past token may be selected by the indexer at some future step.**

To avoid missing one, systems simply keep **the entire KV cache resident in GPU HBM**. The memory bill still grows linearly with L_ctx — so **memory runs out long before compute does**. That is the capacity wall the paper names.

The numbers are blunt:

| Scenario | KV memory footprint | Consequence |
|---|---|---|
| GLM-5.1, single 128K-context request | ≈ **13.09 GB** | An 80GB card serves only a handful concurrently |
| Single 1M-context request | Exceeds total HBM | **Cannot be served at all** |

Note the mismatch: **the compute side already reads only k entries, while the memory side still pays for all of L_ctx.** That is exactly what HiSparse removes — HBM footprint should scale with k, not with L_ctx.

The hard part is that sparse selection is **dynamic**: S_t^(ℓ) changes every step and every layer, so you cannot statically "compress it once and freeze it" the way MLA does.

---

## 2. Two-Level Hierarchy: Authoritative Copy on Host, Hot Cache on GPU

HiSparse never touches model logic — it only changes **where KV records live**:

<div class="arch-fig">
<svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="HiSparse hierarchical KV cache: authoritative full copy in host DRAM, B-slot hot cache in GPU HBM, RESOLVE kernel fetches misses">
  <defs>
    <marker id="arrowEn" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
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
  <text x="46" y="44" class="lab">(1) Host KV pool (Host DRAM, pinned)</text>
  <text x="46" y="64" class="sub">Authoritative full KV history per request, written at prefill</text>
  <text x="46" y="82" class="sub">Capacity tracks host memory, on the order of L_ctx</text>
  <text x="46" y="99" class="tiny">GLM-5.1 128K single request is about 13.09 GB: trivial for host RAM</text>

  <rect x="370" y="20" width="280" height="86" rx="8" class="gpu"/>
  <text x="386" y="44" class="lab">(2) GPU hot cache (HBM, fixed B slots)</text>
  <text x="386" y="64" class="sub">One per request-layer pair, B >= k, holds recently selected KV</text>
  <text x="386" y="82" class="sub">HBM usage = N_l x B x W_KV x s, decoupled from L_ctx</text>
  <text x="386" y="99" class="tiny">Up to 30x HBM savings for GLM-5.1 at 128K</text>

  <rect x="175" y="160" width="330" height="72" rx="8" class="kern"/>
  <text x="191" y="184" class="lab">(3) RESOLVE fused CUDA kernel (inside decode graph)</text>
  <text x="191" y="204" class="sub">Stage, Mark, Scan, Fetch, Publish in one launch</text>
  <text x="191" y="222" class="tiny">GPU threads pull KV from pinned host memory via ld.global.nc.v2.b64</text>

  <line x1="150" y1="106" x2="230" y2="158" stroke="#8B4513" stroke-width="1.6" marker-end="url(#arrowEn)"/>
  <text x="140" y="140" class="sub">miss fetch</text>
  <line x1="500" y1="160" x2="500" y2="110" stroke="#8B4513" stroke-width="1.6" marker-end="url(#arrowEn)"/>
  <text x="510" y="140" class="sub">write back / page table</text>

  <rect x="175" y="268" width="330" height="52" rx="8" class="gpu"/>
  <text x="191" y="290" class="lab">(4) Sparse attention kernel</text>
  <text x="191" y="308" class="sub">Receives physical device offsets and computes as usual, reading exactly what full residency would</text>
  <line x1="340" y1="232" x2="340" y2="266" stroke="#8B4513" stroke-width="1.6" marker-end="url(#arrowEn)"/>

  <text x="30" y="352" class="lab">Key invariant</text>
  <text x="30" y="372" class="sub">Only physical placement changes: no model or math change, so output is unchanged (exact)</text>
  <text x="30" y="390" class="warn">The one and only price: host-to-device IO bandwidth</text>
</svg>
<p class="cap">Figure 1: The two-level hierarchy. Host DRAM holds the authoritative copy; GPU HBM degrades into a hot cache managed by RESOLVE.</p>
</div>

Three components, three jobs:

1. **Host KV pool** — KV records produced during prefill go straight into pinned host memory. This is the only authoritative copy.
2. **GPU hot cache** — B slots per request-layer pair holding recently selected KV records. To guarantee the current attention can always proceed, **B ≥ k**.
3. **Metadata** — compact page tables on the GPU map a logical token position to a physical slot or mark it host-only, plus recency bits for LRU.

Per-request HBM consumption becomes:

```
HBM usage = N_l x B x W_KV x s
```

where N_l is layer count, W_KV is KV elements per token, and s is bytes per element. **L_ctx is gone** — that is what "decoupling decoding throughput from GPU memory capacity" actually means.

### 2.1 How Slots Are Actually Allocated (DRAM to HBM Mapping)

<div class="arch-fig">
<svg viewBox="0 0 680 470" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="HiSparse slot allocation: host DRAM stores everything by token, GPU HBM gives each request-layer pair B slots, and a page table maps logical positions to physical slots">
  <defs>
    <marker id="arAllocEn" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
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

  <text x="20" y="24" class="lab">Slot allocation: DRAM stores everything per token; HBM grants B slots per request-layer pair</text>

  <rect x="20" y="44" width="250" height="250" rx="10" class="host"/>
  <text x="34" y="68" class="lab">(1) Host DRAM (authoritative full copy)</text>
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
  <text x="34" y="228" class="tiny">Every (request, layer, pos) K/V pair is retained</text>
  <text x="34" y="246" class="tiny">Written directly at prefill; never evicted, never LRU'd</text>
  <text x="34" y="264" class="tiny">Capacity follows host RAM: a 13 GB, 128K request is fine</text>
  <text x="34" y="284" class="tiny">Addressed contiguously by logical position; the only authority</text>

  <rect x="290" y="44" width="90" height="250" rx="10" class="pt"/>
  <text x="335" y="68" class="lab" text-anchor="middle">(2) Page table</text>
  <text x="335" y="88" class="tiny" text-anchor="middle">logical pos to slot</text>
  <rect x="300" y="100" width="70" height="20" rx="4" class="slot"/><text x="335" y="114" class="cell" text-anchor="middle">pos to slot</text>
  <rect x="300" y="126" width="70" height="20" rx="4" class="slot"/><text x="335" y="140" class="cell" text-anchor="middle">pos to slot</text>
  <rect x="300" y="152" width="70" height="20" rx="4" class="slot"/><text x="335" y="166" class="cell" text-anchor="middle">pos to slot</text>
  <rect x="300" y="178" width="70" height="20" rx="4" class="slot"/><text x="335" y="192" class="cell" text-anchor="middle">pos to slot</text>
  <text x="335" y="224" class="tiny" text-anchor="middle">hit</text>
  <text x="335" y="240" class="tiny" text-anchor="middle">means in slot</text>
  <text x="335" y="260" class="tiny" text-anchor="middle">miss</text>
  <text x="335" y="276" class="tiny" text-anchor="middle">means host only</text>

  <rect x="400" y="44" width="260" height="250" rx="10" class="gpu"/>
  <text x="414" y="68" class="lab">(3) GPU HBM (hot cache, fixed B slots)</text>
  <text x="414" y="86" class="sub">One set per request-layer pair, with B >= k</text>
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
  <text x="414" y="226" class="tiny">B = 6 here is illustrative; real B is a few multiples of k (thousands)</text>
  <text x="414" y="244" class="tiny">green = hot slot hit this step, red = LRU victim (next miss evicts it)</text>
  <text x="414" y="262" class="tiny">position 17 can occupy a slot in every layer: granularity is request x layer</text>
  <text x="414" y="284" class="tiny">Each slot = 1 KV record + logical position + LRU recency bits</text>

  <line x1="270" y1="150" x2="286" y2="150" stroke="#8B4513" stroke-width="1.6" marker-end="url(#arAllocEn)"/>
  <line x1="380" y1="150" x2="396" y2="150" stroke="#8B4513" stroke-width="1.6" marker-end="url(#arAllocEn)"/>
  <text x="271" y="140" class="tiny">lookup</text>

  <rect x="20" y="310" width="640" height="70" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="36" y="336" class="form">HBM usage = N_l x B x W_KV x s, independent of L_ctx</text>
  <text x="36" y="358" class="sub">Allocation granularity is request x layer, not request: different layers pick different entries, so each needs its own slots.</text>
  <text x="36" y="376" class="sub">The constraint B >= k guarantees the k entries selected this step always fit, so attention can always proceed.</text>

  <rect x="20" y="394" width="640" height="62" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="36" y="418" class="sub">Example: 128K context, k = 2048, take B = 2k = 4096.</text>
  <text x="36" y="440" class="sub">Per layer that is 4096 resident entries instead of 131072, about 3.1%, roughly a <tspan font-weight="700">32x saving</tspan> (matching the 30x the paper reports).</text>
  <text x="36" y="458" class="sub">Push the context to 1M and B stays 4096: <tspan font-weight="700">HBM usage does not grow</tspan>. That is what decoupling means.</text>
</svg>
<p class="cap">Figure 3: How DRAM and HBM allocation correspond. The DRAM side stores everything contiguously by (request, layer, pos); the HBM side grants each request-layer pair a fixed set of B slots, with a page table mapping logical positions to physical slots and LRU deciding who gets evicted.</p>
</div>

Three points worth memorizing on their own:

1. **Allocation granularity is request x layer, not request.** Different layers select different entries, so each layer needs its own set of slots — which is why N_l appears in the formula.
2. **DRAM and HBM are decoupled by the page table.** The DRAM side is addressed contiguously by logical position; HBM slots have no fixed relation to logical position and simply hold "the recently selected ones," with all mapping in the page table.
3. **B is a constant, independent of context length.** This is exactly where the capacity wall comes down: as context grows from 128K to 1M, the DRAM side grows linearly (fine, host memory is large) while the HBM side does not move at all.

---

## 3. RESOLVE: Five Stages in One Launch

Hierarchical caching is an old idea. The hard part is **making a miss cheap enough to tolerate**, because sparse selection access patterns are scattered, and a conventional CPU-side paging path exposes the full latency.

HiSparse writes a single fused CUDA kernel named `RESOLVE`, **launched once per sparse layer**, doing five things in parallel across GPU threads:

| Stage | What it does | Why it matters |
|---|---|---|
| **Stage** | Loads the indexer-selected logical positions into a shared-memory hash table | Later comparisons stay on-chip |
| **Mark** | Checks existing GPU cache slots against the hash table to identify hits and evictable slots | Hit detection is pure metadata work |
| **Scan** | Parallel scan over slots, updates LRU metadata, picks victims for this step's misses | Prefix-sum-style slot allocation |
| **Fetch** | Miss threads use **vectorized non-coherent loads** (`ld.global.nc.v2.b64`) to pull KV from pinned host memory into their assigned slots | **GPU-assisted IO: GPU threads fetch for themselves** |
| **Publish** | Updates page tables and hands physical device offsets to the sparse attention kernel | Downstream kernel is unaware |

**GPU-assisted IO is the real technical contribution here.** Letting GPU threads read host memory directly saturates PCIe or NVLink bandwidth even for scattered access, bypassing the long "CPU notices a fault, notifies, then copies" path. And the whole flow runs **inside the decode CUDA Graph**, so dynamic control flow never breaks graph capture.

---

## 4. Locality: Why a Small B Is Enough

Every cache lives on locality, and HiSparse shows sparse selection has two exploitable properties:

- **Temporal locality** — consecutive decoding steps **re-select many of the same tokens**;
- **Cross-layer correlation** — different layers tend to attend to nearby regions of the context.

Measured miss rates on GLM-5.1 (k = 2048) make the point:

<div class="arch-fig">
<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Top-k miss rate versus GPU cache size B as a multiple of k">
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

  <text x="70" y="22" class="lab">GLM-5.1, k = 2048: top-k miss rate falls fast as the cache grows</text>
  <text x="80" y="244" class="sub">Doubling from k to 2k halves the miss rate; 4k halves it again. B only needs a few multiples of k.</text>
</svg>
<p class="cap">Figure 2: The locality dividend under LRU. A cache a few times larger than k already pushes misses into the single digits.</p>
</div>

### Exact Layer-Wise Prefetching

There is a sharper trick. Some models (for example GLM-5.2) **share indexer output across groups of layers** — once an anchor layer fixes its selection, the system immediately knows the selections of the layers that share it.

HiSparse turns this into **exact layer-wise prefetching**:

1. The anchor layer computes its selection, which yields the shared layers' miss plans deterministically;
2. A **background copy-only kernel** replays those future layers' miss plans;
3. Host-to-device transfer then **overlaps with the current layer's compute**.

Result: **roughly half of the remaining IO latency is hidden.** Note the word *exact* — this is not a heuristic guess but deterministic knowledge, so it introduces no correctness risk.

---

## 5. Why It Is Exact and Indexer-Agnostic

These two properties are why it can go straight into production:

- **Exact** — it changes only the **physical placement** of KV records: no model surgery, no change to attention math, no approximate recall. **Model output is unchanged token by token**, which separates it fundamentally from "trade memory for approximate retrieval" schemes.
- **Indexer-agnostic** — it does not care how you picked those k positions. The paper evaluates **DSA, NSA and Quest** and all three benefit.

In short, HiSparse is a **general-purpose memory backend for sparse attention**: if your model already has a sparse indexer, it just plugs in.

---

## 6. Measured Results

| Setup | Result |
|---|---|
| DeepSeek-V4-Flash (NSA), 2x B200, 64 concurrent requests | **2.1x** generation throughput |
| Qwen3 with Quest, GH200, 200K input length | up to **4.7x** generation throughput |
| High load (prefill and decode share the GPU) | **TTFT drops significantly** (new requests are no longer blocked by HBM exhaustion) |
| Per-token latency | **Comparable** to baseline (not faster, but not slower) |
| No-IO oracle experiment | The resolution mechanism adds **no measurable per-token cost** |

That last row matters: **host-device IO is the only, and the entire, price of this design.** Which means link bandwidth directly sets the payoff:

> **Link sensitivity**: KV fetch time on GH200 (NVLink-C2C) is nearly **4x lower** than on H200 (PCIe Gen5), allowing a smaller GPU cache and higher concurrency.

---

## 7. When Not to Use It (Limits and Preconditions)

This section matters more than the speedups:

1. **It assumes host DRAM is much larger than GPU HBM.** On **Grace-based GB200 / GB300**, CPU and GPU share unified memory and the host side has no capacity advantage — the whole offload-for-capacity argument **does not hold**. The paper lists this as a fundamental limitation.
2. **It relies on GPU-assisted IO reaching near-link bandwidth for scattered fetches.** That is borrowed from the authors' Strata work and is **not independently benchmarked** in this paper. Misses cost noticeably more on PCIe Gen5 than on NVLink-C2C.
3. **It buys concurrency and capacity, not single-request speed.** Per-token latency is merely comparable; it will not make one request finish faster.
4. **It does nothing for dense-attention models.** You need a sparse indexer first — without top-k selection there is no footing for "only k entries must be resident."
5. **Payoff varies with load.** At low concurrency or short context the baseline never hits the capacity wall, and the hierarchy only adds an IO hop.

---

## 8. The One-Line Thread, Completed

Stack sys3, sys5 and this post together and the KV story is whole:

> **MLA makes each token store smaller; NSA / DSA / Quest read only the important k entries; CSA compresses tokens first and then does sparse reads; HCA compresses very long history into a short summary and reads all of it — while HiCache and HiSparse answer a different question: once KV exists, which tier of the memory hierarchy should hold it?**

One more distinction:

- **HiCache** asks "should KV sit on GPU, CPU, or even lower tiers" — a **storage-tier** question;
- **HiSparse** asks "under sparse decoding, how much must actually be resident on the GPU" — making the **capacity bill scale with k rather than L_ctx**.

**Cutting compute** (sparse attention) and **cutting capacity** (compression plus hierarchy) are orthogonal axes. HiSparse sits at the far end of the second one, and is currently the closest to production (it is in upstream SGLang).

---

## 9. Links to Earlier Posts

- **Full KV compression landscape (MHA to HiCache)** — see [`sys5-attention-evolution-kvcache`](/blog/sys5-attention-evolution-kvcache)
- **MSA / CSA / HCA: three attention redesign routes** — see [`sys3-attention-evolution`](/blog/sys3-attention-evolution)
- **HiCache and the NUMA x PCIe x NIC data path** — see [`sys1-pcie-numa-nic-hicache`](/blog/sys1-pcie-numa-nic-hicache)
- **SGLang internals (prefix reuse and throughput optimization)** — see [`fw3-sglang-internals`](/blog/fw3-sglang-internals)
- **HiCache / Mooncake KV reuse and compositional correctness** — see [`tr2-vllm-sglang-20260807`](/blog/tr2-vllm-sglang-20260807)

---

## Appendix: Paper

```
HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management
Zhiqiang Xie, Zhangheng Huang, Tingwei Huang, Ziyi Xu, Ruiyang Ma, Christos Kozyrakis
Stanford MAST Lab - 2026-08-07 - arXiv:2608.07009
Code: merged into upstream SGLang
```

> This post is a close reading and engineering interpretation of the paper. It is not technical or investment advice. Figures are taken from the paper and its alphaxiv reading page (retrieved 2026-09-07).
