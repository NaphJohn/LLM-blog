---
title: 'Inference Systems Infrastructure Notes (5): From MHA to HiCache — The Full KV Cache Compression Landscape'
description: 'The 2026 KV cache evolution of long-context models as one complete chain: MHA (no compression), GQA (fewer heads), MLA (compress the feature dimension), DSA (select important tokens), CSA (compress tokens then sparsify), HCA (compress tokens extremely then go dense), and HiCache (decide which tier stores the result). The key shift is that compression moved from the feature dimension to the token dimension, and HCA and HiCache are two stacked layers of optimization, not competitors.'
pubDate: 2026-08-26
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys5-attention-evolution-kvcache
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

> **MHA stores everything, GQA cuts KV heads, MLA compresses along the feature dimension, DSA sparsely selects along the token dimension, CSA compresses tokens first and then sparsifies, HCA compresses tokens extremely and then goes dense, and HiCache decides which tier (GPU / CPU / SSD) holds whatever KV is left.**

The single most important change across this line is that **the direction of KV compression shifted** — from "store each token more compactly" (feature dimension) to "shorten the long history itself" (token dimension). The final step, HiCache, is not compression at all: it answers a different question — **the KV already exists, so where should it live?**

---

## 1. The Complete Chain

<div class="fig">
<svg viewBox="0 0 680 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The KV cache evolution from MHA to HiCache">
  <defs>
    <marker id="arrEn5" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="24" font-size="15" font-weight="700" fill="#1a1a1a">KV cache evolution: compression moves from the feature dimension to the token dimension</text>

  <rect x="20" y="44" width="120" height="34" rx="6" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="80" y="66" font-size="13" font-weight="700" fill="#374151" text-anchor="middle">MHA</text>
  <text x="156" y="58" font-size="12" fill="#444">Stores full K/V per token, no compression</text>
  <text x="156" y="74" font-size="11" fill="#9ca3af">KV length equals context length N</text>

  <rect x="20" y="92" width="120" height="34" rx="6" fill="#ecfeff" stroke="#06b6d4"/>
  <text x="80" y="114" font-size="13" font-weight="700" fill="#0e7490" text-anchor="middle">GQA</text>
  <text x="156" y="106" font-size="12" fill="#444">Several Q heads share one KV head, fewer KV heads</text>
  <text x="156" y="122" font-size="11" fill="#9ca3af">Token count still N; only head count drops</text>

  <rect x="20" y="140" width="120" height="34" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="80" y="162" font-size="13" font-weight="700" fill="#1e40af" text-anchor="middle">MLA</text>
  <text x="156" y="154" font-size="12" fill="#444">Each token compressed into a low-rank latent c (feature dim)</text>
  <text x="156" y="170" font-size="11" fill="#9ca3af">Still N entries, but each is much smaller, about 1/64</text>

  <rect x="20" y="188" width="120" height="34" rx="6" fill="#eef2ff" stroke="#6366f1"/>
  <text x="80" y="210" font-size="13" font-weight="700" fill="#4338ca" text-anchor="middle">DSA</text>
  <text x="156" y="202" font-size="12" fill="#444">An indexer picks the top-k important tokens (token dim)</text>
  <text x="156" y="218" font-size="11" fill="#9ca3af">Saves compute, but KV still must be stored</text>

  <rect x="20" y="236" width="120" height="34" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="80" y="258" font-size="13" font-weight="700" fill="#1e40af" text-anchor="middle">CSA</text>
  <text x="156" y="250" font-size="12" fill="#444">First 4 tokens to 1 KV block, then top-k sparsity</text>
  <text x="156" y="266" font-size="11" fill="#9ca3af">1M to 250K compressed KV, then select 1024</text>

  <rect x="20" y="284" width="120" height="34" rx="6" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="80" y="306" font-size="13" font-weight="700" fill="#92400e" text-anchor="middle">HCA</text>
  <text x="156" y="298" font-size="12" fill="#444">128 tokens to 1 entry, then dense over all entries</text>
  <text x="156" y="314" font-size="11" fill="#9ca3af">1M to about 7800 entries; top-k no longer needed</text>

  <rect x="20" y="332" width="120" height="34" rx="6" fill="#faf5ff" stroke="#a855f7"/>
  <text x="80" y="354" font-size="13" font-weight="700" fill="#7e22ce" text-anchor="middle">HiCache</text>
  <text x="156" y="346" font-size="12" fill="#444">Compressed KV stored in tiers (systems dimension)</text>
  <text x="156" y="362" font-size="11" fill="#9ca3af">GPU HBM to CPU DRAM to SSD NVMe</text>

  <line x1="80" y1="78" x2="80" y2="92" stroke="#9ca3af" marker-end="url(#arrEn5)"/>
  <line x1="80" y1="126" x2="80" y2="140" stroke="#9ca3af" marker-end="url(#arrEn5)"/>
  <line x1="80" y1="174" x2="80" y2="188" stroke="#9ca3af" marker-end="url(#arrEn5)"/>
  <line x1="80" y1="222" x2="80" y2="236" stroke="#9ca3af" marker-end="url(#arrEn5)"/>
  <line x1="80" y1="270" x2="80" y2="284" stroke="#9ca3af" marker-end="url(#arrEn5)"/>
  <line x1="80" y1="318" x2="80" y2="332" stroke="#a855f7" marker-end="url(#arrEn5)"/>
</svg>
<p class="cap">Figure: the full evolution from MHA to HiCache. Colors mark the dimension being compressed: blue is the feature dimension (MLA), indigo and orange are the token dimension (DSA / CSA / HCA), gray is no compression (MHA), cyan is fewer heads (GQA), and purple is the systems layer (HiCache).</p>
</div>

---

## 2. Where Each Step Actually Cuts

| Mechanism | Where it cuts | Compression ratio | What it solves | What it does not |
|---|---|---|---|---|
| **MHA** | nothing | 1x | Baseline: exact | KV grows linearly with N |
| **GQA** | KV head count | 4x to 8x | Fewer KV heads, less memory | Token count is still N |
| **MLA** | feature dimension (smaller per token) | about 1/64 | Each cached token is smaller | Still N entries |
| **DSA** | token dimension (read the important ones) | compute down | Only top-k is computed, saving FLOPs | KV capacity unchanged (saves compute, not memory) |
| **CSA** | token dimension (compress, then select) | 4x + top-k | KV down to 1/4, then sparsified | Still needs an indexer to pick blocks |
| **HCA** | token dimension (extreme compression, then dense) | 128x | KV compressed to the limit, dense is faster | Per-token information heavily abstracted |
| **HiCache** | storage tier (where to put it) | none | Multi-tier storage, less GPU pressure | Performs no compression at all |

---

## 3. The Key Point: The Compression Direction Changed

People often lump MLA, DSA, CSA and HCA together, but their **compression directions are completely different**:

- **MLA = store each token more compactly** (feature / head dimension)

  ```text
  T1 -> latent1      <- still N entries
  T2 -> latent2
  ...
  TN -> latentN
  ```

  Only "each entry gets smaller"; the number of entries does not change.

- **HCA = the token count itself shrinks** (token dimension)

  ```text
  T1..T128   -> C1      <- N/128 entries
  T129..T256 -> C2
  ```

  This is "compress a very long history into a very short summary."

- **DSA = read only the important tokens each time** (selection along the token dimension, not compression)

  ```text
  1M tokens -> indexer -> top-k = 2048 -> attend to those only
  ```

  It does not reduce how much KV is stored, only how much is **computed** — so "how many tokens you verify or select" and "how many tokens the KV cache itself holds" are two separate questions.

- **CSA = shrink tokens first, then read sparsely**

  ```text
  1M -> 4x compression -> 250K compressed KV -> top-k -> 1024 -> attention
  ```

  In other words: compress coarsely first, then pick the important parts.

- **HCA = compress the long history into a short summary, then read all of it**

  ```text
  1M -> 128x compression -> 7800 entries -> dense attention
  ```

  It trades "harsher compression" for "sparse retrieval" — because at 7800 items, going dense is faster than irregular sparsity.

**One line to tell them apart**: MLA cuts along the **feature dimension** (each token is smaller), while DSA / CSA / HCA cut along the **token dimension** (history gets shorter, or only the important part is read). That is the key to reading the whole evolution.

---

## 4. Where Does MSA Fit?

`sys3` already covered the **comparison and cooperation of MSA / CSA / HCA** on its own: MSA only changes "where you look" (sparse selection, no KV compression), CSA compresses then selects, and HCA compresses extremely then goes dense. This post places them in a wider context — MSA is the natively-in-model version of the DSA idea, whereas DeepSeek V4 follows the CSA / HCA route of "compress first, then read." They are not alternatives; they are layered collaborators.

---

## 5. HCA Is Not HiCache: Two Stacked Layers

This is the easiest thing to confuse. **HCA and HiCache are not competitors; they are upstream and downstream of each other.**

```text
Layer 1: model compression (architecture)
MHA -> MLA -> CSA / HCA
             |
             v
         KV size drops (1M tokens: ~100GB down to a few GB)

Layer 2: system tiering (systems)
HiCache
             |
             v
     GPU HBM / CPU DRAM / SSD NVMe tiering
             |
             v
     GPU HBM pressure falls further
```

- **HCA reduces how much KV representation exists**: 1M tokens of KV goes from roughly 100 GB down to a few GB (cost reduction at the architecture layer).
- **HiCache manages what KV is left**: of those few GB, the hot part sits in GPU HBM, the warm part in CPU DRAM, the cold part on SSD, moved with H2D / D2H as needed (tiered management at the systems layer).

<div class="fig">
<svg viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="HCA and HiCache as two stacked layers of optimization">
  <defs>
    <marker id="arrEn5b" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">HCA (architecture compression) and HiCache (system tiering) are upstream and downstream</text>

  <rect x="20" y="48" width="640" height="56" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="35" y="72" font-size="13" font-weight="700" fill="#92400e">Layer 1 - Model architecture compression</text>
  <text x="35" y="92" font-size="11.5" fill="#444">MHA to MLA to CSA / HCA: shrink the KV representation, 1M tokens from ~100GB down to a few GB</text>

  <line x1="340" y1="104" x2="340" y2="124" stroke="#9ca3af" marker-end="url(#arrEn5b)"/>
  <text x="350" y="120" font-size="11" fill="#9ca3af">KV cache</text>

  <rect x="20" y="128" width="640" height="56" rx="8" fill="#faf5ff" stroke="#a855f7"/>
  <text x="35" y="152" font-size="13" font-weight="700" fill="#7e22ce">Layer 2 - System tiering (HiCache)</text>
  <text x="35" y="172" font-size="11.5" fill="#444">GPU HBM (hot) / CPU DRAM (warm) / SSD NVMe (cold), moved with H2D/D2H on demand, relieving GPU pressure</text>

  <rect x="40" y="200" width="170" height="44" rx="6" fill="#ecfdf5" stroke="#10b981"/>
  <text x="125" y="222" font-size="12" font-weight="700" fill="#047857" text-anchor="middle">GPU HBM</text>
  <text x="125" y="238" font-size="10.5" fill="#047857" text-anchor="middle">L1 hot KV</text>

  <rect x="255" y="200" width="170" height="44" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="340" y="222" font-size="12" font-weight="700" fill="#1d4ed8" text-anchor="middle">CPU DRAM</text>
  <text x="340" y="238" font-size="10.5" fill="#1d4ed8" text-anchor="middle">L2 warm KV</text>

  <rect x="470" y="200" width="170" height="44" rx="6" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="555" y="222" font-size="12" font-weight="700" fill="#374151" text-anchor="middle">SSD NVMe</text>
  <text x="555" y="238" font-size="10.5" fill="#374151" text-anchor="middle">L3 cold KV</text>

  <line x1="210" y1="222" x2="253" y2="222" stroke="#9ca3af" marker-end="url(#arrEn5b)"/>
  <line x1="425" y1="222" x2="468" y2="222" stroke="#9ca3af" marker-end="url(#arrEn5b)"/>
</svg>
<p class="cap">Figure: HCA reduces the amount of KV representation; HiCache manages which tier holds it. The two are orthogonal and stack, forming two layers of optimization from architecture down to systems.</p>
</div>

> Incidentally, DeepSeek-V4 KV management is indeed designed separately for "CSA / HCA compressed KV" and "SWA uncompressed KV" — which shows that HCA compression and systems-level KV management like HiCache can be joined up. This distinction matters most if you are weighing **whether GLM-5.2/5.3 DSA, HiCache, and DeepSeek-V4 CSA/HCA are the same idea**: **DSA / CSA / HCA solve "how attention reads"; HiCache solves "where the KV cache lives."**

---

## 6. The One-Line Thread

> **MLA makes each token store smaller; DSA reads only the important tokens; CSA compresses tokens first and then reads sparsely; HCA compresses a very long history into a very short summary and reads all of it; and HiCache, once all that KV exists, answers whether it goes on GPU, CPU or a lower storage tier.**

---

## 7. Links to Earlier Posts

- **MLA at the operator level** — see [`op1-mla-operator`](/blog/op1-mla-operator) (includes the MHA baseline and c_kv pseudocode; a GQA/MQA operator comparison is still to come)
- **CSA / HCA comparison and cooperation** — see [`sys3-attention-evolution`](/blog/sys3-attention-evolution) (MSA / CSA / HCA comparison plus the V4 layering diagram)
- **DeepSeek V4 architecture deep dive (MLA to NSA to DSA to CSA+HCA)** — see [`mm5-efficiency-frontier`](/blog/mm5-efficiency-frontier), section 2
- **HiCache and the NUMA / PCIe / NIC topology** — see [`sys1-pcie-numa-nic-hicache`](/blog/sys1-pcie-numa-nic-hicache)
- **Long-context training (Ring Attention)** — see [`sys2-ring-attention`](/blog/sys2-ring-attention)

This post is the first attempt to pull the "compression direction" thread scattered across those posts into one complete chain from MHA to HiCache.
