---
title: "Inference Systems Infrastructure Notes (12): Mooncake — A KVCache-centric PD-Disaggregated Architecture, and Three Ways It Composes with vLLM"
description: "The core optimization of Mooncake in one line: storage-compute disaggregation built around KVCache. Four architecture diagrams show which layer PD disaggregation actually lives at — in Mooncake the Conductor only schedules while the split happens at the instance-role layer; compare that with the vLLM Disagg Proxy plus KV Connector design and the non-PD monolithic baseline; finally, how Mooncake plugs into vLLM through the Connector interface as an external KV layer."
pubDate: 2026-10-13
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys12-mooncake-kvcache-centric-serving
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

The core optimization of Mooncake (the inference system behind Moonshot Kimi, from the FAST'25 paper "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving") in one line: **storage-compute disaggregation built around KVCache** — split "computing KV" (prefill) from "consuming KV" (decode) into two independent cluster types, store and reuse prefixes in a tiered distributed KV pool, and let KV flow cheaply between hardware tiers. That simultaneously attacks idle compute, memory capacity, and redundant computation.

This post uses four architecture diagrams to nail down three things: what Mooncake itself looks like, which layer PD disaggregation actually lives at (a widely repeated misconception is "in the router"), and how Mooncake attaches to vLLM as an external KV layer.

## 1. The Core Principle: Build the System Around KVCache

Traditional inference systems are built around **model weights**: a request arrives, find a GPU, run it, done. Mooncake flips the priority — KVCache is a first-class citizen and every design decision serves it. Why?

### 1.1 Prefill and decode have completely different load profiles

| Stage | What it does | Bottleneck |
|-------|--------------|------------|
| Prefill | Computes the entire KV for a prompt in one pass | **Compute-bound**, saturates GPU FLOPs |
| Decode | Reads the full KV for every generated token | **Memory-bound**, limited by bandwidth, not FLOPs |

Colocate them and they drag on each other: prefill compute spikes inflate decode latency, while decode idle compute cannot cover the prefill gap. Separate them and each cluster saturates its own bottleneck resource — that is the motivation for PD disaggregation (sys8 covered the principle; this post looks at an industrial implementation).

### 1.2 Prefix reuse is a goldmine

Real workloads repeat prefixes heavily: system prompts, multi-turn history, few-shot templates, shared sections of long documents. Compute the KV for a common prefix once, store it, and every later request reuses it — prefill compute drops by multiples and TTFT collapses with it. **Whether these KVs can be cached and found across requests and instances sets the practical ceiling of the whole system.**

### 1.3 One machine of HBM cannot hold all the KV

KV Cache grows linearly with context length (sys9 ran the numbers), so single-card HBM runs out fast. The natural answer is tiering: GPU HBM for hot data, CPU DRAM as the main tier, SSD for cold prefixes — at the price of moving data, which in turn demands a transfer layer fast enough to hide the tiering.

In one sentence: **split the KV computers from the KV consumers (PD disaggregation), store KV in one pool (distributed store), move it with RDMA (transfer engine), and route requests to wherever KV already lives (KV-aware routing)** — assemble those four pieces and you get Mooncake.

## 2. The Mooncake Architecture, End to End

<div class="fig">
<svg viewBox="0 0 680 580" width="100%" font-size="13" font-weight="400" role="img" aria-label="Mooncake PD-disaggregated architecture">
<title>Figure 1: The Mooncake PD-disaggregated architecture</title>
<defs>
<marker id="ar1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<rect x="240" y="40" width="200" height="44" rx="8" fill="#f6f8fa" stroke="#30363d" stroke-width="0.5"/>
<text x="340" y="62" font-size="13" fill="#1a1a1a" text-anchor="middle" dominant-baseline="central">Request</text>
<path d="M340 84 L340 116" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)"/>

<rect x="190" y="120" width="300" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="140" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Conductor (Global Scheduler)</text>
<text x="340" y="158" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">KV-aware routing · routes only, never moves KV</text>
<path d="M270 176 L215 224" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)"/>
<path d="M410 176 L465 224" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)"/>

<rect x="110" y="230" width="200" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="210" y="250" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Prefill-only Instance</text>
<text x="210" y="268" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">role fixed at launch · computes KV only</text>
<rect x="370" y="230" width="200" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="470" y="250" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Decode-only Instance</text>
<text x="470" y="268" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">role fixed at launch · generates only</text>

<rect x="240" y="330" width="200" height="56" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="340" y="350" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Transfer Engine</text>
<text x="340" y="368" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">RDMA / NVLink KV transfer</text>
<path d="M215 286 L295 324" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)"/>
<path d="M465 286 L385 324" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)"/>
<text x="228" y="302" font-size="11" fill="#6b7280" text-anchor="middle" dominant-baseline="central">KV transfer</text>

<rect x="80" y="430" width="520" height="110" rx="16" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="340" y="450" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Mooncake Store · Distributed KV Pool</text>
<rect x="100" y="468" width="146" height="50" rx="8" fill="#f6f8fa" stroke="#10b981" stroke-width="0.5"/>
<text x="173" y="485" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">GPU HBM</text>
<text x="173" y="503" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">low latency · small</text>
<rect x="266" y="468" width="146" height="50" rx="8" fill="#f6f8fa" stroke="#10b981" stroke-width="0.5"/>
<text x="339" y="485" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">CPU DRAM</text>
<text x="339" y="503" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">mid capacity · main tier</text>
<rect x="432" y="468" width="146" height="50" rx="8" fill="#f6f8fa" stroke="#10b981" stroke-width="0.5"/>
<text x="505" y="485" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">SSD</text>
<text x="505" y="503" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">high capacity · cold prefixes</text>
<path d="M340 386 L340 424" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar1)" marker-start="url(#ar1)"/>

<text x="340" y="562" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">The PD split lives in the instance-role layer; the Conductor only schedules</text>
</svg>
</div>

Five roles, one job each:

| Role | Responsibility | Key point |
|------|----------------|-----------|
| **Conductor** | Global scheduler; receives requests, picks instances | KV-aware routing by predicted cache hit; **routes only, never touches the data plane** |
| **Prefill-only instance** | Computes KV, then hands off | Compute-bound; batches are packed for FLOP utilization |
| **Decode-only instance** | Takes KV and generates token by token | Memory-bound; latency-sensitive |
| **Transfer Engine** | Moves KV between instances and to/from the Store | General RDMA-based transfer library with batched send/recv |
| **Mooncake Store** | Distributed KV pool with put/get APIs | HBM / DRAM / SSD tiers, token-granular addressing |

The full journey of a request: the Conductor first checks the Store — **a prefix hit skips prefill for that span entirely**; the miss is scheduled to some P instance, and the computed KV is written to the Store and handed to the chosen D instance via the Transfer Engine, while D streams new KV back asynchronously as it generates. Incremental decode KV lands in DRAM/SSD for reuse by later turns.

One detail worth noting: Mooncake instances are not limited to P and D roles — you can also run "does both" normal instances and mix them by cluster load. PD disaggregation is a **deployment strategy**, not a hard-coded topology.

## 3. Which Layer Does PD Disaggregation Actually Live At

This is the key to understanding how Mooncake and vLLM differ, and it answers a common misconception directly: **"Is PD disaggregation done inside the router?" — No.**

### 3.1 Mooncake: the split is in the instance-role layer; scheduling is in the Conductor

In Figure 1, the Conductor only answers "which instance should this request go to" — it does not split requests or move KV. The real phase split happens **at the instance layer**: each process is designated prefill-only or decode-only at launch, so the phase boundary is drawn by "what this instance is capable of doing"; KV movement is entirely on the data plane, done by the Transfer Engine. The router points the way, instances divide the labor, the transfer layer moves the bytes — three separate concerns.

### 3.2 vLLM: proxy coordination + separate processes + connector transfer

The vLLM flavor is a different combo, with no global Conductor:

<div class="fig">
<svg viewBox="0 0 680 525" width="100%" font-size="13" font-weight="400" role="img" aria-label="vLLM PD-disaggregated architecture">
<title>Figure 2: The vLLM PD-disaggregated architecture</title>
<defs>
<marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<rect x="240" y="40" width="200" height="44" rx="8" fill="#f6f8fa" stroke="#30363d" stroke-width="0.5"/>
<text x="340" y="62" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Client</text>
<path d="M340 84 L340 116" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>

<rect x="200" y="120" width="280" height="44" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="142" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">API Server (OpenAI-compatible)</text>
<path d="M340 164 L340 196" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>

<rect x="200" y="200" width="280" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="220" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Disagg Proxy</text>
<text x="340" y="238" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">two-phase coordination: P first, then D</text>
<path d="M270 256 L215 304" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>
<path d="M410 256 L465 304" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>

<rect x="110" y="310" width="200" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="210" y="330" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">P Instance · Producer</text>
<text x="210" y="348" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">separate process · prefill only</text>
<rect x="370" y="310" width="200" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="470" y="330" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">D Instance · Consumer</text>
<text x="470" y="348" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">separate process · decode only</text>

<rect x="110" y="420" width="460" height="56" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="340" y="440" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">KV Connector</text>
<text x="340" y="458" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">NIXL / Mooncake · chosen via kv_transfer_config</text>
<path d="M205 366 L250 414" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>
<path d="M475 366 L430 414" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar2)"/>
<text x="222" y="396" font-size="11" fill="#6b7280" text-anchor="middle" dominant-baseline="central">KV transfer</text>

<text x="340" y="500" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">PD split = proxy routing + separate P/D processes + KV Connector transfer (responses stream back through the proxy)</text>
</svg>
</div>

Three parts, one job each: at launch, `--kv-transfer-config` declares each process as `kv_producer` (P) or `kv_consumer` (D); the **Disagg Proxy coordinates the two-phase flow** — send the request to P to compute KV, wait for the KV to be ready, then send it to D to generate, with responses streaming back through the proxy; which physical path the KV takes is decided by the KV Connector (NIXL, Mooncake, LMCache, and friends). Same-node 1P1D can use NVLink, cross-node 1PnD uses RDMA. The contrast with Mooncake is immediate: **in vLLM the phase boundary is drawn by the proxy request-forwarding timeline; in Mooncake it is drawn by instance roles.**

### 3.3 The baseline: non-PD monolithic serving

Without PD disaggregation, the world looks like this — the router sends a request to some Engine instance, and inside that instance continuous batching plus chunked prefill interleaves prefill and decode in the same batch, with the KV cache staying in local GPU memory (offloadable to CPU) the whole time and **no KV ever crossing instances**:

<div class="fig">
<svg viewBox="0 0 680 430" width="100%" font-size="13" font-weight="400" role="img" aria-label="Non-PD monolithic architecture">
<title>Figure 3: The non-PD monolithic architecture</title>
<defs>
<marker id="ar3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<rect x="240" y="40" width="200" height="44" rx="8" fill="#f6f8fa" stroke="#30363d" stroke-width="0.5"/>
<text x="340" y="62" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Client</text>
<path d="M340 84 L340 114" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar3)"/>

<rect x="200" y="118" width="280" height="44" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="140" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">API Server + Router</text>
<path d="M280 162 L140 206" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar3)"/>
<path d="M340 162 L340 206" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar3)"/>
<path d="M400 162 L540 206" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar3)"/>

<rect x="50" y="210" width="180" height="170" rx="16" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="140" y="230" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Engine Instance</text>
<text x="140" y="248" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">Continuous Batching</text>
<rect x="65" y="262" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="140" y="277" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Prefill + Decode</text>
<text x="140" y="295" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">interleaved in one process</text>
<rect x="65" y="314" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="140" y="329" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">KV Cache</text>
<text x="140" y="347" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">local GPU · never crosses instances</text>

<rect x="250" y="210" width="180" height="170" rx="16" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="230" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Engine Instance</text>
<text x="340" y="248" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">Continuous Batching</text>
<rect x="265" y="262" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="277" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Prefill + Decode</text>
<text x="340" y="295" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">interleaved in one process</text>
<rect x="265" y="314" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="329" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">KV Cache</text>
<text x="340" y="347" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">local GPU · never crosses instances</text>

<rect x="450" y="210" width="180" height="170" rx="16" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="540" y="230" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Engine Instance</text>
<text x="540" y="248" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">Continuous Batching</text>
<rect x="465" y="262" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="540" y="277" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Prefill + Decode</text>
<text x="540" y="295" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">interleaved in one process</text>
<rect x="465" y="314" width="150" height="44" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="540" y="329" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">KV Cache</text>
<text x="540" y="347" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">local GPU · never crosses instances</text>

<text x="340" y="405" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">No PD: each instance runs the full prefill-to-decode flow in-process; KV never crosses instances</text>
</svg>
</div>

The monolithic form is the simplest and works out of the box, at the cost of two load types squeezing each other, prefix reuse locked to a single machine, and long contexts capped by single-card memory — exactly the three problems PD disaggregation and distributed caching set out to solve.

## 4. How Mooncake Composes with vLLM: As an External KV Layer

In the previous sections Mooncake and vLLM looked like two separate worlds; in fact they stack. The composition is layered: **the vLLM serving layer (API Server / Router / Engine) stays untouched, while the vLLM KV Connector interface (`kv_transfer_config`) attaches Mooncake Transfer Engine + Store underneath**, replacing or supplementing local KV management:

<div class="fig">
<svg viewBox="0 0 680 485" width="100%" font-size="13" font-weight="400" role="img" aria-label="Mooncake composed with vLLM as layers">
<title>Figure 4: Mooncake composed with vLLM, layered</title>
<defs>
<marker id="ar4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<rect x="80" y="40" width="520" height="130" rx="16" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="62" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">vLLM Serving Layer · scheduling unchanged</text>
<rect x="100" y="80" width="140" height="50" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="170" y="98" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">API Server</text>
<text x="170" y="116" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">OpenAI-compatible</text>
<rect x="260" y="80" width="160" height="50" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="340" y="98" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Router / Proxy</text>
<text x="340" y="116" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">routing &amp; scheduling</text>
<rect x="440" y="80" width="140" height="50" rx="8" fill="#f6f8fa" stroke="#2563eb" stroke-width="0.5"/>
<text x="510" y="98" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Engine Instances</text>
<text x="510" y="116" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">PD or monolithic</text>
<path d="M340 170 L340 196" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar4)" marker-start="url(#ar4)"/>

<rect x="80" y="200" width="520" height="56" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="340" y="220" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">vLLM KV Connector Interface</text>
<text x="340" y="238" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">configured via kv_transfer_config</text>
<path d="M340 256 L340 282" fill="none" stroke="#6b7280" stroke-width="1.5" marker-end="url(#ar4)" marker-start="url(#ar4)"/>

<rect x="80" y="286" width="520" height="150" rx="16" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="340" y="308" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Mooncake · External KV Layer</text>
<rect x="100" y="326" width="200" height="60" rx="8" fill="#f6f8fa" stroke="#10b981" stroke-width="0.5"/>
<text x="200" y="348" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Transfer Engine</text>
<text x="200" y="368" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">RDMA / NVLink transfer</text>
<rect x="320" y="326" width="260" height="60" rx="8" fill="#f6f8fa" stroke="#10b981" stroke-width="0.5"/>
<text x="450" y="348" font-size="13" fill="#1a1a1a" font-weight="500" text-anchor="middle" dominant-baseline="central">Mooncake Store</text>
<text x="450" y="368" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">distributed DRAM / SSD KV pool</text>

<text x="340" y="462" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">What it does: ① moves KV between P and D (can replace NIXL) ② offloads and reuses prefix cache ③ shares KV across instances</text>
</svg>
</div>

In this composition, Mooncake does three things:

1. **KV transfer between P and D**: as one implementation of the KV Connector, it carries KV from P instances to D instances, alongside or instead of NIXL;
2. **Prefix cache offloading**: prefix KV that no longer fits locally is offloaded into the DRAM/SSD tiers of the Store and fetched back on demand, breaking the single-card cap on total cache size;
3. **Cross-instance sharing**: KV leaves the single-machine boundary, and any instance that hits a prefix reuses it — bringing the AttentionStore-style cluster-wide global index from sys10 into the vLLM stack.

The division of labor is clean: **vLLM decides how to schedule requests; Mooncake decides how KV is stored and moved.** This is also the most practical combination in production today — keep the scheduling stack, gain a distributed KV layer for free.

## 5. Three Forms, Side by Side

| Dimension | Mooncake (Fig 1) | vLLM PD disaggregation (Fig 2) | Non-PD monolithic (Fig 3) |
|-----------|------------------|-------------------------------|---------------------------|
| **Where the split lives** | Instance-role layer (P-only / D-only processes) | Separate P/D processes | No split; interleaved in one process |
| **Router / coordinator** | Conductor (KV-aware global scheduling) | Disagg Proxy (two-phase coordination) | Router (instance selection only) |
| **KV movement** | Transfer Engine (RDMA / NVLink) | KV Connector (NIXL / Mooncake) | None; KV never leaves the machine |
| **Cache reuse** | Distributed Store (HBM / DRAM / SSD tiers) | Local prefix cache + connector | Local prefix cache |
| **Deployment complexity** | High (full scheduling and storage stack) | Medium (configure kv_transfer_config) | Low (works out of the box) |
| **Best for** | Large clusters, long context, prefix-heavy production | Flexible deployment, the official PD path | Single machine or small scale, fast start |

One line to tell them apart: **Mooncake redesigns everything with KV as a first-class citizen; the vLLM PD split adds a KV express lane on top of the existing engine; the non-PD monolith is the default "get it running first" shape.**

## 6. Connections to Earlier Posts

- **sys8 (PD disaggregation)** covered why to split and how to size the prefill:decode ratio; this post adds the industrial implementations: Mooncake instance roles and the vLLM proxy + connector design;
- **sys9 (the KV Cache landscape)** ran the numbers on KV size and the three-layer optimization taxonomy; the distributed Store here is the extreme form of the "buy capacity and reuse" branch;
- **sys10 (KV Cache in production)** covered single-machine management (SGLang two-level pools, vLLM paging, AttentionStore cluster indexing); this post covers cross-machine movement (Transfer Engine) and how engines consume it (the Connector interface);
- **fw2 / fw3 (vLLM / SGLang internals)** — PagedAttention and Radix Cache are the foundation of local caching; the Mooncake Store here can be read as pushing that cache "out to the cluster";
- **fa5 (Mamba Hybrid + PD Cache)** warned that hybrid architectures risk silently misplaced cache hits — when KV is shared across instances, keeping prefix hashing consistent with scheduling routing is the same class of problem Mooncake has to answer.
