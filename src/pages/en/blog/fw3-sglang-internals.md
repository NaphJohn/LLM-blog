---
title: '(3) SGLang Internals: RadixAttention and the Program View of Inference'
description: 'Starting from the real structure of LLM applications: radix-tree prefix reuse, compressed-FSM constrained decoding, zero-overhead Spec V2, and UnifiedRadixTree.'
pubDate: 2026-08-01
series: vLLM & SGLang Serving Notes
lang: en
altLang: zh
altHref: /blog/fw3-sglang-internals
layout: ../../../layouts/BlogPost.astro
---

## 1. A Different Starting Point

vLLM asks "how do we manage memory without waste?" SGLang asks a different question:

> What do requests in a real LLM application actually look like relative to each other?

The answer: **highly repetitive**.

- **Multi-turn chat**: turn 5's prefix is all of turn 4
- **Agent / ReAct**: the same system prompt plus tool definitions resent every step
- **Few-shot batch jobs**: thousands of requests sharing one multi-thousand-token exemplar block
- **Tree-of-thought / self-consistency**: N branches forked from one intermediate node
- **Batch evaluation**: one question template, only the final options differ

If every request recomputes prefill from scratch, these **identical prefixes get recomputed thousands of times**.

The name states the stance: **S**tructured **G**eneration **Lang**uage — it treats an LLM call as a structured program, not a stream of unrelated requests.

## 2. RadixAttention: Prefix KV Reuse via a Radix Tree

### 2.1 The Mechanism

vLLM also has prefix caching, but early on it was mostly exact-hash matching. SGLang goes further: it organizes the KV of all active requests into a **radix tree**.

- each node stores the KV blocks for a token span;
- a new request does **longest-prefix matching** down the tree, reuses the match, and only prefills the remainder;
- **LRU eviction** manages capacity, so hot prefixes stay resident;
- branches share naturally — tree-of-thought branches are just sibling subtrees.

<div class="fig">
<svg viewBox="0 0 680 290" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <rect x="250" y="14" width="180" height="34" rx="6" fill="#d1fae5" stroke="#10b981"/>
  <text x="340" y="30" font-size="11" font-weight="700" fill="#065f46" text-anchor="middle">system prompt + tools</text>
  <text x="340" y="43" font-size="10" fill="#047857" text-anchor="middle">1800 tok, KV stored once</text>

  <path d="M300 48 L180 78" stroke="#10b981" stroke-width="1.5"/>
  <path d="M340 48 L340 78" stroke="#10b981" stroke-width="1.5"/>
  <path d="M380 48 L500 78" stroke="#10b981" stroke-width="1.5"/>

  <rect x="100" y="80" width="160" height="32" rx="6" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="180" y="100" font-size="10" fill="#1e3a8a" text-anchor="middle">Chat A, history 620</text>
  <rect x="262" y="80" width="156" height="32" rx="6" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="340" y="100" font-size="10" fill="#1e3a8a" text-anchor="middle">Chat B, history 340</text>
  <rect x="424" y="80" width="160" height="32" rx="6" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="504" y="100" font-size="10" fill="#1e3a8a" text-anchor="middle">Eval batch, stem 900</text>

  <path d="M150 112 L110 142" stroke="#3b82f6" stroke-width="1.2"/>
  <path d="M210 112 L250 142" stroke="#3b82f6" stroke-width="1.2"/>
  <path d="M340 112 L340 142" stroke="#3b82f6" stroke-width="1.2"/>
  <path d="M470 112 L430 142" stroke="#3b82f6" stroke-width="1.2"/>
  <path d="M504 112 L504 142" stroke="#3b82f6" stroke-width="1.2"/>
  <path d="M538 112 L578 142" stroke="#3b82f6" stroke-width="1.2"/>

  <g font-size="10" text-anchor="middle">
    <rect x="60" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="110" y="162" fill="#78350f">turn 5 q</text>
    <rect x="200" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="250" y="162" fill="#78350f">retry branch</text>
    <rect x="290" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="340" y="162" fill="#78350f">turn 3 q</text>
    <rect x="380" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="430" y="162" fill="#78350f">option A</text>
    <rect x="454" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="504" y="162" fill="#78350f">option B</text>
    <rect x="528" y="144" width="100" height="28" rx="5" fill="#fef3c7" stroke="#f59e0b"/><text x="578" y="162" fill="#78350f">option C</text>
  </g>

  <rect x="60" y="192" width="270" height="60" rx="6" fill="#fef2f2" stroke="#fca5a5"/>
  <text x="195" y="212" font-size="11" font-weight="700" fill="#b91c1c" text-anchor="middle">No reuse (independent prefill)</text>
  <text x="195" y="230" font-size="10" fill="#7f1d1d" text-anchor="middle">6 requests x (1800 + history + q)</text>
  <text x="195" y="245" font-size="10" fill="#7f1d1d" text-anchor="middle">~16,000 tokens recomputed</text>

  <rect x="350" y="192" width="270" height="60" rx="6" fill="#ecfdf5" stroke="#6ee7b7"/>
  <text x="485" y="212" font-size="11" font-weight="700" fill="#047857" text-anchor="middle">RadixAttention (tree sharing)</text>
  <text x="485" y="230" font-size="10" fill="#065f46" text-anchor="middle">shared spans computed once</text>
  <text x="485" y="245" font-size="10" fill="#065f46" text-anchor="middle">~2,000 tokens, lower TTFT and memory</text>

  <text x="16" y="276" font-size="11" fill="#6b7280">Fig 1: the radix tree folds repeated prefixes into shared paths. Longer shared prefixes and more branches mean bigger gains.</text>
</svg>
</div>

### 2.2 The Boundary of the Gain

<div class="keybox">
RadixAttention's benefit <strong>depends entirely on the prefix-sharing rate of your workload</strong>. Multi-turn agents, batch evaluation, and tree-of-thought can cut over 70% of prefill; but if your requests share no common prefix (e.g. summarizing distinct documents), you only pay a little tree-maintenance overhead. <strong>Look at your traffic shape first.</strong>
</div>

**SGLang v0.5.16** (July 25, 2026) pushed this further: **UnifiedRadixTree became the default prefix cache**, unifying previously separate cache paths (plain prefix cache, tiered cache, PD-scenario cache) into a single tree.

## 3. Structured Output: A Compressed State Machine

The second differentiator is **constrained decoding**.

When you require strict JSON, a regex match, or an EBNF grammar, the naive approach checks the constraint after each token and masks illegal logits to -inf — on the CPU, **yet another per-step sync point**.

SGLang uses a **compressed finite state machine**:

- the grammar is compiled into an FSM ahead of time;
- FSM paths with a **unique legal successor** (e.g. `{"name"` in JSON is necessarily followed by `:`) can **emit multiple tokens at once** without going through the model;
- mask computation is pushed onto the GPU and overlapped with the forward pass.

For agent tool calls that return fixed schemas, this is very effective — many structural characters are free.

<div class="warnbox">
This is also a bug-prone area. <strong>SGLang PR #30747</strong> fixed a <strong>crash when PP (pipeline parallelism) and structured output are enabled together</strong> (Issue #28424). Root cause: scheduling and constraint-validation logic <strong>racing on shared state</strong>; the fix aligns constraint validation with micro-batch boundaries.<br/>
vLLM addressed the sibling issue too: <strong>0.26 makes a grammar failure a per-request error instead of crashing the engine</strong>.
</div>

## 4. Zero-Overhead Spec V2

This is SGLang's most important 2026 performance work, **default since v0.5.15** (July 10), for **about +11% end-to-end TPS**.

### 4.1 The Hidden Cost of Speculative Decoding

The algorithm itself (draft model proposes, target model verifies in one parallel pass) is covered in the [Speculative Decoding Notes](/LLM-blog/en/blog/ep1-speculative-decoding). Here we focus on the **engineering waste**.

A traditional implementation, every step:

```
GPU: draft model proposes 4 candidates
GPU: target model verifies in parallel
GPU -> CPU (D2H): copy back "how many accepted"      <- sync point!
CPU: decide next sequence length and KV layout
CPU -> GPU (H2D): copy new metadata back             <- sync point!
```

Between those copies the **GPU is fully idle**. And because the accept count is a runtime-dependent value, the whole flow **cannot be captured by a CUDA Graph** — every step relaunches many small kernels.

### 4.2 The Fix

<div class="fig">
<svg viewBox="0 0 680 230" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#b91c1c">Traditional: two syncs per step, no full-graph capture</text>
  <rect x="16" y="26" width="80" height="26" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="56" y="43" font-size="10" fill="#065f46" text-anchor="middle">draft x4</text>
  <rect x="100" y="26" width="80" height="26" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="140" y="43" font-size="10" fill="#065f46" text-anchor="middle">verify</text>
  <rect x="184" y="26" width="50" height="26" rx="3" fill="#fecaca" stroke="#ef4444"/><text x="209" y="43" font-size="10" fill="#7f1d1d" text-anchor="middle">D2H</text>
  <rect x="238" y="26" width="86" height="26" rx="3" fill="#fed7aa" stroke="#f97316"/><text x="281" y="43" font-size="10" fill="#7c2d12" text-anchor="middle">CPU len/KV</text>
  <rect x="328" y="26" width="50" height="26" rx="3" fill="#fecaca" stroke="#ef4444"/><text x="353" y="43" font-size="10" fill="#7f1d1d" text-anchor="middle">H2D</text>
  <rect x="382" y="26" width="80" height="26" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="422" y="43" font-size="10" fill="#065f46" text-anchor="middle">draft x4</text>
  <rect x="184" y="56" width="194" height="6" fill="#fca5a5"/>
  <text x="184" y="76" font-size="10" fill="#b91c1c">GPU idle bubble (every step)</text>

  <line x1="16" y1="92" x2="664" y2="92" stroke="#e5e7eb"/>

  <text x="16" y="114" font-size="12" font-weight="700" fill="#047857">Spec V2: draft-extend captured in a CUDA Graph, branching on GPU</text>
  <rect x="16" y="122" width="200" height="26" rx="3" fill="#86efac" stroke="#10b981"/>
  <text x="116" y="139" font-size="10" fill="#065f46" text-anchor="middle">one graph: draft + verify + metadata</text>
  <rect x="220" y="122" width="200" height="26" rx="3" fill="#86efac" stroke="#10b981"/>
  <text x="320" y="139" font-size="10" fill="#065f46" text-anchor="middle">next step graph</text>
  <rect x="424" y="122" width="200" height="26" rx="3" fill="#86efac" stroke="#10b981"/>
  <text x="524" y="139" font-size="10" fill="#065f46" text-anchor="middle">next step graph</text>
  <text x="16" y="166" font-size="10" fill="#047857">CPU assembles the next step ahead, GPU has no bubbles, +about 11% TPS</text>

  <rect x="16" y="180" width="648" height="34" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="28" y="194" font-size="10" fill="#1e3a8a">Follow-up PRs: #31468 DFlash removes per-step host sync (CPU leads by a full step) - #31487 fewer prefill CUDA graph pads</text>
  <text x="28" y="208" font-size="10" fill="#1e3a8a">#31986 stack DSpark dense-draft per-layer ctx KV projections into one GEMM - #31985 fold draft embedding into the draft graph via forward_embed</text>
</svg>
</div>

Three moves:

1. **Make draft-extend CUDA-graph capturable** — fixed upper-bound tensor shapes plus GPU masking replace runtime-variable lengths;
2. **Cut D2H / H2D** — keep the accept count on the GPU, rewrite branching into GPU-executable form;
3. **Fuse metadata computation** — page-table and sequence-length updates enter the graph too.

What is saved is **accelerator idle time, not compute** — so **low-concurrency, latency-sensitive agent workloads benefit most**.

### 4.3 IndexShare MTP

The GLM-5.2 optimizations add another highlight: **IndexShare MTP**.

MTP (Multi-Token Prediction) is speculative decoding with the model's own draft head. On DSA (sparse attention) models, the draft step would normally recompute its own top-k sparse indices. IndexShare lets the draft step **reuse the top-k indices the target model already computed**, cutting draft cost by about **1.9x** on long context.

Add **TopK-V2 (Lightning-TopK)** — a selection algorithm replacing full sorting, tuned for 80k-scale inputs.

Combined result: **GLM-5.2 NVFP4 hits 500+ tok/s/user on Blackwell** (lmsys blog, July 13).

## 5. Other Capabilities Worth Knowing

| Feature | Note | Version |
|---|---|---|
| **Breakable CUDA Graph** | Graphs can be interrupted; DP attention defaults to breakable prefill graph (#31682) | since v0.5.15 |
| **MLA context parallel decoding** | Context parallelism for MLA models (DeepSeek family) | v0.5.15 |
| **FlashInfer all-to-all MoE routing** | MoE routing via FlashInfer all-to-all | v0.5.15 |
| **HPC-Ops attention backend** | More attention operator options (#30540) | July 22 main |
| **Native web search** | Built-in retrieval tool calling | v0.5.15 |
| **Cross-request ViT batching** | Batch vision encoding under multimodal concurrency (#24013) | July 18 main |
| **Large-scale EP** | Expert parallelism for MoE models | ongoing |

<div class="warnbox">
<strong>Security note:</strong> SGLang 0.5.5–0.5.12 had a multimodal path-traversal vulnerability (GHSA-qwrp-wghp-94q2), fixed in 0.5.15+. Upgrade any older deployment.
</div>

## 6. The Two Personalities

Across many releases, the personalities are clear:

| | vLLM | SGLang |
|---|---|---|
| **Origin** | Memory management | Program structure / prefix reuse |
| **Strength** | **Breadth** of models + hardware + quant | **Depth** in prefix sharing, structured output, frontier throughput |
| **Release style** | Steady, dense bug fixes after big changes | Aggressive, frontier work lands fast on main |
| **Sweet spot** | General base, multi-hardware, multi-model | Agent / multi-turn / batch eval / large-scale EP |
| **Quant aggressiveness** | Full-format coverage | Faster NVFP4 / MXFP4 expansion |

A telling observation from the July 22 window: **SGLang's main branch was all speculative-decoding polish** (#31986/#31985), while **vLLM was all stability fixes** (#48524/#49302/#48843/#49306) — the same theme in different phases.

## 7. Summary

- SGLang's core insight is that **an LLM application is a structured program**, from which RadixAttention and compressed-FSM constrained decoding grow;
- **zero-overhead Spec V2** drives speculative-decoding scheduling overhead near zero, the headline of v0.5.15 (+11% TPS);
- **UnifiedRadixTree** (v0.5.16) unifies prefix-cache paths;
- the payoff is workload-dependent: **high prefix-sharing loads favor SGLang**.

Next: the **shared frontier** — why "killing the sync stall" became the mid-2026 storyline, and the emerging PD disaggregation paradigm.
