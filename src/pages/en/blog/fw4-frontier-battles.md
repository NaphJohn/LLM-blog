---
title: '(4) The Shared Frontier: Sync Stalls, Speculative Decoding, PD Disaggregation, Low-Bit Quant'
description: 'In mid-2026 both engines converged on the same battle lines — kill GPU idle time, squeeze decode compute, split prefill from decode, and compress weights to 4 bits.'
pubDate: 2026-08-01
series: vLLM & SGLang Serving Notes
lang: en
altLang: zh
altHref: /blog/fw4-frontier-battles
layout: ../../../layouts/BlogPost.astro
---

## 1. The One-Line Summary of This Era

> **In mid-2026, the throughput frontier is no longer "faster kernels" but "stop letting the GPU wait for the host to feed it data".**

Modern matmul is near saturation; paging, continuous batching, and tree speculative decoding are all mature. Where is the remaining throughput? **In the gaps where the accelerator idles waiting for the host.**

vLLM attacks from the **model-runner side** (MRv2), SGLang from the **speculative-decoding scheduler side** (Spec V2). **Both dismantle the same synchronization bottleneck from opposite ends.**

<div class="fig">
<svg viewBox="0 0 680 240" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <text x="340" y="20" font-size="14" font-weight="700" fill="#374151" text-anchor="middle">One bottleneck, attacked from both ends</text>

  <rect x="30" y="40" width="200" height="90" rx="8" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="130" y="62" font-size="13" font-weight="700" fill="#1d4ed8" text-anchor="middle">vLLM - MRv2</text>
  <text x="130" y="82" font-size="10" fill="#1e3a8a" text-anchor="middle">from the model-runner side</text>
  <text x="130" y="100" font-size="10" fill="#1e3a8a" text-anchor="middle">async-first, zero CPU-GPU sync</text>
  <text x="130" y="118" font-size="10" fill="#1e3a8a" text-anchor="middle">overlap step N and N+1</text>

  <rect x="450" y="40" width="200" height="90" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="550" y="62" font-size="13" font-weight="700" fill="#047857" text-anchor="middle">SGLang - Spec V2</text>
  <text x="550" y="82" font-size="10" fill="#065f46" text-anchor="middle">from the spec-decode scheduler</text>
  <text x="550" y="100" font-size="10" fill="#065f46" text-anchor="middle">draft-extend full-graph capture</text>
  <text x="550" y="118" font-size="10" fill="#065f46" text-anchor="middle">cut D2H / H2D sync</text>

  <rect x="256" y="58" width="168" height="54" rx="8" fill="#fef2f2" stroke="#ef4444" stroke-width="2"/>
  <text x="340" y="80" font-size="13" font-weight="700" fill="#b91c1c" text-anchor="middle">Sync Stall</text>
  <text x="340" y="98" font-size="10" fill="#7f1d1d" text-anchor="middle">time the GPU waits on the host</text>

  <path d="M232 85 L252 85" stroke="#3b82f6" stroke-width="2.5" marker-end="url(#ax)"/>
  <path d="M448 85 L428 85" stroke="#10b981" stroke-width="2.5" marker-end="url(#ax2)"/>
  <defs>
    <marker id="ax" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#3b82f6"/></marker>
    <marker id="ax2" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#10b981"/></marker>
  </defs>

  <text x="340" y="156" font-size="11" fill="#6b7280" text-anchor="middle">saves accelerator idle time, not compute; biggest win for low-concurrency latency-sensitive agents</text>

  <rect x="30" y="172" width="620" height="48" rx="6" fill="#f9fafb" stroke="#e5e7eb"/>
  <text x="44" y="190" font-size="10" fill="#374151">vLLM: MRv2 default (#39337) - PagedAttention removed (#47361) - full-step CUDA Graph (300us to 5us)</text>
  <text x="44" y="208" font-size="10" fill="#374151">SGLang: Spec V2 default (+11% TPS) - #31468 DFlash removes per-step host sync - #31487 fewer prefill graph pads - #31986/#31985 DSpark GEMM stacking</text>
</svg>
</div>

## 2. Speculative Decoding: Squeezing Idle Decode Compute

### 2.1 Why Decode Must Speculate

Decode is **memory-bound**: a forward fetches the whole model from HBM to compute just one token, leaving compute units idle.

Speculative decoding's essence: **since you fetch the weights anyway, verify a few extra candidate tokens for free**. Same fetch cost, several times the output.

### 2.2 Draft Methods Compared

| Method | Draft source | Traits | Adoption |
|---|---|---|---|
| **Standalone small model** | A smaller same-family model | General but needs extra memory and matching vocab | Both; vLLM #38174 supports **heterogeneous vocab** |
| **EAGLE / EAGLE-3** | Lightweight draft head reusing target hidden states | High accept rate, low overhead, mainstream | First-class in both |
| **MTP** | The model's own multi-token head | Learned at training time, ~4 tokens per step | Built into DeepSeek / GLM / Step |
| **DFlash** | Block-diffusion drafting + KV injection | One parallel pass proposes a whole block | SGLang #31468 removes host sync; vLLM #48524 fixes layer sizing |
| **DSpark** | Semi-autoregressive drafting + Markov head + confidence scheduling | Draft length adapts to confidence | vLLM native for DeepSeek V4 Pro; SGLang #31986/#31985 optimize |

<div class="keybox">
Real numbers: <strong>DeepSeek V4 Pro + DSpark hits ~250 tok/s on 8xB300, 12-42% above MTP</strong> (native in vLLM 0.25).<br/>
For the difference between DFlash and DSpark, see the sibling series: <a href="/LLM-blog/en/blog/ep5-dflash-vs-dspark">DFlash vs DSpark head to head</a>.
</div>

### 2.3 Details Worth Noting

- **Thinking-budget-aware speculation** (vLLM #34668): reasoning models behave differently in the thinking phase vs the answer phase; this PR makes speculation budget-aware, safely speeding up reasoning models.
- **Heterogeneous-vocab speculation** (vLLM #38174): draft and target models with different vocabularies can be paired, greatly widening usable draft models.
- **Quant + speculation pitfalls**: vLLM #48816 fixes a GPTQ Qwen3.5 **MTP weight-loading error** when speculation is on. Teams combining quantization and spec decode should watch for these interaction bugs.

## 3. PD Disaggregation: An Emerging Paradigm

### 3.1 Why Split

Prefill and decode have opposite characteristics; on the same GPU they hurt each other:

- a long prompt's prefill **monopolizes a step**, stalling all streaming outputs (TPOT spikes);
- decode wants large batches for throughput, prefill wants to finish quickly for low TTFT;
- their optimal parallelism differs (prefill likes TP + context parallel, decode likes large EP + large batch).

**PD disaggregation** deploys the two phases on different GPU pools; prefill computes and ships the KV cache to decode nodes.

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <rect x="20" y="16" width="110" height="40" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="75" y="34" font-size="11" font-weight="700" fill="#1d4ed8" text-anchor="middle">Router</text>
  <text x="75" y="48" font-size="9" fill="#1e3a8a" text-anchor="middle">Rust / K8s / Planner</text>

  <path d="M132 36 L176 36" stroke="#6b7280" stroke-width="1.5" marker-end="url(#a4)"/>
  <defs><marker id="a4" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6b7280"/></marker></defs>

  <rect x="180" y="10" width="200" height="118" rx="8" fill="#fff7ed" stroke="#fb923c"/>
  <text x="280" y="30" font-size="12" font-weight="700" fill="#9a3412" text-anchor="middle">Prefill pool</text>
  <text x="280" y="48" font-size="10" fill="#9a3412" text-anchor="middle">compute-bound, sets TTFT</text>
  <g font-size="9" text-anchor="middle" fill="#7c2d12">
    <rect x="196" y="58" width="52" height="26" rx="4" fill="#fed7aa" stroke="#f97316"/><text x="222" y="75">GPU</text>
    <rect x="254" y="58" width="52" height="26" rx="4" fill="#fed7aa" stroke="#f97316"/><text x="280" y="75">GPU</text>
    <rect x="312" y="58" width="52" height="26" rx="4" fill="#fed7aa" stroke="#f97316"/><text x="338" y="75">GPU</text>
  </g>
  <text x="280" y="104" font-size="9" fill="#9a3412" text-anchor="middle">TP + context parallel (PCP)</text>
  <text x="280" y="118" font-size="9" fill="#9a3412" text-anchor="middle">higher-compute cards</text>

  <path d="M382 68 L432 68" stroke="#8b5cf6" stroke-width="2" marker-end="url(#a5)"/>
  <defs><marker id="a5" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#8b5cf6"/></marker></defs>
  <text x="407" y="60" font-size="9" fill="#6d28d9" text-anchor="middle">KV transfer</text>
  <text x="407" y="84" font-size="8" fill="#6d28d9" text-anchor="middle">NIXL / Mooncake</text>
  <text x="407" y="96" font-size="8" fill="#6d28d9" text-anchor="middle">NVLink / RDMA</text>

  <rect x="436" y="10" width="216" height="118" rx="8" fill="#f0fdf4" stroke="#4ade80"/>
  <text x="544" y="30" font-size="12" font-weight="700" fill="#166534" text-anchor="middle">Decode pool</text>
  <text x="544" y="48" font-size="10" fill="#166534" text-anchor="middle">memory-bound, sets TPOT</text>
  <g font-size="9" text-anchor="middle" fill="#14532d">
    <rect x="452" y="58" width="44" height="26" rx="4" fill="#bbf7d0" stroke="#22c55e"/><text x="474" y="75">GPU</text>
    <rect x="502" y="58" width="44" height="26" rx="4" fill="#bbf7d0" stroke="#22c55e"/><text x="524" y="75">GPU</text>
    <rect x="552" y="58" width="44" height="26" rx="4" fill="#bbf7d0" stroke="#22c55e"/><text x="574" y="75">GPU</text>
    <rect x="602" y="58" width="44" height="26" rx="4" fill="#bbf7d0" stroke="#22c55e"/><text x="624" y="75">GPU</text>
  </g>
  <text x="544" y="104" font-size="9" fill="#166534" text-anchor="middle">large EP + large batch</text>
  <text x="544" y="118" font-size="9" fill="#166534" text-anchor="middle">bandwidth and capacity bound</text>

  <rect x="20" y="146" width="632" height="62" rx="8" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="336" y="166" font-size="12" font-weight="700" fill="#6d28d9" text-anchor="middle">Tiered KV cache (4 layers)</text>
  <g font-size="9" text-anchor="middle">
    <rect x="40" y="174" width="132" height="24" rx="4" fill="#ede9fe" stroke="#a78bfa"/><text x="106" y="190" fill="#5b21b6">GPU HBM (fastest)</text>
    <rect x="184" y="174" width="132" height="24" rx="4" fill="#ede9fe" stroke="#a78bfa"/><text x="250" y="190" fill="#5b21b6">NVLink pool</text>
    <rect x="328" y="174" width="132" height="24" rx="4" fill="#ede9fe" stroke="#a78bfa"/><text x="394" y="190" fill="#5b21b6">Host DRAM</text>
    <rect x="472" y="174" width="160" height="24" rx="4" fill="#ede9fe" stroke="#a78bfa"/><text x="552" y="190" fill="#5b21b6">SSD / object (Mooncake)</text>
  </g>

  <text x="20" y="230" font-size="11" fill="#6b7280">Fig 2: PD disaggregation. Each phase uses its optimal parallelism and hardware, KV shipped via NIXL / Mooncake.</text>
  <text x="20" y="250" font-size="11" fill="#374151" font-weight="700">Representative numbers:</text>
  <text x="20" y="268" font-size="10" fill="#374151">- NVIDIA Dynamo 1.0 (Planner + 4-tier KV manager): DeepSeek-R1 671B compound 30x, Llama-70B Hopper clean 2x</text>
  <text x="20" y="284" font-size="10" fill="#374151">- DeepSeek V3 full disagg stack ~545 tok/s/GPU (north star) - Together CPD long-context B200 +35-40% - Mooncake covers 75% of real requests</text>
</svg>
</div>

### 3.2 The Players

| Project | Role | Trait |
|---|---|---|
| **NVIDIA Dynamo 1.0** | Orchestration on top of vLLM/SGLang | Planner + 4-tier KV manager; DeepSeek-R1 671B compound 30x |
| **llm-d** (CNCF) | K8s-native PD orchestration | Standard cloud-native path |
| **vLLM V1 Rust router** | Built-in router | **25% higher RPS than llm-d**, no K8s dependency |
| **SGLang PD router** | Built-in, multiple routing policies | Cancels the paired decode on prefill failure |
| **Mooncake** | KV storage and transfer backend | Multi-tier, covers 75% of real requests |
| **NIXL** | Unified transport layer | Both converge on it |

### 3.3 Interesting Variants

- **PPD (ICML 2026)**: a critique of classic PD — **keep KV on the decode node and only append new tokens**. TTFT drops **68%** from turn 2 onward with no TPOT regression. Big for agents.
- **Together CPD**: split prefill again into pre-prefill + prefill, +**35–40%** on long context on B200.
- **vLLM TieringManager (PR #42285)**: two-tier P/D cache across HBM / DRAM / SSD.
- **SGLang PrefillDelayer (#31835)**: delay prefill negotiation until KV-budget admission, avoiding "computed but decode can't hold it".

<div class="warnbox">
<strong>PD disaggregation is not free:</strong> it adds KV network transfer (possibly gigabytes on long context), cross-node scheduling complexity, and a wider failure domain. <strong>It pays off only at sufficient scale, with a high share of long prompts and fast interconnect (NVLink / RDMA).</strong> For small models on a single 8-GPU box, colocated deployment is fine.
</div>

## 4. Low-Bit Quantization

Decode is memory-bound, so **shrinking the weights** is the most direct speedup.

| Format | Bits | Note | Status |
|---|---|---|---|
| **FP8** | 8 | Native since Hopper, tiny precision loss | A production default; vLLM #42569 adds **FP8 KV cache** on SM100 via FA4 |
| **INT4 / AWQ / GPTQ** | 4 | Classic weight quant | Mature; Step-series Int4 weights not yet supported in vLLM |
| **NVFP4** | 4 | Blackwell-native 4-bit float, better than INT4 | 2026 headline; needs FlashInfer (since vLLM 0.26) |
| **MXFP4** | 4 | Micro-scaled 4-bit, AMD-side push (SGLang #28291) | Expanding |
| **NVFP4_AWQ** | 4 | NVFP4 + AWQ calibration (SGLang #31825) | Frontier combo |

**Observation**: SGLang expands more aggressively on FP4 (NVFP4_AWQ, marlin_nvfp4 fixes, unified per-token-group quant kernel #30924); vLLM wins on format coverage.

<div class="warnbox">
Quantization is bug-prone; real cases:<br/>
- vLLM <strong>#48330</strong>: a mixed-dtype fusion kernel reads NVFP4 with the wrong bit pattern, producing silent <code>!!!!!</code> (fixed in 0.25.1)<br/>
- vLLM <strong>#48816</strong>: GPTQ + speculation MTP weight-loading error<br/>
- SGLang <strong>#31762</strong>: wrong routed_scaling_factor in marlin_nvfp4<br/>
<strong>Conclusion: when quant + speculation + new hardware stack up, always run full correctness regression, not just throughput.</strong>
</div>

## 5. A Side Debate: Should You Leave transformers

A persistent disagreement in the community.

- **vLLM's choice**: **dual track**. Native for performance, the Transformers backend for coverage — and since 0.25 the Transformers backend **matches native speed**, so new models are day-zero full-speed. The cost is following transformers v5 (v4 deprecated #40389).
- **Model authors' choice**: more Chinese models ship `trust_remote_code` custom modeling (e.g. Step 3.7 Flash, where transformers 5.0+ is only for debug).
- **Tradeoff**: custom pure-PyTorch modeling gives controllable operators, low overhead, and easy attention/sampling customization; the cost is implementing your own tokenizer, weight loading, and parallelism.

An observation: in the July 22 window, vLLM had zero "leave transformers" PRs — instead it was **converging the two backends** (#49292 M-RoPE for Qwen3-VL on the Transformers backend, #47298 Ovis2_5 transformers-v5 special tokens). **The direction is "native MRv2 + transformers v5 in parallel", not either/or.**

## 6. Summary: How the Four Battle Lines Relate

```
                 Sources of throughput / latency
                          |
   +----------+-----------+-----------+----------+
   |          |           |           |          |
kill sync  speculative  PD disagg   low-bit
           decoding                 quant
   |          |           |           |
GPU never  one forward  phases       fewer weight
waits      many tokens  do not       bytes moved
                        interfere
   |          |           |           |
 MRv2      EAGLE/MTP   Dynamo      FP8/NVFP4
 Spec V2   DFlash      NIXL         MXFP4
           DSpark      Mooncake     AWQ
```

They **stack**: MRv2's zero-sync lets speculation be fully graph-captured; PD disaggregation lets decode pools run bigger batches, which makes speculative verification more worthwhile; quant frees memory that holds more KV blocks.

Next: assemble these into a timeline and see exactly what the two July 2026 release waves changed, and in what order.
