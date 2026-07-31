---
title: '(6) Supported Models & Selection Guide — Who to Pick for What Scenario'
description: 'Model matrix, hardware matrix, quant matrix and a selection decision tree, plus real deployment notes for Step 3.7 Flash, GLM-5.2, DeepSeek-V4 and other mainstream models.'
pubDate: 2026-08-01
series: vLLM & SGLang Serving Notes
lang: en
altLang: zh
altHref: /blog/fw6-model-support-selection
layout: ../../../layouts/BlogPost.astro
---

## 1. Model Support Matrix

Conclusion first: **both cover mainstream models well; the difference is "how fast a new model is usable" and "how deep the tuning goes".**

| Model | Scale / structure | vLLM | SGLang | Note |
|---|---|---|---|---|
| **DeepSeek-V4 / V4 Pro** | MoE + MLA + DSA | First-class, **native DSpark spec** | First-class, official cookbook benchmark | 8×B300 ≈ 250 tok/s, DSpark 12–42% above MTP |
| **GLM-5.2** | MoE + DSA | Production plan: NVFP4+MTP+P/D | **Deep-tuning benchmark** | Blackwell 500+ tok/s/user; PCP prefill 20.1k→27.3k |
| **Qwen3 / Qwen3.5 / Qwen3-VL** | dense + MoE + VLM | First-class (incl Transformers backend M-RoPE) | First-class | Watch GPTQ + spec combo trap (#48816) |
| **Step 3.7 Flash** | 196B MoE / 11B active / 256K | Prebuilt image `vllm/vllm-openai:stepfun37` | Dev image `lmsysorg/sglang:dev-step-3.7-flash` | FP8 / BF16 / NVFP4 + MTP / EAGLE |
| **Step-3.5-Flash** | 196B MoE + 3:1 sliding window | Official recipe deploy guide | Supported | Int4 weights not yet in vLLM |
| **Step3-VL-10B** | 10B edge VLM | nightly ≥ 0.14.0rc2 | latest main + cookbook | Single RTX 4090, AIME2025 94.43% |
| **Tencent Hy3** | 295B MoE | Day-one | Day-one | Domestic model day-one case |
| **Llama / Mistral / Gemma** | dense | Full | Full | Gemma4: NVFP4 needs 0.25.1+ |
| **Cosmos3 Edge video** | multimodal | Supported (#49190 fix) | — | vLLM wider multimodal |
| **New arch just out on HF** | any | **Day-0 full speed** (Transformers parity) | Wait for native | vLLM clear edge |

<div class="keybox">
<strong>The single most important difference:</strong> vLLM's <code>Transformers backend parity</code> means <strong>as long as HF has an implementation, a new model is served at full speed the same day</strong> — no waiting for the framework to write a native kernel. For teams chasing new models, this one point often decides the selection.
</div>

## 2. Hardware Support Matrix

| Hardware | vLLM | SGLang | Note |
|---|---|---|---|
| **NVIDIA Hopper (H100/H200)** | ✅ mature | ✅ mature | FP8 native |
| **NVIDIA Blackwell (B200/B300/GB200)** | ✅ FA4 + SM100 FP8 KV | ✅ NVFP4 deep tuning | SGLang more aggressive on NVFP4 |
| **Consumer Blackwell (SM120)** | ✅ | ✅ #30272 DeepSeek-V4 mxfp4 MoE + TP2 | Single / dual-card path |
| **AMD ROCm** | ✅ AITER v0.1.16.post5 | ✅ MXFP4 (#28291) | vLLM wider |
| **Intel XPU** | ✅ DeepSeek-V4 `fuse_index_q` SYCL | Limited | **vLLM exclusive edge** |
| **TPU / CPU** | ✅ | Limited | vLLM leads breadth |

**Conclusion**: on hardware breadth **vLLM leads clearly**; but on the most frontier combo, Blackwell + NVFP4, **SGLang tunes deeper**.

## 3. Selection Decision Tree

<div class="fig">
<svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <rect x="240" y="10" width="200" height="34" rx="17" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="340" y="32" font-size="12" font-weight="700" fill="#374151" text-anchor="middle">vLLM or SGLang?</text>

  <path d="M340 44 L340 62" stroke="#9ca3af" stroke-width="1.5"/>
  <rect x="216" y="62" width="248" height="34" rx="6" fill="#fffbeb" stroke="#fcd34d"/>
  <text x="340" y="84" font-size="11" fill="#92400e" text-anchor="middle">Do requests share a long prefix? (system / multi-turn / eval)</text>

  <path d="M240 96 L120 122" stroke="#10b981" stroke-width="1.5"/>
  <text x="150" y="112" font-size="10" fill="#047857">Yes, high share</text>
  <path d="M440 96 L560 122" stroke="#6b7280" stroke-width="1.5"/>
  <text x="500" y="112" font-size="10" fill="#6b7280">No / unsure</text>

  <rect x="20" y="124" width="200" height="34" rx="6" fill="#ecfdf5" stroke="#6ee7b7"/>
  <text x="120" y="146" font-size="11" font-weight="700" fill="#047857" text-anchor="middle">Prefer SGLang</text>

  <rect x="460" y="124" width="200" height="34" rx="6" fill="#fffbeb" stroke="#fcd34d"/>
  <text x="560" y="146" font-size="11" fill="#92400e" text-anchor="middle">Need to run a just-released model?</text>

  <path d="M120 158 L120 178" stroke="#10b981" stroke-width="1.5"/>
  <rect x="20" y="180" width="200" height="76" rx="6" fill="#f0fdf4" stroke="#86efac"/>
  <text x="34" y="198" font-size="10" fill="#065f46">· Agent / multi-turn chat</text>
  <text x="34" y="214" font-size="10" fill="#065f46">· Batch eval / tree-of-thought</text>
  <text x="34" y="230" font-size="10" fill="#065f46">· Strict structured output</text>
  <text x="34" y="246" font-size="10" fill="#065f46">· Large EP · NVFP4 extreme tuning</text>

  <path d="M510 158 L440 182" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="452" y="176" font-size="10" fill="#1d4ed8">Yes</text>
  <path d="M610 158 L610 182" stroke="#6b7280" stroke-width="1.5"/>
  <text x="618" y="176" font-size="10" fill="#6b7280">No</text>

  <rect x="330" y="184" width="180" height="34" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="420" y="206" font-size="11" font-weight="700" fill="#1d4ed8" text-anchor="middle">Choose vLLM (Day-0 full speed)</text>

  <rect x="520" y="184" width="140" height="34" rx="6" fill="#fffbeb" stroke="#fcd34d"/>
  <text x="590" y="206" font-size="10" fill="#92400e" text-anchor="middle">Non-NVIDIA hardware?</text>

  <path d="M560 218 L500 242" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="506" y="236" font-size="10" fill="#1d4ed8">Yes</text>
  <path d="M630 218 L630 242" stroke="#6b7280" stroke-width="1.5"/>
  <text x="638" y="236" font-size="10" fill="#6b7280">No</text>

  <rect x="380" y="244" width="180" height="30" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="470" y="264" font-size="11" font-weight="700" fill="#1d4ed8" text-anchor="middle">Choose vLLM (ROCm/XPU/TPU)</text>

  <rect x="570" y="244" width="90" height="30" rx="6" fill="#f9fafb" stroke="#d1d5db"/>
  <text x="615" y="264" font-size="10" fill="#374151" text-anchor="middle">Either works</text>

  <rect x="20" y="290" width="640" height="96" rx="8" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="340" y="310" font-size="12" font-weight="700" fill="#6d28d9" text-anchor="middle">Practical advice: it is not a single-choice question</text>
  <text x="36" y="332" font-size="11" fill="#5b21b6">· Both expose OpenAI-compatible APIs → low switching cost; worth <tspan font-weight="700">benchmarking each with your real traffic</tspan> before deciding</text>
  <text x="36" y="352" font-size="11" fill="#5b21b6">· Large-scale setups can <tspan font-weight="700">co-deploy</tspan>: Agent main path on SGLang (prefix reuse), long-tail / new models on vLLM (breadth)</text>
  <text x="36" y="372" font-size="11" fill="#5b21b6">· Benchmarking must look at <tspan font-weight="700">TTFT / P99 TPOT / throughput</tspan>; total throughput alone leads to wrong picks</text>
</svg>
</div>

## 4. One-Line Selection Mantra

<div class="keybox">
<strong>Heavy shared prefix → SGLang; need model / hardware breadth → vLLM.</strong>
</div>

Spelled out a bit:

**Signals leaning SGLang**

- Main scenario is Agent, multi-turn dialogue, ReAct loops (system prompt + tool defs resent every step)
- Batch eval, tree-of-thought, self-consistency sampling (one prefix, many forks)
- Strict JSON Schema / regex constrained output, and high share
- Running DeepSeek / GLM-style MoE + MLA/DSA models, wanting NVFP4 extreme throughput
- Needing large-scale expert-parallel (EP) deployment

**Signals leaning vLLM**

- Diverse model types, frequently needing just-released new architectures
- Hardware not pure NVIDIA (ROCm / Intel XPU / TPU)
- Diverse quant-format needs (AWQ / GPTQ / FP8 / NVFP4 / MXFP4 mixed)
- Want to plug into existing ecosystem: KV Connector, Rust router, Dynamo, llm-d
- Team values "stable version, complete docs, pitfalls already stepped on"

## 5. Deployment Notes for Typical Models

### 5.1 Step 3.7 Flash (196B MoE / 11B active / 256K)

- **vLLM**: the official prebuilt image `vllm/vllm-openai:stepfun37` is the most stable, supports MTP spec + NVFP4 4-card deployment
- **SGLang**: `lmsysorg/sglang:dev-step-3.7-flash` + EAGLE
- Both frameworks' adaptation is "first-class mature", but **deep tuning and public-benchmark maturity still lag GLM / DeepSeek**
- Model side requires `transformers ≥ 5.0` (custom modeling, via `trust_remote_code`)

### 5.2 Step-3.5-Flash: why it's both fast and lean

This model is a good example of "three tricks stacked":

1. **3:1 hybrid sliding-window attention**: most layers use the sliding window, dropping the main cost from O(n²) to O(n·w); a few global layers handle long-range information flow;
2. **MTP built-in draft head**: ~4 tokens per step, i.e. built-in speculative decoding;
3. **Sparse MoE**: 196B total params, only 11B active per token — **memory pays total params, compute pays active**.

<div class="warnbox">
Deployment note: on vLLM, <strong>MoE / MTP related optimizations are NOT all on by default</strong> — they need explicit config per the official recipe; <strong>Int4 weights are not yet supported in vLLM</strong>.
</div>

### 5.3 GLM-5.2: currently the deepest-tuned combo

The production plan stacks three pieces: **NVFP4 quant + MTP spec + P/D disagg**.

- `IndexerCache` lifts MTP acceptance
- **PCP (context parallel) lifts prefill throughput from 20.1k to 27.3k**
- SGLang side: IndexShare MTP (draft reuses top-k, long-context cost cut 1.9×) + TopK-V2 (Lightning-TopK, optimizes 80k-level inputs)
- result: **Blackwell 500+ tok/s/user**
- recommend locking SGLang **v0.5.15.post1** or higher

### 5.4 Step3-VL-10B: the edge-multimodal landing point

- 10B VLM (PE-lang 1.8B + Qwen3-8B), **runnable on a single RTX 4090** (BF16 / FP8)
- AIME2025 94.43%
- needs vLLM nightly ≥ 0.14.0rc2 or SGLang latest main (official cookbook)
- depends on nightly / main, **production must lock the version**
- currently the most worth-following landing point for domestic edge multimodal inference

## 6. Deployment Practice Checklist

**Before launch**

- [ ] Benchmark with **real traffic distribution**, not fixed-length synthetic requests
- [ ] Measure all three metrics: TTFT, P99 TPOT, total throughput
- [ ] Quant models run an **output-correctness regression** (NVFP4 garbage is silent)
- [ ] Confirm the prefix-sharing rate — decides whether RadixAttention pays off for you
- [ ] Lock the release tag, don't follow main

**Suggested tuning order**

1. Set `--max-model-len` to the real need (directly decides the KV budget)
2. Enable prefix cache / RadixAttention (almost no side effects)
3. Tune `--max-num-seqs` and `gpu-memory-utilization` for the throughput knee
4. Then enable speculative decoding (MTP / EAGLE), watch acceptance > 60%
5. Only then consider quant and PD disaggregation (big gain but big complexity)

**When you don't need PD disaggregation**

- Single 8-GPU box or smaller, small/medium model → colocated deployment is simpler
- Prompts generally short → prefill doesn't interfere anyway
- No NVLink / RDMA fast interconnect → KV transfer eats the gains

## 7. Series Summary

Six posts in, the main line looks like this:

| # | Topic | One line |
|---|---|---|
| 1 | Why an engine is needed | Naive inference wastes on "memory holes, slot idle, GPU waiting" |
| 2 | vLLM internals | Started with paged KV + continuous batching; kills sync via MRv2, stands on breadth |
| 3 | SGLang internals | Starts from "an LLM app is a structured program"; stands on prefix reuse + frontier throughput |
| 4 | The shared frontier | Four battle lines — sync stall, spec decode, PD disagg, low-bit quant — stack on each other |
| 5 | Version evolution | July 2026, both nearly synchronized an architecture turnover; a big release is always followed by fixes |
| 6 | Models & selection | Heavy prefix → SGLang, breadth → vLLM; best to benchmark each with real traffic |

<div class="keybox">
Last observation: the two engines' competition is <strong>purely good for users</strong>. One ships MRv2, the other soon has Spec V2; one goes NVFP4, the other immediately follows with NVFP4_AWQ. Both expose OpenAI-compatible APIs, so <strong>switching cost is low</strong> — don't treat selection as a one-time lifelong decision; periodically re-benchmark with your own real traffic.
</div>

> Related reading: the algorithm behind speculative decoding is in the [Speculative Decoding Notes](/LLM-blog/en/blog/ep1-speculative-decoding); model-side MoE, attention, and multimodal structures are in the [Multimodal Decoding Notes](/LLM-blog/en/blog/mm5-efficiency-frontier).
