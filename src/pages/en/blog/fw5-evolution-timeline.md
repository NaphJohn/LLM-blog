---
title: '(5) The July 2026 Release Waves — vLLM 0.25→0.26 and SGLang 0.5.15→0.5.16'
description: 'A day-by-day timeline of the July 2026 architecture turnover in both engines, plus the upgrade traps we hit along the way.'
pubDate: 2026-08-01
series: vLLM & SGLang Serving Notes
lang: en
altLang: zh
altHref: /blog/fw5-evolution-timeline
layout: ../../../layouts/BlogPost.astro
---

## 1. The Whole Timeline

July 2026 was the most change-dense month for these two engines in years: **both nearly synchronized a generation of architecture turnover, then nearly released their next versions on the same day.**

<div class="fig">
<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, Segoe UI, sans-serif">
  <line x1="60" y1="40" x2="60" y2="316" stroke="#d1d5db" stroke-width="2"/>

  <g font-size="11" fill="#6b7280" text-anchor="end">
    <text x="50" y="60">7/10</text>
    <text x="50" y="100">7/11-12</text>
    <text x="50" y="140">7/13-14</text>
    <text x="50" y="180">7/18-19</text>
    <text x="50" y="220">7/21-23</text>
    <text x="50" y="260">7/25</text>
    <text x="50" y="300">7/29-31</text>
  </g>

  <g>
    <circle cx="60" cy="56" r="5" fill="#10b981"/>
    <rect x="76" y="42" width="270" height="26" rx="5" fill="#ecfdf5" stroke="#6ee7b7"/>
    <text x="88" y="59" font-size="11" fill="#047857"><tspan font-weight="700">SGLang v0.5.15</tspan> · Spec V2 default +11% TPS</text>
  </g>

  <g>
    <circle cx="60" cy="96" r="5" fill="#3b82f6"/>
    <rect x="76" y="82" width="330" height="26" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="88" y="99" font-size="11" fill="#1d4ed8"><tspan font-weight="700">vLLM v0.25.0</tspan> · MRv2 default + PagedAttention removed</text>
  </g>

  <g>
    <circle cx="60" cy="136" r="5" fill="#f59e0b"/>
    <rect x="76" y="122" width="470" height="26" rx="5" fill="#fffbeb" stroke="#fcd34d"/>
    <text x="88" y="139" font-size="11" fill="#92400e"><tspan font-weight="700">vLLM v0.25.1</tspan> (NVFP4 garbage fix) · <tspan font-weight="700">SGLang v0.5.15.post1</tspan> · GLM-5.2 500 TPS blog</text>
  </g>

  <g>
    <circle cx="60" cy="176" r="5" fill="#10b981"/>
    <rect x="76" y="162" width="420" height="26" rx="5" fill="#ecfdf5" stroke="#6ee7b7"/>
    <text x="88" y="179" font-size="11" fill="#047857">SGLang main: #31468 DFlash drops host sync · #31487 less prefill graph pad</text>
  </g>

  <g>
    <circle cx="60" cy="216" r="5" fill="#8b5cf6"/>
    <rect x="76" y="202" width="500" height="26" rx="5" fill="#faf5ff" stroke="#c4b5fd"/>
    <text x="88" y="219" font-size="11" fill="#6d28d9">Split work: SGLang polishes DSpark (#31986/#31985) · vLLM dense stability fixes (#48524/#49302…)</text>
  </g>

  <g>
    <circle cx="60" cy="256" r="6" fill="#ef4444"/>
    <rect x="76" y="242" width="470" height="26" rx="5" fill="#fef2f2" stroke="#fca5a5"/>
    <text x="88" y="259" font-size="11" fill="#b91c1c"><tspan font-weight="700">Same-day release: vLLM 0.26.0 + SGLang 0.5.16</tspan> · sync-stall war ends</text>
  </g>

  <g>
    <circle cx="60" cy="296" r="5" fill="#6b7280"/>
    <rect x="76" y="282" width="450" height="26" rx="5" fill="#f9fafb" stroke="#d1d5db"/>
    <text x="88" y="299" font-size="11" fill="#374151">Axis shifts: production stability · GLM-5.2 NVFP4+MTP+P/D landing</text>
  </g>

  <text x="60" y="332" font-size="11" fill="#6b7280">Fig 1: The two product lines moved in almost perfect lockstep through July — evidence they face the same bottlenecks.</text>
</svg>
</div>

## 2. vLLM Evolution Detail

### 2.1 v0.25.0 (7/11–7/12): the architecture turnover

This was **the biggest leap**, four headline changes:

| Change | PR | Meaning |
|---|---|---|
| **Model Runner V2 becomes default for all dense models** | #39337 | async-first, zero CPU-GPU sync, overlap step N and N+1 |
| **PagedAttention removed** | #47361 | abstraction pushed down to the attention-backend kernel; the paging mechanism itself is kept |
| **Transformers v4 deprecated** | #40389 | must migrate to v5 |
| **Transformers backend parity** | — | any new architecture HF implements is served day-0 at full speed |

Also in tow: unified streaming-parse engine (#46610), native DSpark speculative decoding for DeepSeek V4 Pro (8×B300 ≈ 250 tok/s, 12–42% above MTP), heterogeneous-vocab general speculation (#38174), thinking-budget-aware speculation (#34668), and the compile requirement raised to C++20.

<div class="keybox">
<strong>Performance character:</strong> MRv2's gains show most in <strong>small/medium batch</strong> — exactly the regime real Agent traffic lives in. If you only run large-batch offline jobs, you'll feel it far less.
</div>

### 2.2 v0.25.1 (7/14): a two-commit must-have patch

Two days after release came a patch, because a **silent correctness bug** was found:

- **#48330 mixed-dtype quant fusion guard**: FlashInfer's `allreduce + RMSNorm + static-quant` fused kernel, when the activation is BF16 and the RMSNorm weight is FP32, hits a dtype mismatch that reads 4-bit NVFP4 with the wrong bit pattern → corrupts hidden states → output degrades into repeated `!!!!!`. The fix adds a dtype-match guard: mismatched takes the safe path, matched keeps the fusion.
- **#47888**: TorchCodec no longer blocks startup when FFmpeg is missing.

<div class="warnbox">
For anyone serving <strong>NVFP4</strong> (Gemma4 / Qwen-family / GLM-5.2, etc.), <strong>0.25.0 must upgrade to 0.25.1</strong>. This bug doesn't error or crash — it just quietly outputs garbage, so it's easy to miss in production.
</div>

### 2.3 7/18–7/25 main: stability wrap-up

0.25 was a big change, and a wave of edge-config fixes followed:

| PR | Fixes |
|---|---|
| #48524 | DFlash layer sizing wrong when `num_target_layers ≠ num_hidden_layers` |
| #49302 | DSA crash under interruptible segmented CUDA Graph |
| #48843 | `graph_pool_id` not set before full CUDA Graph capture |
| #49306 | FA4 JIT warmup MLA fallback |
| #48860 | KV Connector delayed request double-counts prefix-cache metric |
| #49292 | Qwen3-VL M-RoPE on the Transformers backend |
| #49190 | Cosmos3 Edge video model |
| #48816 | GPTQ Qwen3.5 MTP weight-loading error when speculation on |
| #42569 | FA4 on SM100 (Blackwell) adds **FP8 KV cache** support |
| #48683 | ROCm up to AITER v0.1.16.post5 |
| #45991 | XPU adds DeepSeek-V4 `fuse_index_q` SYCL path |
| #49244 | remove old partial-prefill params |
| #48914 / #49427 | FlashInfer up 0.6.15; restore `dequant_cache` OOB guard |

<div class="warnbox">
vLLM main merged roughly <strong>64 commits/day</strong> in June. That speed means: <strong>production must lock release tags, don't follow main</strong>; run a full regression before upgrading.
</div>

### 2.4 v0.26.0 (7/25)

- **grammar failure no longer crashes the engine**: a structured-output parse failure becomes a single-request error, not affecting other requests;
- **prefix-cache hit-rate reporting**: observability filled in;
- **NVFP4 requires FlashInfer**: dependency made explicit;
- **some startup flags renamed**: check your launch script on upgrade.

### 2.5 7/29–7/31: shifting to production deployment

As the version cadence slowed, the focus turned to landing:

- **GLM-5.2 production deployment plan**: NVFP4 + MTP + P/D combo. `IndexerCache` lifts MTP acceptance; **PCP (context parallel) lifts prefill throughput from 20.1k to 27.3k**;
- **P/D two-tier cache TieringManager** (PR #42285);
- **async KV-load prefetch fix** (PR #46694);
- week-27 weekly-report theme: PD disaggregation via ZMQ + NIXL, Rust protocol refactor.

## 3. SGLang Evolution Detail

### 3.1 v0.5.15 (7/10): zero-overhead Spec V2

| Feature | Note |
|---|---|
| **Zero-overhead Spec V2 default** | draft-extend made CUDA-graph-capturable, cut D2H/H2D, fused metadata → **end-to-end +~11% TPS** |
| **IndexShare MTP** | draft step reuses target model's DSA top-k index, long-context draft cost down **~1.9×** |
| **Breakable CUDA Graph default** | graph interruptible, balancing graph gains with scheduling flexibility |
| **MLA context parallel decoding** | context-parallel decoding for DeepSeek-family MLA |
| **FlashInfer all-to-all MoE routing** | MoE routing via FlashInfer |
| **Native web search** | built-in retrieval tool |

It also fixed the **0.5.5–0.5.12 multimodal path-traversal vulnerability** (GHSA-qwrp-wghp-94q2).

### 3.2 v0.5.15.post1 (7/14): GLM-5.2 tuning

Coinciding with the lmsys official blog "GLM-5.2 NVFP4 500 TPS":

- Spec V2 default **+11% TPS**
- **IndexShare MTP** reuses top-k → ~**1.9×** cost cut
- **TopK-V2 (Lightning-TopK)**: "select" replaces "full sort", optimizing 80k-level ultra-long inputs
- result: **GLM-5.2 NVFP4 on Blackwell hits 500+ tok/s/user**

Teams running GLM-5.2 should lock post1.

### 3.3 7/18–7/23 main: speculative decoding keeps getting polished

| PR | Content |
|---|---|
| #31468 | **DFlash removes per-step host sync in spec decode** — CPU leads GPU by a full step (pipeline); branch logic made CUDA-graph-capturable |
| #31487 | reduce prefill CUDA graph padding |
| #31986 | **DSpark dense-draft per-layer ctx KV projection stacked into a single big GEMM** |
| #31985 | fold draft embedding into the draft graph via `forward_embed` |
| #31682 | DP attention default breakable prefill cuda graph |
| #31981 | DSA draft-extend metadata kernel skips page-table columns beyond KV length |
| #30272 | SM120 DeepSeek V4 flashinfer_mxfp4 MoE + TP2 (consumer Blackwell) |
| #24013 | VLM cross-request ViT encoding batching (multimodal concurrency) |
| #31825 | ModelOpt FP4 path supports **NVFP4_AWQ** |
| #31762 | fix marlin_nvfp4 `routed_scaling_factor` |
| #30924 | unified per-token-group quant kernel |
| #30540 | new **HPC-Ops attention backend** |
| #31109 | remove QServe and FBGEMM FP8 quant (old paths migrate to ModelOpt / compressed-tensors) |
| #32047 | fix stale per-token-group quant callers; `evict_from_tree_cache` only evicts the KV gap |
| #27894 / #31835 | NIXL disagg functional tests into CI; PrefillDelayer delays until KV-budget admission |

<div class="keybox">
Read #31468, #31986, #31985 together and a very clear main line emerges: <strong>eliminate one by one every "ask the CPU every step" in speculative decoding, merge small kernels into big kernels, finally let the whole step into CUDA Graph.</strong> What's saved is accelerator idle time, not compute.
</div>

### 3.4 v0.5.16 (7/25): UnifiedRadixTree

**UnifiedRadixTree becomes the default prefix cache** — unifying previously scattered cache paths (ordinary prefix cache, hierarchical cache, PD-scenario cache) into one tree for hit and eviction logic.

### 3.5 7/29–7/31

- **PR #30747** fixes a **PP (pipeline parallel) + structured-output crash** (Issue #28424). Root cause: the scheduler and constraint logic **raced for state concurrently**; fix: align the constraint check with the micro-batch boundary.
- **PR #30440** grpc disaggregated generation.

## 4. Upgrade Checklist

From a month of traps, a practical checklist:

**vLLM 0.24.x → 0.25.x/0.26.x**

- [ ] Go straight to **0.25.1 or higher**, don't stop at 0.25.0 (NVFP4 silent garbage)
- [ ] Upgrade transformers to **v5**
- [ ] Compile environment supports **C++20**
- [ ] Check whether you use custom ops (may fall back to MRv1 and lose the gains)
- [ ] Check whether partial-prefill related old params were removed
- [ ] From 0.26, check whether startup flags were renamed; confirm FlashInfer is installed for NVFP4
- [ ] **Run an output-correctness regression**, don't just watch throughput

**SGLang 0.5.x**

- [ ] Versions below **0.5.15** must upgrade (multimodal path-traversal vulnerability)
- [ ] Running GLM-5.2 → lock **v0.5.15.post1**
- [ ] Used QServe / FBGEMM FP8 → migrate to ModelOpt / compressed-tensors
- [ ] Want DSpark / DFlash latest → need main, **but wait for a stable tag in production**
- [ ] Running PP + structured output together → confirm it includes #30747

## 5. What the Timeline Tells Us

1. **The two move in tight lockstep**: 0.25.0 (7/11) vs 0.5.15 (7/10), 0.26.0 vs 0.5.16 (both 7/25). This means they face **the same bottlenecks**, not separate wheel-reinvention.
2. **A big release is always followed by a wave of fixes**: a patch within a week of 0.25.0, then two weeks of dense edge-config fixes. **Waiting 1–2 weeks after a new version before production is the rational choice.**
3. **The gap between main and release is large**: SGLang's frontier optimizations (DSpark GEMM stacking, etc.) are all on main; stable tags lag. Want performance → follow main and benchmark yourself; want stability → lock the tag.
4. **The main axis is migrating**: early July was "kill sync stalls"; after the same-day late-July release, the axis shifted to **speculative decoding + prefix cache + day-one model support + production deployment**.

The next post is the most practical one: **which models are supported, and who to pick for what scenario.**
