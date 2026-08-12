---
title: vLLM & SGLang Community Tracker · 2026-08-07 — KV Reuse Enters "Combinatorial Correctness" (HiCache / Mooncake Principles & Architecture)
description: LLM Infra Daily #7. Window 08-05~08-07 — vLLM / SGLang 100+ commits each; the theme is silent correctness bugs when prefix-cache × tiered-storage × spec-decode × PD-disagg combine. Deep dives into HiCache tiered KV cache and Mooncake distributed KV store — principles, evolution, architecture.
pubDate: 2026-08-12
series: Community Tracker Notes
lang: en
altLang: zh
altHref: /blog/tr2-vllm-sglang-20260807
layout: ../../../layouts/BlogPost.astro
---

## ★ Most Worth Watching Today

**SGLang #30393 "HiCache supports packed / sidecar draft cache" — not a perf optimization, but the disclosure of a silent correctness defect.** Any production environment running tiered KV cache (HiCache L2/L3) together with speculative decoding (MTP / EAGLE / DSpark) may, with perfectly normal cache-hit rates, have been silently losing accept length because the draft-side state was never restored — no alert, no error log, just flat throughput. Do today: compare `accept_length` before/after upgrading to main.

Runners-up: **SGLang #28836 (torch 2.13 major bump, schedule ahead)** and **vLLM #49206 (PRIORITY scheduler silently dropping requests — a correctness fix)**.

## 1. Baseline (no new tag from any of the three)

| Object | Latest stable | Released | Previous |
|--------|---------------|----------|----------|
| vLLM | v0.26.0 | 2026-07-27 | v0.25.1 (07-14) |
| SGLang | v0.5.16 | 2026-07-25 | v0.5.15.post1 (07-14) |
| StepFun | Step-3.7-Flash / 3.5-Flash | push frozen 06-01 / 04-03 | — |

All dynamics this period are from main: vLLM 100+ commits, SGLang 100+ commits (48h hit the single-page cap).

## 2. The Theme: KV Reuse Enters "Combinatorial Correctness"

For a year, the prefix-cache / KV-offload race was about **hit rate** — how much recompute saved. This window both frameworks exposed and fixed the same class of problem: **cache reuse silently breaks when combined orthogonally with other features**.

- SGLang #30393: HiCache restores only the target-model KV, not the draft-side state → prefix "reports hit" but accept rate drops.
- SGLang #30545: PD-disagg staging buffer + radix cache → a hit pushes the first shard past a grid slot, shifting every subsequent chunk index.
- vLLM #50507: fine-grained intra-block tail prefix reuse (Mamba hybrid); #48069 / #44956: tenant + group semantics for Mooncake.

> **One line**: cache is no longer "a buffer in VRAM" but a **storage system** that must coexist correctly with speculative decoding, PD disaggregation, and multi-tenancy.

## 3. 🔬 Deep Dive: What HiCache and Mooncake Are, and How They Evolved

### 3.1 HiCache — SGLang's Tiered KV Cache

**Positioning**: extends KV cache from "a buffer in VRAM" into **cross-tier storage** — L1 = GPU VRAM, L2 = host memory, L3 = remote storage (object store / distributed KV pool). Lower tiers save more recompute; cross-node prefix reuse means multi-replica deployments need not each recompute long system prompts.

**Core structure**: prefix matching on a **radix tree** by token sequence — the判定 is "is the target model's KV available". Cache moves between L2/L3 as a unit of "host pool".

**Key evolution this period (#30393)**: speculative decoding introduces a **second state machine**; HiCache used to move only the target pool, leaving draft state missing. Two integration paths:

| Path | Topology | Host-pool layout |
|------|----------|------------------|
| **Packed** | Standard NextN MTP/EAGLE: DeepSeek-V3.2/V4, GLM-5.x, MiMo-V2.5; incl. DeepSeek-V4 DSpark | Draft KV / indexer / SWA buffers appended as a tail layer of the target host pool, same slots, one HiCache operation |
| **Sidecar** | Standalone EAGLE/EAGLE3, DFlash, non-V4 DSpark | Build standalone DRAFT / DRAFT_INDEXER / DRAFT_SWA pools only for non-empty draft state; index derived from target KV/SWA; attached to the same L2/L3 op |

> Constraint: the draft cache's index space must be **derivable** from the target's, or one move cannot align both. Packed = "draft layers are just extra tail layers of the target"; Sidecar = structurally independent draft models. Sidecar built only for non-empty draft state — zero overhead when spec-decoding is off.

### 3.2 Mooncake — Distributed KV Store (vLLM's main push)

**Positioning**: Mooncake is the distributed KV store standard in vLLM's **KV connector** ecosystem — it hosts KV cache on a separate cluster so multiple vLLM deployments share one prefix cache, enabling PD disaggregation and cross-deployment reuse.

**Key evolution this period (#48069 / #44956)**: multi-tenancy and group semantics.

- **#48069 tenant namespace**: read `tenant_id` from Mooncake JSON config and pass to `MooncakeDistributedStore.setup()`; different vLLM deployments get isolated namespaces (default normalizes to `default` and passes nothing — backward compatible; old Mooncake with a non-default tenant raises `RuntimeError`).
- **#44956 group semantics (enable_group_semantics)**: multiple physical objects belonging to one logical KV entry carry a shared `group id`, enabling **group-aware metadata routing, lease refresh, and eviction** in Mooncake — sharply reducing "one cache entry fragmented and evicted" waste.

**Deployment impact**: running multiple vLLM deployments on one Mooncake cluster (multi-line / multi-env) finally has official isolation — previously only key-prefix hacks. #51067 further switches Docker to Mooncake's **official wheel** instead of self-building. Mooncake has gone from "one connector impl" to a standard part both frameworks depend on — a sign of ecosystem convergence.

### 3.3 Why "Tiered Cache × Speculative Decoding" Silently Loses Performance

A "prefix hit" is **boolean** in metrics. But in a spec-decoding system, a hit has two qualities:

- **Full hit**: target KV + draft state both present.
- **Partial hit**: only target KV.

A partial hit errors nowhere, drops no hit-rate metric — it just silently shortens **accept length**, ending as "high cache-hit rate, flat throughput".

**What speculative decoding actually caches** (the second state machine):

- Draft-model KV — EAGLE/EAGLE3/DFlash are standalone small models with their own layers and KV.
- DSA indexer state — DeepSeek's sparse-attention index structure deciding Top-K tokens.
- SWA state — sliding-window attention's in-window cache.

These live in device pools **separate from** the target KV. When HiCache tiers L2/L3 and moves only the target pool, the draft pool is either unbacked or restored with mismatched slot mapping. Prefix matching judges "computed" by target KV → skip prefill → draft model proposes with empty/mismatched draft KV → mass rejection → `accept_length` 4~5 → 1~2, `cache_hit_rate` pretty but TPS flat. **No log tells you what broke.**

**Two more versions of the same bug, same day**:

- #30545 (PD disagg × radix cache): staging locates shards by `chunk_idx = start_page // full_chunk_pages`, assuming an even grid; a radix hit pushes the first send past one slot, shifting every later chunk index. Fix: prefill floors to grid; decode fully arrival-driven scatter.
- #50507 (Mamba hybrid): physical blocks huge, offload only whole-block, tail-computed tokens wasted (112 of 900 lost). The assumption "block is the minimal reuse unit" broken by "blocks got huge".

> **Transferable judgment**: the feature matrix is too large to exhaustively combo-test (prefix cache × tiered store × spec decode × PD disagg × hybrid arch × heterogeneous TP). These bugs share the shape "module A's implicit assumption broken by module B, with no assertion to catch it". For users: **don't assume two features both marked stable are safe together**; for every new feature, diff `accept_length`, `cached_tokens`, `TTFT` — not just whether it errors.

## 4. 🟢 vLLM Highlights (main 08-05 ~ 08-07)

- **KV/PD #48069 / #44956**: Mooncake multi-tenant + group semantics (see 3.2).
- **Perf #50507 / #50992**: intra-block tail prefix reuse for KV offload; Attention–Mamba hybrids (Qwen3.6) restore tail fragments by `prefix_match_unit`, pick correct source block per Attention KV vs Mamba recurrent state, copy-on-write continue. Measured Qwen3.6-27B / 900-token prompt: hit 784 → 896 tokens (+112 ≈ 14%). #50992 removes ARC batch-eviction quadratic cost.
- **Multimodal #50390**: EPD drops decode-side duplicate image preprocessing and moves preprocessing to GPU (256²→2048² speedup 2.3~8.6×). Key: encoder instances allocate no KV cache, so the frontend doesn't fight the LM for resources — only safe after "splitting E out".
- **MTP/SP kernel speedups #50904 (2.0×) / #51070 (1.5~3×) / #50230**: #50904 fixes MTP `set_skip_topk(True)` not really skipping redundant recompute (7 lines); #51070 merges Kimi-K3 SP's multiple all-gathers into one final; #50230 enables PDL to cut kernel-launch serialization. All main-only, await v0.26.1 or cherry-pick.
- **Serving #51089 / #49206 / #50289**: HTTP-header priority parsing; #49206 fixes PRIORITY scheduler **silently skipping** requests (correctness fix — evaluate soon).
- **Model/hardware #51045 / #49453 / #47106**: Ling 3.0 Flash full stack (BF16 + MTP + parser merged to main at once); CPU MLA backend runs DeepSeek-V2/V3 on pure CPU; NVFP4 CuTeDSL MoE supports `swiglu-oai` / `relu²` (the Step-3.5 NVFP4 kernel gap flagged 08-01).

## 5. 🟠 SGLang Highlights (main 08-05 ~ 08-07)

- **Major dep change #28836**: CUDA PyTorch stack bumped torch 2.11→2.13, triton 3.6→3.7.1 (34 files). Next stable tag likely ships torch 2.13. **Teams with custom Triton kernels should pre-test against main now.**
- **KV×spec #30393**: HiCache packed/sidecar draft cache (see 3.1). Hidden accept-rate loss when HiCache + spec-decode both on.
- **PD disagg #30545**: disagg staging buffer finally coexists with radix cache (see 3.3).
- **Multimodal #32365 (35 files)**: rust-server native end-to-end Qwen VL, `SGLANG_RUST_SERVER=1`, no Python mm_processor, no fallback. Key architectural call: **MM output features never traverse ingress** — large buffers (features/grids/hashes/M-RoPE) stay in a `rid`-indexed sidecar; ingress only passes `MmEncoded{rid, input_ids}`; Python scheduler `take_mm(rid)` zero-copy.
- **Diffusion #33823 + 15 commits**: SGLang-Diffusion becomes a standalone product line (FLUX.2 residual gating −1.2%, FLUX.1 −1.1% lossless, ERNIE-Image e2e 15.63→15.00s, all bit-exact). SGLang is moving from "LLM inference framework" to "unified generative-workload serving layer".
- **Default change #33618**: MoE deferred finalize on by default (behavior changes on upgrade — regression-test); #33459 DFlash supports logprobs; #33138 cache_aware router random tie-break.

## 6. 🇨🇳 StepFun: MTP Still Stuck on an Open PR

**vLLM #49642 (still open, +18/−5, two weeks)**: Step3p5AMultiTokenPredictor builds draft blocks over `range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)`, i.e. MTP layer `layer_idx == num_hidden_layers`, one beyond base decode layers; several layer-indexed config tables in `step3p5.py` (layer_types / rope_theta / use_rope_layers / partial_rotary_factors / swiglu_limits) are length `num_hidden_layers` with no bounds check → adding `--speculative-config '{"method":"mtp"}'` on Step-3.7-Flash raises `IndexError` at load. Fix: bounds-check every layer-indexed lookup; out-of-range draft layers fall back to standard defaults.

**Why it matters**: Step-3.7-Flash's three selling points are "mixed sliding-window attention + MTP self-draft + sparse MoE (196B total / 11B active)" — MTP is exactly the official throughput pitch, yet won't launch on vLLM main. Same window, vLLM merged Ling 3.0 Flash's "BF16 + MTP + parser" whole stack to main. **The gap isn't model capability, it's upstream maintenance investment.**

**Action**: ① run Step-3.7-Flash without MTP for now, or cherry-pick #49642; ② for MTP gains short-term, SGLang (`--reasoning-parser step3p5 --tool-call-parser step3p5` + NVFP4) is steadier.

## 7. ⚖️ Side-by-Side

| Dimension | vLLM | SGLang |
|-----------|------|--------|
| KV reuse focus | External storage ecosystem: Mooncake tenant/group/official wheel, intra-block tail reuse | Internal tier coupling: HiCache draft-state + observability |
| Multimodal speedup | Split physical topology: E/P/D separation kills dup transform, preprocess to GPU (1.9~8.6×) | Change execution language: whole vision pipeline pure Rust, worker pool + rid sidecar zero-copy |
| Attitude to Python | Keep fallback + transformers compat, breadth-first | Unsupported = fail to start, no Python fallback, determinism-first |
| Dep cadence | FlashAttention → torch stable ABI, gradual decouple | torch 2.11→2.13 one-shot big bump, aggressive |
| Product boundary | LLM/VLM serving, front-end Rust + scheduling governance | Big expansion into diffusion (FLUX/ERNIE-Image/Ideogram/Z-Image + dp-size) |
| Step support | Official recipe complete, but MTP won't launch (#49642 open) | NVFP4 + trtllm_mha + dual parser usable, community activity frozen 07-06 |

**One-line contrast**: both solve the same problem's two halves — "the path before a request hits the GPU and after cache leaves VRAM". vLLM re-partitions physical resources (encoder as separate node, preprocess to GPU, KV to external multi-tenant store); SGLang rewrites the execution carrier (pipeline into Rust, draft state into tiered cache, PD transport aligned to grid). Choose: breadth + fast new-model onboarding → vLLM; tail-latency determinism + can absorb torch 2.13 cadence → SGLang.

> Sources: GitHub API direct queries of vLLM / SGLang releases, main commits, and named PRs (#30393 / #30545 / #28836 / #48069 / #44956 / #50507 / #49206 / #51045 / #49642, etc.). Perf numbers and line deltas are from official repos / PR descriptions; judgments and selection advice are analysis — test against your own workload.
