---
title: vLLM & SGLang Community Tracker · 2026-08-11：vLLM v0.27.0 ships in one drop; SGLang triple-fixes the unified memory pool and wires up PD disaggregation
description: LLM Infra Daily #8. Window 08-10~08-11: vLLM ships v0.27.0 (561 commits — full Kimi K3 stack / PyTorch 2.13 environment-level breaking upgrade / MRv2 to non-generative loads / Rubin sm_107); SGLang fixes three silent-correctness defects on the new unified-memory-pool data plane in one day and simultaneously wires up PD disaggregation and DSPARK speculative decoding. The theme of the cycle — delivering on one side, paying debt on the other.
pubDate: 2026-08-12
series: 社区跟踪手记
lang: en
altLang: zh
altHref: /blog/tr3-vllm-sglang-20260811
layout: ../../../layouts/BlogPost.astro
---

## ★ Most Worth Your Attention Today

**SGLang #33974 "unified pool supports DSPARK speculative decoding + fixes two NaN root causes"** — one PR that demonstrates two silent-corruption bugs at completely different levels, both the same classic shape: module A's implicit assumption is broken by module B, with no assertion in between. Understanding it gives you a self-check method portable to any framework (see "Principles / Code Walkthrough" below).

## 1. Daily Overview

- **Headline**: vLLM ships **v0.27.0** (08-10 21:18 UTC, 561 commits / 242 contributors / 64 first-timers), 14 days after v0.26.0 (07-27). SGLang stays at **v0.5.17** (08-08). Step series: no new release.
- **Main-branch activity**: vLLM main 51 commits, SGLang main 66 commits (same window).
- **Theme of the cycle: delivering on one side, paying debt on the other.**
  - vLLM delivers two weeks of work in one 561-commit release (full Kimi K3 stack, PyTorch 2.13 environment-level breaking upgrade, MRv2 to non-generative loads, Rubin sm_107).
  - SGLang fixes three silent-correctness defects on the unified memory pool (new data plane) in a single day.
  - Together it is a clear signal: **the new data plane is replaying the same silent bugs KV cache hit early** — virtual/physical id separation, page-envelope layout, relocatable pages, each manufacturing "no error thrown, but the answer is wrong".

## 2. vLLM Notes

### Release v0.27.0 (08-10 21:18 UTC)

561 commits / 242 contributors (64 new). Core content:

- **Full Kimi K3 stack in one pass**: model & kernels (#50089/#50000), Python (#50093) and Rust (#50104) frontends, AttnRes kernel (#50090), DeepGEMM (#50458), compressed-tensors quantized weights (#50500), DSpark AR fusion (#50242), shared experts splittable rather than replicated (#50656).
- **Major dependency upgrade (breaking environment change)**: PyTorch 2.13.0 + torchvision 0.28.0 + Triton 3.7.1 (#48155), with XPU (#48677) and CPU (#50412) following; Transformers 5.14.1, FlashInfer 0.6.16.post3, AITER 0.1.19, NCCL 2.30.7 (DeepEPv2 enabled in image), Helion 1.4.0.
- **FlashAttention 4 deepened on SM100**: FP8 KV cache (#42569), headdim-256 (#42669), plus new JIT warmup infra (#47451) and runner-owned Triton kernel warmup (#49903) to kill first-request compile stalls.
- **Model Runner V2 to non-generative loads**: encoder-only attention (#49331), sequence pooling for embedding/classification (#48791), encoder token classification (#50293) and embedding (#50574), BGE-M3 pooling (#50661), CPU multimodal (#50073), multi-layer MTP speculator (#48892).
- **Large-scale serving resilience**: simplified fault-tolerance framework for DP+EP externally-balanced deployments (#44428), async prep for elastic EP scale (#47288).
- **Next-gen hardware early enablement**: NVIDIA Rubin sm_107 target (#49387) + NVLink all-reduce on SM107 (#49647); ROCm gfx1250 (#46516).
- **Removals**: models Plamo2 (#49729), Ouro (#49786); args `max_num_partial_prefills` / `max_long_partial_prefills` (#49244).

> **Deployment impact**: this is an **environment-level, not package-level** upgrade — torch 2.13 means you cannot `pip install -U vllm` in place; you must rebuild the image/virtualenv and re-validate all self-compiled kernel plugins and torch extensions. Suggested cadence: rebuild image → smoke test → accuracy regression → canary. Do not roll to prod directly.

### Correctness fix #51727 DeepSeek V4/3.2 over-counts vocab size, crashes structured output

`DeepseekV4Tokenizer.__len__` returns `vocab_size + len(get_added_vocab()) = 128000 + 1283 = 129283`, but 3 of those 1283 added tokens already have ids < 128000 (they live in the base vocab), so the true vocab is exactly `config.vocab_size = 129280` — the +3 blows up guided decoding (issue #50924 / #51467).

Historical note: this override was written in the era when "HF fast tokenizer `len()` did not yet include added tokens"; transformers 5.x `TokenizersBackend` now does — so it is both redundant and wrong (the same stale pattern in `deepseek_v32.py` was deleted too, net −11 lines).

Impact: services running `DSV4-Flash + --structured-outputs-config.backend guidance` fail to start. The fix landed on main after v0.27.0 and is not in this release.

### Perf #51430 Narrow the eager CUDA graph region of DeepSeek V4 → short-prefill TTFT −53.7%

Move the "producer" ops (Q up-projection, fused Q norm/RoPE/KV write, indexer input prep, MLA/indexer compression) from the eager region back into the forward captured by CUDA graph; the eager region keeps only sparse indexer + forward_mqa.

Measured (DeepSeek-V4-Flash / 4×GB200 / Rust frontend / FP8 KV / FP4 indexer cache / PIECEWISE breakable graph / DeepGEMM mega-MoE / MTP2; 1000 reqs, 128 input tokens, concurrency 1):

| Metric | main | this PR | change |
|--------|------|---------|--------|
| Mean TTFT | 41.45 ms | 19.20 ms | **−53.7%** |
| P50 TTFT | 41.18 ms | 19.09 ms | **−53.7%** |
| Throughput (sat @ conc 64) | 430.05 req/s | 433.29 req/s | +0.75% |

GSM8K 1319Q 5-shot: TP4/EP4+MTP2 95.30%, DP4/EP4+MTP2 95.07%, no accuracy regression.

> ⚠️ Context: the PR explicitly notes "a prior same-shape region narrowing was rolled back because real model output was corrupted." This version deliberately keeps `forward_mqa` and the sparse indexer in the eager region (rather than capturing the whole attention backend), and uses a persistent eager scratch to keep tensor addresses stable for eager-region consumers.
> Impact: huge for short-prefill / low-concurrency interactive scenarios; this is a 128-input-token test, so long-prefill won't scale proportionally (kernel-launch overhead's share of TTFT shrinks under real compute).

### Security #49948 Audio sample-rate forgery bypasses decode-duration guard → 11.7 GiB per request, OOM kills the API server

`load_audio_soundfile`'s `max_duration_s` guard computes duration via `f.frames / f.samplerate`, trusting the container header entirely. An attacker sets `samplerate` to 655350 (FLAC max) with 8 channels, making hundreds of millions of frames "look like" a short clip and bypassing the duration check, while `f.read()` actually allocates up to 11.7 GiB of float32 PCM.

Fix: add `VLLM_MAX_AUDIO_DECODE_BYTES` (default 256 MiB), bounding the estimated PCM buffer independently of sample rate — soundfile path pre-checks `frames × channels × 4`, PyAV path counts bytes incrementally during decode. Two guards, two jobs: the duration guard early-rejects legitimately long files; the byte guard ignores claimed duration and rejects forgery.

Impact: any service exposing audio / multimodal endpoints publicly should care. After SGLang 0.5.5–0.5.12 multimodal path traversal, this is another resource-type DoS on the multimodal ingress — **multimodal ingress is becoming the new attack surface for LLM serving**.

### Feature / PD #48414 KV offload adopts a "canonical CPU layout", fully decoupled from parallel topology

Built on #48408 (per-layer canonical KV page mapping). Offloaded KV is stored in a layout independent of parallelism degree: each worker scatters its page fragment into the canonical position within the CPU region shared by the whole worker group.

```
legacy row:     [worker0: T0|T1][worker1: T0|T1]...   each worker owns a slot
canonical row:  [T0 canonical page][T1 canonical page]  all heads + all tokens, stored once
```

MLA latent and replicated GQA heads are stored once instead of once per rank (non-writers run empty stores). Config `kv_connector_extra_config: {"canonical_layout": true}`; requesting it on a configuration it cannot certify fails at startup instead of silently degrading (a design stance worth copying); persistence-format identity (v1-nhd/v1-hnd) is folded into the FileMapper namespace, preventing canonical / legacy / cross-family bytes from resolving to the same files. 4×H100 end-to-end: tp1 legacy, tp1/tp2 canonical, tp4+dcp2 canonical, token-exact.

Impact: a key step toward "KV cache topology-agnostic reuse" — previously TP4-persisted KV became garbage at TP2; the canonical layout allows cross-topology sharing and migration, directly affecting the cost model of multi-cluster / elastic scaling / cold-start warmup.

### Also worth noting

- #47352 MRv2 MTP shares topk index buffer across draft steps (DSV3.2-NVFP4 / TP4, 1629.90 → 1640.71 tok/s, ~+0.66%; bigger win: introduces `AutoRegressiveSpeculatorCallbacks` so model-specific optimizations stop polluting the generic speculator)
- #50484 Kimi-K3 supports DCP
- #51602 fix DSpark `parallel_drafting_token_id` init bug
- #51507 top-k/top-p Triton sampling kernel moves to 8 warps
- #49436 3D grid tiling for state-copy Triton kernel
- #51265 Ling-3.0-flash-fp8 support
- #51178 Rust gRPC explicit DP rank routing
- #51573 fix YAML `false`-typed `BooleanOptionalAction` not emitting `--no-{key}`

## 3. SGLang Notes

The only SGLang storyline today: the **unified memory pool (`--enable-unified-memory`)**. This new data plane — "KV + mamba state in one pool" — designed for hybrid Mamba models (Kimi-Linear / Kimi-K3), had three silent-correctness defects fixed within 24h and simultaneously wired up PD disaggregation and speculative decoding. No new tag; stable line remains v0.5.17 (08-08).

### Key PR · Correctness #33974 Unified pool supports DSPARK speculative decoding + fixes two NaN root causes　★ Most worth attention

13 files +661/−21, merged 08-10 17:35Z. Previously `server_args.py` hard-asserted `self.speculative_algorithm is None`, forbidding any speculative decoding on the unified pool; this PR connects chained DSPARK and, while validating end-to-end on Kimi-K3 (TP8, 2×4 GB300) and Kimi-Linear-48B (TP1×8 B300-class), surfaced two real root causes at completely different levels (see "Principles / Code Walkthrough").

The most glaring accuracy line: clean start + zeroing fix only, GSM8K drops to 0.120 while accept rate shows 0.52 (fake); with both fixes, back to 0.985 / 0.990.

EAGLE / tree / DFLASH / NGRAM remain forbidden pending their own virtual-vs-dense loc audits.

### Correctness fix #33517 unified pool × Triton backend × deterministic inference all on → NaN logits (silent locally)

Root cause: `--enable-deterministic-inference` routes Triton extend to `TritonAttnBackend._forward_extend_unified` — a single-stage kernel that reads both prefix and extend KV halves from the pool (the default two-stage path only reads prefix from the pool; the extend half comes straight from k/v inputs). But the passed `forward_batch.out_cache_loc` is untranslated: in the unified pool it is a virtual id, while the prefix `kv_indices` were already flipped to physical ids in `init_forward_metadata` → prefix read by physical, just-written extend token read by virtual, misaligned data becomes NaN through the LM head.

"Three-missing-one" never reproduces (no determinism → two-stage kernel doesn't read extend from pool; no unified pool → out_cache_loc is physical; no Triton → other backends don't take this kernel), so all existing tests missed it.

> ⚠️ The nastiest part: only armed `SGLANG_ENABLE_ASYNC_ASSERT` (CI enables it) aborts with "NaN detected! sampler: next_token_logits"; locally without it, the job finishes normally and returns finite top-k logprobs — **completely silent**.

### Feature / PD disaggregation #33362 Unified memory pool supports PD disaggregation (kimi-linear MLA hybrid-Mamba)

15 files +835/−51, stacked on #33517 (otherwise test configs would hit the NaN above).

The conflict: the unified pool stores virtual ids above the allocator, interleaves layers in a page-major "envelope" layout, and physically relocates pages on compaction; while the PD transfer engine (mooncake RDMA) addresses raw bytes by `ptr + index * item_len` outside any CUDA stream ordering. Three assumptions collide head-on.

The three-piece solution:

1. **Transfer switches to "whole envelope + physical id"**: `get_contiguous_buf_infos()` registers a single region (raw byte buffer), `item_len` = one page's full-layer envelope bytes (guaranteed contiguous by page-major construction) — one RDMA block moves every layer of that page; mamba side mirrors this with physical mamba slot as index. Per-tensor dim / layer-id metadata is deliberately left empty (envelopes can't be TP-resliced or PP-subsetted); `maybe_send_extra` fails loudly on item-length mismatch, catching "unified ↔ non-unified pool mixing" and "attn TP unequal".
2. **Virtual→physical translation collapses to the handoff point**: new `translate_kv_indices_for_transfer()` (identity on base, overridden by unified allocator), called once each in prefill's `send_kv_chunk` and decode's `pop_preallocated`.
3. **Compaction safety**: PD + unified forces lazy compaction (assert-gated; the immediate-release path would move pages while RDMA is in flight); new `disagg_move_gate` refuses to move a single page whenever any transfer might be in flight.

### Perf / PD disaggregation #34191 PD prefill server skips speculative-verify scratch — ~24 GB dead weight saved per rank

Only 3 files +23/−2, but large payoff. Under PD disaggregation the prefill server never runs `TARGET_VERIFY` (server_args already rejects `--enable-linear-replayssm-spec` for this reason), yet on hybrid-linear-attention models with speculative decoding (GDN/KDA) it still pays for the whole verify machinery:

- **Pool**: allocates per-draft-token mamba state snapshots for verify only, `intermediate_ssm_state_cache` ≈ `num_draft_tokens × entire mamba pool`. 256-slot pool (~6 GB ssm_state/rank) + `num_draft_tokens=4` (NEXTN 3 steps) = ~24 GB wasted/rank — alone enough to push deep-PP prefill layout into OOM during CUDA graph capture.
- **Graph**: captures the never-replayed target-verify CUDA graph.
- **Warmup**: dummy forward runs in `TARGET_VERIFY` mode.

All three gated by `disaggregation_mode == "prefill"`. Measured Qwen3.5-397B-A17B-FP8, 1P1D (TP4 prefill / TP4 decode, mooncake, both NEXTN `num_steps=3, eagle_topk=1, num_draft_tokens=4`), GSM8K 200Q 8-shot: accuracy 0.99 vs 0.98, MTP accept length 3.474 vs 3.481 (both within sampling noise), prefill `intermediate_ssm_state_cache` 4.31 GB → not allocated.

### Correctness fix #31700 DeepSeek-V4/V4-Pro DP-attention gather semantic error → hidden state amplified by attn_tp_size×

Trigger: `moe_a2a_backend=none + data_parallel_size>1 + attn_tp_size>1`.

Root cause: when MoE gather runs, `self_attn` has already reduced within the attention-TP group, so hidden states are replicated; but the existing `dp_gather_partial` treats them as unreduced partial contributions, and its reduce-scatter sums the replicated values again — every MoE layer multiplies hidden-state magnitude by `attn_tp_size`. NextN input-id gather is the same.

Fix: both switch to `dp_gather_replicate`; `input_ids` cloned before the two replicate gathers (MAX_LEN zeroes local input on non-leader attn-TP ranks, and `input_ids[:, None]` would alias the caller's tensor). Hidden state deliberately not cloned (dead after gather; cloning large activations per layer isn't worth it).

Reproduced at scale: 2×8 H200 (TP16 / DP4 → attn TP4×DP4) + DeepSeek-V4-Pro-NVFP4; independently on TP8×DP2 and 2×8 H100.

### Also worth noting

- #34240 DCP drops two per-layer launches on MLA target-verify path
- #33662 DSV4 EAGLE prefill drops host sync
- #34167 fix DSA top-k v2 dropping non-primary-rank output on CUDA 13.1+
- #33639 HiCache supports Mamba branch in Unified Radix Cache (Host-side node does component-wise incremental backup instead of recopying full KV)
- #30392 multimodal global cache decoupled from Mooncake (pluggable storage backend, works without Mooncake)
- #34234 DFLASH draft KV pool sized by its own attention geometry (previously reused EAGLE's "draft≡target" assumption, but DFLASH is independently trained: −11.9% underestimate at 8 KV heads, meaning the pool emits more tokens than both pools' real capacity)
- #33912 DFLASH draft KV pool size counted into DCP
- #31847 speculative decoding supports Inkling DSPARK
- #33484/#33085 ROCm hisparse swap-in copy fusion and 128-bit non-temporal copy
- diffusion side 15+ commits (Z-Image single-card BCG fix bit-exact with eager; LTX-2 quality=high fuses RMSNorm+modulate; H200 ltx23 single-stage denoise 45.85→43.24s)

## 4. Principles / Code Walkthrough: the two NaN root causes in SGLang #33974

Picked because one PR shows two silent corruptions at completely different levels, both the same classic shape: **module A's implicit assumption is broken by module B, with no assertion in between.** Understand it and you get a self-check method portable to any framework.

### Background: what is the "unified memory pool"

Hybrid models like Kimi-Linear / Kimi-K3 need both attention KV and Mamba state. The traditional approach pre-allocates two static pools via `torch.zeros` each, wasting fixed ratios. The unified pool puts both into one byte pool, interleaves layers in a page-major "envelope" layout, and introduces a virtual id ↔ physical id two-layer addressing (virtual ids above the allocator; physical pages can move on compaction).

The cost of flexibility: anything that touches physical memory while bypassing the allocator (kernels, RDMA, CUDA graph capture) must do explicit address translation; miss one spot and it won't error — it just reads someone else's bytes.

### Root cause 1: allocator doesn't zero pages on issue → NaN leaks from historical bytes into attention output

The trtllm-gen MLA kernel reads whole pages; rows beyond seq_len are masked arithmetically. The problem: arithmetic masks are unsafe against NaN — `NaN × 0` is still NaN, and once it enters a subsequent sum it poisons the whole row. The unified pool only zeroes the raw buffer once at startup; when a page is recycled and re-issued, historical bytes are exposed again. If the tail of a partially-filled last page happens to hold a NaN bit pattern, it leaks into attention output.

Speculative decoding amplifies exposure: every verify window lands on a fresh page at the sequence tail — that's why "it only blows up after adding speculative decoding."

Static pools are naturally immune (`torch.zeros` build, only writes bounded KV). The fix gives the shared pool the same guarantee at the allocation boundary: one contiguous envelope memset on the scheduling stream per page issued.

Cost measured at zero (one_batch_server, isl 8192 / osl 1024, K3 TP8, 2 repeats per bucket): bs 1/8/16/24/32 output tok/s change = −0.07% / −0.02% / +0.05% / +0.03% / +0.02%, TTFT within ±0.34% — per-bucket repeat variance exceeds the fix's delta every time.

> 💡 **Worth stealing**: they added `SGLANG_DEBUG_POISON_POOL` — fill the pool with bf16-NaN bit patterns at startup instead of zeros. This turns the "does the freed GPU heap happen to contain a NaN" lottery into a deterministic switch, kept as a permanent regression fixture for this class of bug (zero overhead by default). Any project building its own memory pool should have one.

### Root cause 2: int32 stride wraparound — a heisenbug that only appears under "unified pool + K3 + high load"

The DSPARK KDA verify kernel takes the `nv_cutedsl` fast path, compiled with a static CuTe layout — so the state pool's stride is a compile-time constant. On the unified pool, the envelope-strided KDA state view's slot stride reaches ~14M (fp32 ssm) / ~28M (bf16 conv) elements. Each constant individually fits int32, so the compiler folds the entire index arithmetic into 32 bits — `slot × stride` silently `mod 2³²` wraps once slot id exceeds ~153 (conv) / ~306 (ssm).

Two severity tiers: mild reads another slot's bytes — silent state corruption on clean start, GSM8K to 0.120 with a fake accept-rate spike; severe flies out of the allocation region into illegal access.

Why only under this triple condition?

- Static contiguous pool slot stride is only ~197K elements — never reaches the wrap point.
- Kimi-Linear's head dim fails the 128 contract and never takes this kernel (falls back to triton).
- Cold-start issues only small slot ids — so it's load-dependent: pressure rises, slot ids grow, then it breaks.

So it's unified-only, K3-only, load-only — a textbook heisenbug.

How proven: offline replay sentry — dump one real in-system call, rebuild state with "contiguous" and "faithful-stride" copies (gaps filled with zero / NaN respectively), then scan slot-id magnitude. Result: pre-fix, slots ≤75 bit-exact, 100–150 illegal access, ≥155 silent divergence; post-fix, bit-exact everywhere.

One-line fix: promote slot id to int64 before multiplying the state stride.

## 5. Actionable Conclusions for You

- **If you run Kimi-K3 with both `--enable-unified-memory` and DSPARK speculative decoding, pre-08-10 versions silently corrupt once load ramps** (no error, accuracy collapses, accept rate still looks great). Sync main now or wait for the next tag.
- **Counter-intuitive diagnostic signal**: an abnormally high and stable accept rate (e.g. 1.00 / 8.00) is a danger sign — `argmax(NaN) = 0` manufactures a stream of all-0 tokens' "perfect acceptance." Don't read it as tuning success.
- **The "three-missing-one" non-reproducible combos are the most dangerous.** #33517 is the perfect example: unified pool × Triton × deterministic inference, any two are fine. Treat "pool type × attention backend × speculation algo × determinism switch" as a Cartesian product for regression, not dimension-by-dimension.
- **General lesson (cross-framework)**: any kernel with "compile-time constant stride + 32-bit index arithmetic" must recompute whether `max_slot × stride` crosses `2³¹` after its data plane moves from a static contiguous pool to a large-stride view. This wraparound is a structural risk on vLLM's KV offload / canonical-layout path too (inference, not a community-reported defect).

## 6. Standing Topic Progress

**① PD disaggregation**: this window SGLang formally wires PD disaggregation onto the unified pool (#33362, 15 files +835/−51): "whole envelope + physical id" transfer, virtual↔physical translation collapsed to the handoff point, and PD+unified forced-lazy-compaction — all three at once; paired with #34191 so the PD prefill server no longer pays for speculative-verify scratch (~24 GB dead weight/rank saved). vLLM side continues KV-offload canonical layout (#48414, fully decoupled from topology) + existing NIXL/Dynamo hardening. Conclusion: PD disaggregation has moved from "works" to "can carry the new data plane" — any change that swaps the KV cache's underlying layout must add one address-translation for the RDMA engine.

**② Latest inference-architecture evolution**: the主轴 converges hard on "new data plane + speculative decoding orthogonal-combination correctness." SGLang's unified pool (KV+Mamba state in one) fixed three silent defects in 24h while connecting DSPARK and PD disaggregation; vLLM used v0.27.0 to deliver the full Kimi-K3 stack (incl. DSpark AR fusion #50242), MRv2 to non-generative loads, and Rubin sm_107 at once. Note: speculative decoding is retreating from "throughput boost" to "correctness carrier when combined with KV reuse / hybrid arch / PD disaggregation" — the most glaring bugs this cycle all surfaced "after turning on speculation."

**③ Pure PyTorch vs HuggingFace transformers**: both sides synced to PyTorch 2.13 "environment-level" this cycle (vLLM #48155 / SGLang earlier #28836); data-plane self-development trend unchanged. Layered conclusion holds: "model definition" converges to transformers for breadth (vLLM keeps generalizing the transformers backend, day-N zero-adaptation cost), "data plane & runtime" converges to self-dev for determinism (SGLang unified pool, Rust vision pipeline, pure Triton kernels). New signal: vLLM #51727 shows `deepseek_v32`'s `__len__` override was written in the era when "HF fast tokenizer `len()` didn't include added tokens," and transformers 5.x now does — stale bypasses are both redundant and wrong, a regression class to check when frameworks chase transformers upgrades.

**④ StepFun Step series adaptation**: sixth consecutive day of no real progress. Hard evidence: StepFun-ai org's latest push is still Step-Realtime-CLI (07-23, 08-10 had a push but commits stalled 07-17); Step-3.7-Flash main repo stalled 06-01 (303★); official vLLM fork stalled 05-28. vLLM side Step PRs: #49642 (MTP per-layer config out-of-bounds, IndexError on start) still open, #50290 etc.; SGLang side #32325 (NVFP4 MTP shared-head 2048≠4096) still open. WebSearch confirms Step-3.7/3.5-Flash community deployment guides are widespread (skillsbot / xiaolandeng / CSDN), but no new model in 24h. Conclusion unchanged: Step's performance selling point heavily depends on MTP's built-in drafts, and MTP is exactly the least stable piece; teams relying on Step in production should pre-check `--speculative-config '{"method":"mtp"}'` can start, and prepare a fallback to non-speculative paths.

## 7. Comparative View

"Delivering on one side, paying debt on the other" is the sharpest contrast of the cycle.

| Dimension | vLLM v0.27.0 | SGLang main (no new tag) |
|-----------|--------------|------------------------|
| This cycle | deliver 561-commit major | pay debt: unified pool triple-fix |
| Spine | breadth (K3 stack / MRv2 non-gen / Rubin) | depth (new-data-plane correctness + PD/spec wired) |
| PyTorch strategy | bump 2.13.0, env-breaking | bumped earlier (#28836, next tag carries it) |
| For deployers | rebuild image + accuracy regression | verify unified-pool regression before syncing main |
| Common | both aligned on PyTorch 2.13 (env deps converge); both treat "silent correctness defects" as first-class (vLLM #50323 NaN CI gate, SGLang async-assert + poison-pool fixtures) | |
| Divergence | vLLM exposes risk on the "upgrade path" (env-breaking, plannable) | SGLang exposes risk "inside the data plane" (unified-pool new code path, backstopped by regression fixtures) |

> Sources (all from raw GitHub API responses): vLLM v0.27.0 Release; vLLM PR #51727 / #51430 / #49948 / #48414 / #47352 / #50323; SGLang PR #33974 / #33517 / #33362 / #34191 / #31700; StepFun-ai org. All version numbers, PR numbers, and performance figures are from direct GitHub API queries; parts not inferred are marked "inference."
