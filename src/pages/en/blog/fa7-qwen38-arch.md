---
title: 'Frontier Architecture Decoding Notes (7): Qwen3.8 Dual Checkpoint — Dense 27B vs Sparse Flash-Next (360GB weights only to buy 6B activation)'
description: 'Based on a direct scan of both repos'' master branch (model cards, config.json, weight indexes, and safetensors header inspection), this post breaks down the architectural and weight-organization differences between Qwen3.8-Flash-Next (Qwen4Exp sparse MoE, 125B main / 6B active / 360GB BF16) and Qwen3.8-27B (Qwen3_5 dense, 55.6GB BF16). The former trades larger static storage for lower token-level activation; the latter is small and direct. Includes 3 hand-drawn architecture/capacity diagrams.'
pubDate: 2026-08-28
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa7-qwen38-arch
layout: ../../../layouts/BlogPost.astro
---

## 0. Conclusion First

These two checkpoints are **not size variants of the same model** — they are two weight sets with clearly different architectures and weight-organization schemes:

- **Qwen3.8-Flash-Next** = `Qwen4ExpForConditionalGeneration` **sparse MoE**. Main model is **125B** by card wording, ~**6B** active per token; plus ~**51B N-gram embedding** and **MTP** weights. Repo has **131 BF16 safetensors shards**, ~**360 GB (335.276 GiB)**.
- **Qwen3.8-27B** = `Qwen3_5ForConditionalGeneration` **dense** model. Repo has **18 BF16 safetensors shards**, ~**55.6 GB (51.747 GiB)**.

Flash-Next's weight files are **6.48×** those of 27B, yet it **does not compute all weights per token** — most of the storage comes from 512 experts and N-gram lookup tables. "Flash" mainly denotes a **compute/access-efficiency design**, not a smaller download. Both share many tokenizer / vision-preprocess / generation configs, but config / index / weights all differ — **shards are not interchangeable**.

> **Model-name correction update (2026-08-28)**: fa6 (Gated DeltaNet) in this series, based on a then-current snapshot, judged that "`qwen3.8-27B` does not exist; Qwen3.8 is MoE like 2.4T-A95B." A direct repo scan on 2026-08-28 confirms: **`Qwen3.8-27B` exists as a real dense checkpoint** (55.6GB BF16, `Qwen3_5ForConditionalGeneration`). fa6's judgment held for its snapshot; this repo now exposes the 27B dense checkpoint, so we correct it here. Repo revision short IDs: Flash-Next `2741eec1`, 27B `1098534a`.

## 1. Architecture Overview: Same Vocab/Context, Opposite Skeletons

Both share a **248,320 vocab** (padded) and **262,144 native context**, but build capacity in opposite ways: Flash-Next uses a **narrow backbone + experts / N-gram lookup**, while 27B uses a **wide backbone + per-layer dense FFN**.

<figure class="arch-fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Flash-Next vs 27B architecture overview">
  <defs>
    <style>
      .ttl{font:700 14px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .nm{font:700 16px -apple-system,'PingFang SC',sans-serif}
      .kv{font:12px -apple-system,'PingFang SC',sans-serif;fill:#374151}
      .sub{font:11px -apple-system,'PingFang SC',sans-serif;fill:#6b7280}
      .blue{fill:#eff6ff;stroke:#1d4ed8;stroke-width:1.5}
      .terra{fill:#fff7ed;stroke:#c2410c;stroke-width:1.5}
      .chip{fill:#fff;stroke:#cbd5e1;stroke-width:1}
      .chipb{fill:#dbeafe;stroke:#1d4ed8;stroke-width:1}
      .chipg{fill:#ecfdf5;stroke:#047857;stroke-width:1}
      .chipa{fill:#fef9c3;stroke:#a16207;stroke-width:1}
    </style>
  </defs>
  <text x="340" y="22" text-anchor="middle" class="ttl">Same 248,320 vocab · 262,144 context, opposite skeletons</text>
  <rect class="blue" x="20" y="40" width="300" height="216" rx="10"/>
  <text x="170" y="66" text-anchor="middle" class="nm" fill="#1d4ed8">Qwen3.8-Flash-Next</text>
  <text x="170" y="84" text-anchor="middle" class="sub">Qwen4Exp · sparse MoE · 125B main</text>
  <rect class="chip" x="36" y="96" width="268" height="30" rx="6"/>
  <text x="170" y="115" text-anchor="middle" class="kv">Narrow backbone width 2,560 · 48 layers</text>
  <rect class="chipg" x="36" y="134" width="268" height="30" rx="6"/>
  <text x="170" y="153" text-anchor="middle" class="kv">512-expert MoE (10+1 per token)</text>
  <rect class="chipa" x="36" y="172" width="268" height="30" rx="6"/>
  <text x="170" y="191" text-anchor="middle" class="kv">51B N-gram lookup (128 shards)</text>
  <rect class="chipb" x="36" y="210" width="268" height="30" rx="6"/>
  <text x="170" y="229" text-anchor="middle" class="kv">≈6B active/token · 360 GB weights</text>
  <rect class="terra" x="360" y="40" width="300" height="216" rx="10"/>
  <text x="510" y="66" text-anchor="middle" class="nm" fill="#c2410c">Qwen3.8-27B</text>
  <text x="510" y="84" text-anchor="middle" class="sub">Qwen3_5 · dense · 27B</text>
  <rect class="chip" x="376" y="96" width="268" height="30" rx="6"/>
  <text x="510" y="115" text-anchor="middle" class="kv">Wide backbone width 5,120 · 64 layers</text>
  <rect class="chip" x="376" y="134" width="268" height="30" rx="6"/>
  <text x="510" y="153" text-anchor="middle" class="kv">Per-layer dense FFN (mid-dim 17,408)</text>
  <rect class="chip" x="376" y="172" width="268" height="30" rx="6"/>
  <text x="510" y="191" text-anchor="middle" class="kv">No experts · no N-gram axis</text>
  <rect class="chipb" x="376" y="210" width="268" height="30" rx="6"/>
  <text x="510" y="229" text-anchor="middle" class="kv">27B active/token · 55.6 GB weights</text>
</svg>
<figcaption>Figure: same vocab/context, but Flash-Next piles capacity via "narrow backbone + 512 experts + 51B N-gram lookup", while 27B uses "wide backbone + per-layer dense FFN".</figcaption>
</figure>

## 2. Precision and Capacity Basis

Both repos are **BF16, not FP8**. config declares `text_config.dtype = bfloat16` (`mamba_ssm_dtype: float32` is SSM-compute only, not FP32 weights). Header scan confirms: Flash-Next has 1,655 BF16 tensors + 3 I64 metadata; 27B has 1,199 BF16 tensors. Neither has `quantization_config`. Separate quantized repos: `Flash-Next-FP8` ≈ 185.5 GB, `27B-FP8` ≈ 30.9 GB.

| Item | Flash-Next | 27B |
|---|---|---|
| Shards | 131 | 18 |
| Index payload | 359.999963 GB / 335.276 GiB | 55.562856 GB / 51.747 GiB |
| Actual .safetensors sum | 360.000193 GB / 335.276 GiB | 55.563007 GB / 51.747 GiB |
| Card param wording | main 125B + 51B N-gram + 4B MTP | 27B |
| License | qwen-community-1.0 | Apache-2.0 |

The capacity gap comes almost entirely from the weights themselves (non-weight files are both ~23 MB, nearly identical).

## 3. Architecture Differences (from config.json)

| Dimension | Flash-Next | 27B | Effect on weights |
|---|---|---|---|
| LM width | 2,560 | 5,120 | Flash projections narrower, but more experts |
| LM layers | 48 | 64 | 27B has more layers |
| Hybrid layout | 12×(3×[GDN→MoE]→1×[QSA→MoE]) | 16×(3×[GDN→FFN]→1×[Gated Attn→FFN]) | both are 3 linear + 1 full-attn cycle |
| Linear-attn layers | 36 (48 V head / 16 QK head / dim 128) | 48 | same structure |
| Full-attn layers | 12 QSA layers | 16 Gated Attention layers | Flash has QSA indexer weights |
| FFN form | 512 experts; 10 routed + 1 shared per token | one dense FFN per layer | Flash stores all experts; 27B stores one set |
| Expert/FFN mid-dim | routed/shared = 640 | dense FFN = 17,408 | Flash scales via expert count; 27B via FFN width |
| N-gram embedding | yes; base vocab 2,000,000, 128 shards, into layer 2 | none | Flash adds ~51B lookup params |
| Residual | Gated Residual, 4 branch, rank 320; hyper-connection tensors | no counterpart | Flash adds a residual-mix set |
| Vision→LM proj out | 2,560 | 5,120 | same preprocess, different connector weights |

## 4. Weight Organization in the Index

**Flash-Next packs experts into large tensors**: `layers.0.mlp.experts.gate_up_proj` shape `[512,1280,2560]`, first dim = 512 experts, then distributed across 131 shards. Each token routes only 10 routed + 1 shared, yet **the checkpoint must store all 512 experts** — compute approaches the few active experts, but disk / reachable storage approaches all of them.

**The N-gram table is an extra capacity axis**: the index holds 128 `ple_embedding.ngram_embedding.shard_*.weight`, shape `[2,500,012, 160]`, totaling **51.2B params / 102.4 GB**. The official note stresses these are fetched mainly via local n-gram lookup, not the per-token regular matmul budget; one design goal is easier Host-Memory placement with async prefetch. But you cannot skip these shards on download, and whether offload is efficient still depends on the inference framework.

**27B uses plain dense FFN**: `gate_proj/up_proj/down_proj` shapes `[17408,5120]`/`[5120,17408]`, one set per layer; capacity from 64 layers + 17,408 mid-dim, no hundreds of expert copies.

## 5. Weight Storage Breakdown (by tensor name)

<figure class="arch-fig">
<svg viewBox="0 0 680 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Flash-Next vs 27B weight storage breakdown">
  <defs>
    <style>
      .ttl{font:700 14px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .lab{font:12px -apple-system,'PingFang SC',sans-serif;fill:#374151}
      .sm{font:10.5px -apple-system,'PingFang SC',sans-serif;fill:#fff}
      .smk{font:10.5px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
    </style>
  </defs>
  <text x="20" y="20" class="ttl">Where the weights go? Flash-Next 95% in 'experts + N-gram', 27B 62% in 'dense FFN'</text>
  <text x="20" y="52" class="lab" fill="#1d4ed8">Flash-Next · 360 GB (equal-width proportion bar)</text>
  <rect x="60" y="60" width="402.6" height="34" fill="#047857"/>
  <text x="261" y="81" text-anchor="middle" class="sm">Routed experts 241.6GB · 67.1%</text>
  <rect x="462.6" y="60" width="170.4" height="34" fill="#a16207"/>
  <text x="547.8" y="81" text-anchor="middle" class="sm">N-gram 102.4GB · 28.4%</text>
  <rect x="633" y="60" width="27" height="34" fill="#94a3b8"/>
  <text x="646" y="108" class="smk">Others≈16GB (MTP/vision/attn)</text>
  <text x="20" y="132" class="lab" fill="#c2410c">27B · 55.6 GB (equal-width proportion bar)</text>
  <rect x="60" y="140" width="369.6" height="34" fill="#c2410c"/>
  <text x="244.8" y="161" text-anchor="middle" class="sm">Dense FFN 34.2GB · 61.6%</text>
  <rect x="429.6" y="140" width="230.4" height="34" fill="#94a3b8"/>
  <text x="544.8" y="161" text-anchor="middle" class="smk">Others≈21.3GB (MTP/vision/attn)</text>
  <rect x="60" y="200" width="14" height="14" fill="#047857"/><text x="80" y="211" class="lab">Routed expert FFN</text>
  <rect x="200" y="200" width="14" height="14" fill="#a16207"/><text x="220" y="211" class="lab">N-gram lookup</text>
  <rect x="320" y="200" width="14" height="14" fill="#c2410c"/><text x="340" y="211" class="lab">Dense FFN</text>
  <rect x="440" y="200" width="14" height="14" fill="#94a3b8"/><text x="460" y="211" class="lab">Others (MTP/vision/attn...)</text>
  <text x="20" y="250" class="lab">Point: Flash-Next's big files come mainly not from attention/vision tower,</text>
  <text x="20" y="268" class="lab">but from all routed experts + N-gram table; 27B is smaller not only for fewer layers,</text>
  <text x="20" y="286" class="lab">but because it has no 512 experts and no 51B N-gram axis.</text>
</svg>
<figcaption>Figure: storage breakdown by tensor name (equal-width proportion bars for easy structure comparison). Flash-Next Routed experts 67% + N-gram 28% ≈ 95%; 27B is mainly dense FFN (62%).</figcaption>
</figure>

## 6. Which Files Are Shared / Not Interchangeable

**Identical SHA256 (reusable)**: `chat_template.jinja`, `configuration.json`, `generation_config.json`, `merges.txt`, `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `video_preprocessor_config.json`, `vocab.json` — tokenizer, vocab, image/video preprocess, and generation config are highly consistent.

**Must be treated as model-specific (not interchangeable)**: `config.json`, `model.safetensors.index.json`, all `model-*.safetensors`, README, LICENSE, `.gitattributes`. Both repos use the generic `model-00001-of-...` naming, but shard counts, index mappings, and internal tensor names differ completely — **do not pair one repo's shard #1 with another repo's index**. Note: identical vision-preprocess files ≠ identical vision weights (LM hidden dim 2,560 vs 5,120, different connector).

## 7. Implications for Inference Deployment

<figure class="arch-fig">
<svg viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Storage vs activation: Flash-Next trades big storage for low activation">
  <defs>
    <style>
      .ttl{font:700 14px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .lab{font:12px -apple-system,'PingFang SC',sans-serif;fill:#374151}
      .v{font:700 12px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .note{font:12px -apple-system,'PingFang SC',sans-serif;fill:#b45309}
    </style>
  </defs>
  <text x="20" y="20" class="ttl">Core principle: trade 'larger static storage' for 'lower token-level activation'</text>
  <text x="20" y="52" class="lab">Static storage (BF16, GiB)</text>
  <rect x="170" y="40" width="460" height="26" fill="#e5e7eb"/>
  <rect x="170" y="40" width="460" height="26" fill="#1d4ed8"/>
  <text x="640" y="58" text-anchor="end" class="v" fill="#fff">335.3</text>
  <rect x="170" y="72" width="70.3" height="26" fill="#c2410c"/>
  <text x="246" y="90" text-anchor="start" class="v">51.7</text>
  <text x="20" y="78" class="lab" fill="#1d4ed8">Flash</text>
  <text x="20" y="110" class="lab" fill="#c2410c">27B</text>
  <text x="20" y="150" class="lab">Per-token activation (B params)</text>
  <rect x="170" y="138" width="480" height="26" fill="#e5e7eb"/>
  <rect x="170" y="138" width="480" height="26" fill="#c2410c"/>
  <text x="656" y="156" text-anchor="end" class="v" fill="#fff">27</text>
  <rect x="170" y="170" width="106.7" height="26" fill="#1d4ed8"/>
  <text x="282" y="188" text-anchor="start" class="v">6</text>
  <text x="20" y="166" class="lab" fill="#1d4ed8">Flash</text>
  <text x="20" y="198" class="lab" fill="#c2410c">27B</text>
  <rect x="170" y="218" width="460" height="34" rx="8" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="184" y="240" class="note">Flash storage is 6.48× of 27B, but activation only 6B (≈22% of 27B). "Flash" saves per-token compute/memory access, not download size.</text>
</svg>
<figcaption>Figure: same-scale comparison (storage full-scale at 335 GiB, activation full-scale at 27B). Flash-Next uses 6.48× static storage to buy token-level activation far below 27B.</figcaption>
</figure>

Static weights only (excluding runtime / activation / KV cache / framework overhead / fragmentation): 27B ≈ **51.75 GiB** (near single-card / few-card); Flash-Next ≈ **335.28 GiB** (usually needs multi-card, multi-node, quantization, or Host-Memory offload). Actual device memory cannot equal "6B activation" — expert weights and N-gram tables must still be reachable somewhere.

- **Shard count (131/18) ≠ GPU count / TP degree**: runtime partitioning is decided by the inference framework.
- **Framework compatibility**: the workspace `vllm-v0.21.0` has `qwen3_5` / `qwen3_5_mtp` paths (close to 27B), but **no `qwen4_exp` / Flash-Next implementation was found** — Flash-Next must be tested against the model card's latest vLLM/SGLang recipe; do not assume direct loading.
- **Long context**: both native 262,144 tokens, YaRN-extensible to 1M; YaRN changes position encoding only, not static weight size, but greatly increases KV cache and runtime memory.

## 8. Selection Guidance

| Preference | Better fit | Reason |
|---|---|---|
| Local deploy threshold, storage, VRAM | **Qwen3.8-27B** | ≈55.6 GB BF16, dense and direct |
| Larger capacity, lower per-token activation | **Qwen3.8-Flash-Next** | 125B main + N-gram extension, ≈6B active |
| Minimize download / VRAM | the **-FP8** variants | the two URLs here are non-quantized BF16 |

<div class="keybox">
<strong>One-line memory:</strong> Qwen3.8-27B is "a 27B model with one dense FFN set"; Qwen3.8-Flash-Next is "a narrow backbone + 512 experts + 51B N-gram lookup + special residual/attention" large-capacity sparse body. The former is small and direct; the latter trades larger static storage for lower token-level activation — 6.48× storage, 0.22× activation.
</div>

---

*Next: how Flash-Next's 512 experts + N-gram lookup are partitioned and offloaded in inference frameworks (EP / Host-Memory prefetch / quantization), and how it fundamentally differs from the "attention-modification" routes of DeepSeek V4 and MiniMax M3.*
