---
title: 'Multimodal Decoding Notes (5): Efficient Systems & Frontier Inference — Attention / Inference Optimization / On-Device / RL'
description: The capstone pulls the view to the substrate — the "optimization techniques" that support all multimodal models. From the attention five-piece (FlashAttention/RoPE/GQA/MLA/RMSNorm), to efficient training & inference systems (mixed precision/KV Cache/PagedAttention/PD disaggregation/speculative decoding/SwiGLU), to on-device intelligence (LoRA/QLoRA/On-Device research/EcoSpec) and frontier inference RL (OAT/Ring-Zero/SPS/HiLS-Attention/exploration-paradox/AMVL).
pubDate: 2026-07-31
series: Multimodal Decoding Notes
lang: en
altLang: zh
altHref: /blog/mm5-efficiency-frontier
layout: ../../../layouts/BlogPost.astro
---

## 0. Why This Chapter

Parts (1)-(4) are about what multimodality "can do"; (5) is about "how to make it run, run fast, run on-device". **Every drop in inference cost makes VLA + world model (Part 4) one notch more feasible on-device in real time.** This chapter splits it into four layers.

## 1. The Attention Five-Piece: the Default Kit of Mainstream Decoder LLMs

- **FlashAttention (FA-1→FA-4)**: tiling + online softmax + kernel fusion; the N×N attention matrix stays in SRAM, never touching HBM. Bit-exact (not approximate), 2~4× speedup on A100, 128K context without OOM; H100 FA-3 FP8 hits 740 TFLOPs/s. Default in Qwen3/DeepSeek V3-V4/MiniMax/GLM/Kimi/Llama 5.
- **RoPE (Rotary Position Embedding)**: rotates Q/K in 2D subspaces to encode **relative position**, making attention depend only on offset `m−n`. `θ_i = 10000^(−2i/d)`; zero-param explicit relative position + NTK/YaRN for long context; **V is not rotated** is a common pitfall. Used in Qwen3/DeepSeek V4/MiniMax/Llama5/GLM/Kimi.
- **GQA (Grouped-Query Attention)**: split H query heads into G groups, **share KV heads within a group**, compressing KV Cache 4~8×. G=8 is the sweet spot (< 0.5% quality loss). In Llama2/3, all Qwen, Mistral, DeepSeek V2-V4, MiniMax.
- **MLA (Multi-head Latent Attention)**: jointly compress K/V into a low-dim latent vector c, cache only c, up-project on use → memory down to 1/4~1/10 with almost no accuracy loss. **Requires decoupled RoPE.** In DeepSeek V2/V3/V4.
- **RMSNorm**: a simplified LayerNorm (drops mean-centering and β), ~7~15% speedup, parity in quality, directly valuable for on-device VLA latency. In Llama/Qwen3/DeepSeek V4/MiniMax/Mistral/Gemma.

> The KV-Cache deciding factor for long-context / low-cost inference: GQA/MLA "save memory", FlashAttention "save IO", RMSNorm "save compute" — together they form today's efficiency baseline.

## 2. DeepSeek-V4 Architecture Deep-Dive: The Four-Stage Evolution of Compressing KV Cache (MLA → NSA → DSA → CSA+HCA)

> Drop the "attention five-piece" from (1) onto a flagship model: across three generations (V2→V3→V4), DeepSeek turned "KV Cache compression" into its core engineering line. Understand this line and you understand why long-context inference can cut cost by an order of magnitude — also the technical anchor for evaluating any long-context / low-cost-inference thesis.

### 2.1 One Main Line: Push KV Cache to the Limit

KV Cache is the cost anchor of long-context inference: every extra token means one more copy of K/V stored. DeepSeek's compression has **two orthogonal directions**:

- **Compress width (MLA)**: squeeze high-dim K/V into a low-dim latent vector `c`, cache only `c`, up-project on use → memory down to 1/4~1/10 with almost no accuracy loss, but needs decoupled RoPE (in V2/V3/V4).
- **Compress length (NSA→CSA+HCA)**: stop caching every token's K/V; keep only "a few compressed memory blocks" → the token count itself drops, and the length dimension is compressed.

Figure 1 shows the contrast between the two directions.

<figure class="fig">
<svg viewBox="0 0 680 220" role="img" aria-label="Two compression directions of KV Cache">
  <defs>
    <marker id="arr1" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--fg)"></path>
    </marker>
  </defs>
  <text x="165" y="20" text-anchor="middle" fill="var(--fg)" font-size="14" font-weight="600">Width · MLA</text>
  <g fill="var(--accent)" opacity="0.85">
    <rect x="36" y="38" width="13" height="116"></rect>
    <rect x="54" y="38" width="13" height="116"></rect>
    <rect x="72" y="38" width="13" height="116"></rect>
    <rect x="90" y="38" width="13" height="116"></rect>
    <rect x="108" y="38" width="13" height="116"></rect>
  </g>
  <text x="93" y="172" text-anchor="middle" fill="var(--muted)" font-size="11">full K/V per token</text>
  <path d="M132 96 H176" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr1)"></path>
  <rect x="186" y="88" width="22" height="18" rx="4" fill="var(--accent)"></rect>
  <text x="197" y="130" text-anchor="middle" fill="var(--fg)" font-size="13">c</text>
  <text x="197" y="150" text-anchor="middle" fill="var(--muted)" font-size="10">low-dim latent</text>

  <line x1="340" y1="28" x2="340" y2="186" stroke="var(--border)"></line>
  <text x="510" y="20" text-anchor="middle" fill="var(--fg)" font-size="14" font-weight="600">Length · NSA→CSA+HCA</text>
  <g fill="var(--accent)" opacity="0.5">
    <rect x="356" y="50" width="16" height="16"></rect>
    <rect x="378" y="50" width="16" height="16"></rect>
    <rect x="400" y="50" width="16" height="16"></rect>
    <rect x="422" y="50" width="16" height="16"></rect>
    <rect x="444" y="50" width="16" height="16"></rect>
    <rect x="466" y="50" width="16" height="16"></rect>
    <rect x="488" y="50" width="16" height="16"></rect>
    <rect x="510" y="50" width="16" height="16"></rect>
    <rect x="532" y="50" width="16" height="16"></rect>
    <rect x="554" y="50" width="16" height="16"></rect>
    <rect x="576" y="50" width="16" height="16"></rect>
    <rect x="598" y="50" width="16" height="16"></rect>
    <rect x="620" y="50" width="16" height="16"></rect>
  </g>
  <text x="488" y="86" text-anchor="middle" fill="var(--muted)" font-size="11">fine-grained K/V for all tokens</text>
  <path d="M510 110 H548" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr1)"></path>
  <g fill="var(--accent)">
    <rect x="556" y="100" width="26" height="22" rx="4"></rect>
    <rect x="588" y="100" width="26" height="22" rx="4"></rect>
    <rect x="620" y="100" width="26" height="22" rx="4"></rect>
  </g>
  <text x="606" y="146" text-anchor="middle" fill="var(--fg)" font-size="11">few memory blocks</text>
</svg>
<figcaption>Fig.1　Two compression directions of KV Cache: compress width (MLA, squeeze each column's K/V into a thin latent c) vs compress length (NSA→CSA+HCA, merge many columns into a few memory blocks). Orthogonal and stackable.</figcaption>
</figure>

### 2.2 Four-Stage Evolution

| Stage | Name | Method | What it saves | Note |
|---|---|---|---|---|
| ① | MLA | K/V→low-dim latent `c` | **width** | V2/V3/V4 baseline |
| ② | NSA (2025.02) | three branches cmp/slc/win, natively trainable | **length** | hardware-aligned (GQA-style grouping, balanced arithmetic intensity) |
| ③ | DSA (V3.2 transition) | Lightning Indexer | **compute** | saves compute not memory, bridges to V4 |
| ④ | CSA+HCA (V4) | three-level memory | **length** to the extreme | 1M context → only ~7800 memory |

- **NSA (Native Sparse Attention)**: compressed / selected / window branches are **natively trainable**; uses GQA-style grouping for arithmetic-intensity balance so it saturates Tensor Cores — not "post-hoc pruning" but sparse-from-training.
- **DSA (V3.2 transition)**: adds Lightning Indexer, a light scorer that quickly picks important tokens; **saves compute but KV still must be stored** ("saves compute not memory"), a bridge to V4.
- **CSA+HCA (V4 three-level memory)**: splits "long/short memory" into three explicit levels (see 2.3).

### 2.3 V4 Three-Level Memory: SWA + CSA + HCA

<figure class="fig">
<svg viewBox="0 0 680 280" role="img" aria-label="V4 three-level memory">
  <defs>
    <marker id="arr2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--fg)"></path>
    </marker>
  </defs>
  <text x="14" y="20" fill="var(--fg)" font-size="13" font-weight="600">V4 three-level memory (older = coarser, nearer = finer)</text>

  <text x="14" y="58" fill="var(--accent)" font-size="13" font-weight="600">SWA short</text>
  <g fill="var(--accent)">
    <rect x="120" y="44" width="13" height="13"></rect>
    <rect x="137" y="44" width="13" height="13"></rect>
    <rect x="154" y="44" width="13" height="13"></rect>
    <rect x="171" y="44" width="13" height="13"></rect>
    <rect x="188" y="44" width="13" height="13"></rect>
    <rect x="205" y="44" width="13" height="13"></rect>
    <rect x="222" y="44" width="13" height="13"></rect>
    <rect x="239" y="44" width="13" height="13"></rect>
    <rect x="256" y="44" width="13" height="13"></rect>
  </g>
  <text x="290" y="56" fill="var(--muted)" font-size="11">sliding window n_win = 128 (fine-grained local KV)</text>

  <text x="14" y="118" fill="var(--accent)" font-size="13" font-weight="600">CSA mid</text>
  <g fill="var(--accent)" opacity="0.6">
    <rect x="120" y="104" width="18" height="18"></rect>
    <rect x="144" y="104" width="18" height="18"></rect>
    <rect x="168" y="104" width="18" height="18"></rect>
    <rect x="192" y="104" width="18" height="18"></rect>
  </g>
  <path d="M220 113 H250" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr2)"></path>
  <rect x="258" y="104" width="22" height="18" rx="4" fill="var(--accent)"></rect>
  <text x="296" y="118" fill="var(--muted)" font-size="11">4→1 compress + Lightning Indexer top-k = 1024</text>

  <text x="14" y="178" fill="var(--accent)" font-size="13" font-weight="600">HCA long</text>
  <rect x="120" y="162" width="120" height="24" rx="4" fill="var(--accent)" opacity="0.25"></rect>
  <text x="180" y="178" text-anchor="middle" fill="var(--muted)" font-size="11">128 chunks</text>
  <path d="M248 174 H278" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr2)"></path>
  <rect x="286" y="162" width="26" height="24" rx="4" fill="var(--accent)"></rect>
  <text x="324" y="179" fill="var(--fg)" font-size="11">128→1 dense (1M→~7800)</text>

  <text x="14" y="250" fill="var(--muted)" font-size="11">Three levels stacked: 1M tokens end up as only ~7800 memory units (≈ 0.8% of original KV)</text>
</svg>
<figcaption>Fig.2　V4 three-level memory: short-term SWA (window n_win=128) → mid-term CSA (4 chunks→1 + Lightning Indexer top-k=1024) → long-term HCA (128→1 dense, 1M context compressed to ~7800).</figcaption>
</figure>

- **SWA (Sliding Window Attention, short)**: looks at only the most recent `n_win=128` tokens' fine-grained KV, covering local dependencies.
- **CSA (Compressed Sliding Attention, mid)**: compresses every 4 chunks into 1 compressed block, then uses Lightning Indexer to pick the `top-k=1024` most critical positions from the full long context → mid-term memory is both compressed and "selective".
- **HCA (Heavy-Cluster Attention, long)**: densely compresses 128 blocks into 1 "long-term memory head"; a 1M-token context ends up keeping only ~**7800** memory units (≈ 0.8% of original KV).

> The trick of three-level memory: the older/farther the info, the coarser; the nearer, the finer. Near needs precision, far only needs semantics — exactly a model of human memory, and it makes 1M-context inference cost controllable.

### 2.4 Two Product Lines: V4-Pro vs V4-Flash

| Metric (vs dense baseline) | V4-Pro | V4-Flash |
|---|---|---|
| FLOPs | ≈ 27% | ≈ 10% |
| KV memory | ≈ 10% | ≈ 7% |

V4-Flash pushes FLOPs to ~1/10 and KV to ~1/14 — the version for "extreme throughput / edge deployment"; V4-Pro takes the more balanced point between quality and cost.

### 2.5 Inference Side: Mooncake & DistServe

Compression solves "fits on one card"; but **high-concurrency long context** still needs PD disaggregation.

- **Mooncake (KV-Cache-centric P/D disaggregation)**: pull KV Cache into a shared pool; Prefill instances compute KV and store it in the pool, Decode instances fetch on demand — memory is shared across instances, never recomputed.

<figure class="fig">
<svg viewBox="0 0 680 250" role="img" aria-label="Mooncake P/D disaggregation">
  <defs>
    <marker id="arr3" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--fg)"></path>
    </marker>
  </defs>
  <rect x="30" y="40" width="120" height="40" rx="6" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="90" y="65" text-anchor="middle" fill="var(--fg)" font-size="13">Prefill ×N</text>
  <rect x="30" y="90" width="120" height="40" rx="6" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="90" y="115" text-anchor="middle" fill="var(--fg)" font-size="13">Prefill</text>

  <rect x="250" y="55" width="180" height="120" rx="10" fill="var(--accent)" opacity="0.12" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="340" y="100" text-anchor="middle" fill="var(--fg)" font-size="13" font-weight="600">KVCache pool</text>
  <text x="340" y="122" text-anchor="middle" fill="var(--muted)" font-size="11">KV computed once</text>
  <text x="340" y="140" text-anchor="middle" fill="var(--muted)" font-size="11">shared across instances</text>

  <rect x="530" y="40" width="120" height="40" rx="6" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="590" y="65" text-anchor="middle" fill="var(--fg)" font-size="13">Decode ×M</text>
  <rect x="530" y="90" width="120" height="40" rx="6" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="590" y="115" text-anchor="middle" fill="var(--fg)" font-size="13">Decode</text>

  <path d="M150 60 H248" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr3)"></path>
  <path d="M432 100 H528" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr3)"></path>
  <text x="340" y="200" text-anchor="middle" fill="var(--muted)" font-size="11">KV-Cache-centric P/D separation: memory shared across instances, never recomputed</text>
</svg>
<figcaption>Fig.3　Mooncake: KVCache-pool-centric Prefill/Decode separation; KV computed once, reused everywhere.</figcaption>
</figure>

- **DistServe (Goodput, OSDI'24)**: split Prefill and Decode into **different resource pools**, each optimized for its own SLO (TTFT / TPOT), measured by "Goodput" (useful tokens that meet the SLO) rather than raw throughput — avoiding the pitfall of "sacrificing first-token latency for high throughput".

<figure class="fig">
<svg viewBox="0 0 680 220" role="img" aria-label="DistServe">
  <defs>
    <marker id="arr4" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--fg)"></path>
    </marker>
  </defs>
  <rect x="40" y="50" width="200" height="80" rx="8" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="140" y="85" text-anchor="middle" fill="var(--fg)" font-size="13" font-weight="600">Prefill pool</text>
  <text x="140" y="108" text-anchor="middle" fill="var(--muted)" font-size="11">optimize TTFT</text>

  <rect x="440" y="50" width="200" height="80" rx="8" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="540" y="85" text-anchor="middle" fill="var(--fg)" font-size="13" font-weight="600">Decode pool</text>
  <text x="540" y="108" text-anchor="middle" fill="var(--muted)" font-size="11">optimize TPOT</text>

  <path d="M240 90 H438" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr4)"></path>
  <text x="340" y="82" text-anchor="middle" fill="var(--muted)" font-size="11">KV handoff</text>

  <rect x="200" y="160" width="280" height="34" rx="6" fill="var(--accent)" opacity="0.12"></rect>
  <text x="340" y="182" text-anchor="middle" fill="var(--fg)" font-size="12">Goal: Goodput = useful tokens meeting the SLO</text>
</svg>
<figcaption>Fig.4　DistServe: Prefill pool and Decode pool separated, each optimizing TTFT vs TPOT, with Goodput (useful tokens meeting SLO) as the objective.</figcaption>
</figure>

> End-to-end cost = compression (MLA/NSA/CSA+HCA cut memory + FLOPs) × scheduling (Mooncake/DistServe raise concurrency). DeepSeek V3's full-disaggregation stack at ~545 output tok/s/GPU is exactly the product of the two — **also the technical substrate of the "inference cost-down → cloud-inference vendors / lower on-device VLA deployment bar" investment thesis.**

## 3. Efficient Training & Inference Systems

- **Mixed-precision training (FP16/BF16/FP8 Hybrid)**: only GEMM inputs use low precision (half memory), master weights stay FP32. FP16 needs Gradient Scaling; BF16 shares FP32's exponent (no scaling); FP8 Hybrid = E4M3 forward + E5M2 backward (H100+). DeepL 172B throughput 400→550 TFLOP/s (1.4×).
- **KV Cache quantization & eviction**: quantization compresses precision (2~4×), eviction cuts token count (3~244×). vLLM/SGLang default FP8 KV (0.3% loss); DeepSeek V4 MLA+FP8 dual compression needs only ~20GB for 1M context; RDKV (2026) 2.48% cache → 97.81% accuracy; ThinKV (ICLR 2026 Oral) <5% KV near-lossless, 5.8× throughput.
- **PagedAttention**: manages KV Cache in 16-token blocks, lifting memory utilization 20~40%→95%+, 2~4× concurrency. vLLM (SOSP 2023), SGLang; also the substrate for "multi-robot concurrent VLA inference".
- **PD Disaggregation (Prefill-Decode Disaggregation)**: split Prefill (compute-bound) from Decode (memory-bound) across instances, cutting TTFT. vLLM/SGLang/Mooncake/DistServe; DeepSeek V3 full-disaggregation stack ~545 output tok/s/GPU; **pays off only with long prompt + long output + high concurrency.**
- **Speculative Decoding (EAGLE / MTP)**: draft-verify loop — draft guesses K tokens, main model verifies in one forward pass, strictly preserving the output distribution (lossless). DeepSeek V3 H200 batch=1 40→60 tok/s (1.8×); EAGLE-3 on Llama-3 8B up to 6.5×. **Pitfall: monitor acceptance rate (< 30% should disable); INT4 main + FP16 draft precision mismatch crashes it.**
- **SwiGLU (Gated FFN)**: dual branch `SiLU(xW_gate) ⊙ (xW_val) · W_down` adds element-wise gating. Parity params with ReLU FFN, but `d_ff` must become `(8/3)d`. In Llama/Qwen/DeepSeek V4/GLM-5/MiniMax/Gemma4/Mistral.

> Inference cost = memory (GQA/MLA + quantized KV) + IO (FlashAttention) + scheduling (PagedAttention + PD disaggregation) + decode efficiency (speculative decoding), four layers stacked. Check whether the inference stack has "all four layers on".

## 4. On-Device Intelligence: Big Models on Small Resources

- **LoRA / QLoRA (PEFT)**: freeze `W₀`, train only low-rank increment `ΔW = B×A` (r≪d), cutting fine-tuning cost 50~100×. QLoRA adds 4-bit NF4 + double quantization so 65B fits on one card; community fine-tuned MiniMax-8B via QLoRA for < $20. LLaVA itself uses LoRA (only 0.5% params); Qwen3.5 recommends `r=64 + DoRA`. Turns "big-model customization" from a rich-kid game into a平民 (civilian) skill.
- **On-Device Deep Research at 4B (arXiv:2607.12257)**: **a 4B model can do "deep research"** — using "exposed boundaries" (faithfulness) to curb hallucination and "retrieval-coverage boundaries" to curb omission. Means **on-device / local research assistants with retrieval** are possible, boosting the on-device-chip / edge-compute narrative.
- **EcoSpec: cost-aware speculative decoding for MoE (arXiv:2607.12696)**: in MoE, **expert scattering** brings hidden activation cost; factoring activation cost into the draft stage and reducing expert movement → faster without quality loss. Benefits sparse-MoE inference like DeepSeek-V3/V4.
- **Test-Time Compute**: run a "good-enough model" locally (4B research, on-device VLA) and use multi-step verification / self-evolving verifiers to compensate for fewer parameters. Saves cloud cost, preserves privacy, cuts latency — **exactly the route robot on-device real-time inference relies on.**

## 5. Frontier Inference & Reinforcement Learning (Summer 2026)

- **OAT: tracing Agent failures from "streams of success" (arXiv:2607.12747)**: uses Neural CDE to learn from successful trajectories "which step failed", spotting trajectory drift earlier than behavioral monitoring. Transferable to "robot manipulation failure localization" — boosting autonomous-system debuggability.
- **Ring-Zero: Zero-RL pushed to 1T (Renmin Univ. × Ant, 2607.12395)**: pushing Zero-RL (no SFT cold-start, learn reasoning directly from RL) to 1T params makes **advanced thinking strategies emerge spontaneously**, weakening dependence on massive SFT data.
- **SPS: State-Prediction Separation (Cornell × Harvard, 2607.01218)**: decoupling representation from prediction training yields 2.6× efficiency, easier reuse and scaling — an engineering idea for efficient world models / oracles.
- **HiLS-Attention: 8K training extrapolates to 4M (Tencent Hunyuan, 2607.02980)**: an attention improvement lets 8K-training extrapolate to 4M, 13.5× prefill speedup, open-sourced. The "train short, use long" cost-effective route for long context.
- **Exploration-Paradox RL (ByteDance Seed × MSU, 2607.06987)**: corrects importance-sampling bias in exploration, easing entropy-collapse / under-exploration in long-horizon reasoning RL.
- **AMVL: latent-space continuous reasoning (SJTU × Ant, 2607.00461)**: continuous reasoning in latent space, BLINK benchmark +10.83, echoing the "diffusion / continuous representation for reasoning" route.

> Together these point to: **reasoning ability is shifting from "stacking parameters" to "better training signals + better long-context / latent-space representations".**

## 6. Series Closure

```
(1) Foundations & Alignment  → multimodality = extrapolating the generation space from text to image/action
(2) VLM Evolution            → ViT/CLIP/LLaVA: the standard "see + speak" paradigm
(3) Generative Models        → diffusion/GAN: the "draw" branch
(4) VLA + World Model        → RT-2→π0.7→HiF-VLA: act + think-then-act
(5) Efficiency + Frontier    → attention/inference opt/on-device/frontier RL: the run-fast, run-on-device substrate
```

**Investment-angle closure**: LLM cost reduction (this part) ↔ embodied real-time feasibility (Part 4) ↔ on-device chips / edge compute (this part's on-device) are three pivots on the same logical chain. To evaluate any multimodal / embodied company, score it item-by-item along this "foundation paradigm → model → generation → action → efficiency" axis.
