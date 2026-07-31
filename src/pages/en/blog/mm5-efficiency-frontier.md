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

## 2. Efficient Training & Inference Systems

- **Mixed-precision training (FP16/BF16/FP8 Hybrid)**: only GEMM inputs use low precision (half memory), master weights stay FP32. FP16 needs Gradient Scaling; BF16 shares FP32's exponent (no scaling); FP8 Hybrid = E4M3 forward + E5M2 backward (H100+). DeepL 172B throughput 400→550 TFLOP/s (1.4×).
- **KV Cache quantization & eviction**: quantization compresses precision (2~4×), eviction cuts token count (3~244×). vLLM/SGLang default FP8 KV (0.3% loss); DeepSeek V4 MLA+FP8 dual compression needs only ~20GB for 1M context; RDKV (2026) 2.48% cache → 97.81% accuracy; ThinKV (ICLR 2026 Oral) <5% KV near-lossless, 5.8× throughput.
- **PagedAttention**: manages KV Cache in 16-token blocks, lifting memory utilization 20~40%→95%+, 2~4× concurrency. vLLM (SOSP 2023), SGLang; also the substrate for "multi-robot concurrent VLA inference".
- **PD Disaggregation (Prefill-Decode Disaggregation)**: split Prefill (compute-bound) from Decode (memory-bound) across instances, cutting TTFT. vLLM/SGLang/Mooncake/DistServe; DeepSeek V3 full-disaggregation stack ~545 output tok/s/GPU; **pays off only with long prompt + long output + high concurrency.**
- **Speculative Decoding (EAGLE / MTP)**: draft-verify loop — draft guesses K tokens, main model verifies in one forward pass, strictly preserving the output distribution (lossless). DeepSeek V3 H200 batch=1 40→60 tok/s (1.8×); EAGLE-3 on Llama-3 8B up to 6.5×. **Pitfall: monitor acceptance rate (< 30% should disable); INT4 main + FP16 draft precision mismatch crashes it.**
- **SwiGLU (Gated FFN)**: dual branch `SiLU(xW_gate) ⊙ (xW_val) · W_down` adds element-wise gating. Parity params with ReLU FFN, but `d_ff` must become `(8/3)d`. In Llama/Qwen/DeepSeek V4/GLM-5/MiniMax/Gemma4/Mistral.

> Inference cost = memory (GQA/MLA + quantized KV) + IO (FlashAttention) + scheduling (PagedAttention + PD disaggregation) + decode efficiency (speculative decoding), four layers stacked. Check whether the inference stack has "all four layers on".

## 3. On-Device Intelligence: Big Models on Small Resources

- **LoRA / QLoRA (PEFT)**: freeze `W₀`, train only low-rank increment `ΔW = B×A` (r≪d), cutting fine-tuning cost 50~100×. QLoRA adds 4-bit NF4 + double quantization so 65B fits on one card; community fine-tuned MiniMax-8B via QLoRA for < $20. LLaVA itself uses LoRA (only 0.5% params); Qwen3.5 recommends `r=64 + DoRA`. Turns "big-model customization" from a rich-kid game into a平民 (civilian) skill.
- **On-Device Deep Research at 4B (arXiv:2607.12257)**: **a 4B model can do "deep research"** — using "exposed boundaries" (faithfulness) to curb hallucination and "retrieval-coverage boundaries" to curb omission. Means **on-device / local research assistants with retrieval** are possible, boosting the on-device-chip / edge-compute narrative.
- **EcoSpec: cost-aware speculative decoding for MoE (arXiv:2607.12696)**: in MoE, **expert scattering** brings hidden activation cost; factoring activation cost into the draft stage and reducing expert movement → faster without quality loss. Benefits sparse-MoE inference like DeepSeek-V3/V4.
- **Test-Time Compute**: run a "good-enough model" locally (4B research, on-device VLA) and use multi-step verification / self-evolving verifiers to compensate for fewer parameters. Saves cloud cost, preserves privacy, cuts latency — **exactly the route robot on-device real-time inference relies on.**

## 4. Frontier Inference & Reinforcement Learning (Summer 2026)

- **OAT: tracing Agent failures from "streams of success" (arXiv:2607.12747)**: uses Neural CDE to learn from successful trajectories "which step failed", spotting trajectory drift earlier than behavioral monitoring. Transferable to "robot manipulation failure localization" — boosting autonomous-system debuggability.
- **Ring-Zero: Zero-RL pushed to 1T (Renmin Univ. × Ant, 2607.12395)**: pushing Zero-RL (no SFT cold-start, learn reasoning directly from RL) to 1T params makes **advanced thinking strategies emerge spontaneously**, weakening dependence on massive SFT data.
- **SPS: State-Prediction Separation (Cornell × Harvard, 2607.01218)**: decoupling representation from prediction training yields 2.6× efficiency, easier reuse and scaling — an engineering idea for efficient world models / oracles.
- **HiLS-Attention: 8K training extrapolates to 4M (Tencent Hunyuan, 2607.02980)**: an attention improvement lets 8K-training extrapolate to 4M, 13.5× prefill speedup, open-sourced. The "train short, use long" cost-effective route for long context.
- **Exploration-Paradox RL (ByteDance Seed × MSU, 2607.06987)**: corrects importance-sampling bias in exploration, easing entropy-collapse / under-exploration in long-horizon reasoning RL.
- **AMVL: latent-space continuous reasoning (SJTU × Ant, 2607.00461)**: continuous reasoning in latent space, BLINK benchmark +10.83, echoing the "diffusion / continuous representation for reasoning" route.

> Together these point to: **reasoning ability is shifting from "stacking parameters" to "better training signals + better long-context / latent-space representations".**

## 5. Series Closure

```
(1) Foundations & Alignment  → multimodality = extrapolating the generation space from text to image/action
(2) VLM Evolution            → ViT/CLIP/LLaVA: the standard "see + speak" paradigm
(3) Generative Models        → diffusion/GAN: the "draw" branch
(4) VLA + World Model        → RT-2→π0.7→HiF-VLA: act + think-then-act
(5) Efficiency + Frontier    → attention/inference opt/on-device/frontier RL: the run-fast, run-on-device substrate
```

**Investment-angle closure**: LLM cost reduction (this part) ↔ embodied real-time feasibility (Part 4) ↔ on-device chips / edge compute (this part's on-device) are three pivots on the same logical chain. To evaluate any multimodal / embodied company, score it item-by-item along this "foundation paradigm → model → generation → action → efficiency" axis.
