---
title: "Efficient Training & Inference Systems: Mixed Precision / KV Cache / PagedAttention / PD Disaggregation / Speculative Decoding / SwiGLU"
description: The six building blocks that make "big enough models" affordable and fast—mixed-precision training, KV Cache quantization & eviction, PagedAttention, PD disaggregation, speculative decoding (draft-verify), SwiGLU. One post on what bottleneck each solves, representative work, and key pitfalls.
pubDate: 2026-07-31
series: AI Knowledge Base
lang: en
altLang: zh
altHref: /blog/kb-efficient-training-inference
layout: ../../../layouts/BlogPost.astro
---

## 1. Mixed-precision training (FP16 / BF16 / FP8 Hybrid)

- **Mechanism**: only GEMM inputs use low precision (half memory), master weights stay FP32; three routes:
  - **FP16 AMP**: needs Gradient Scaling to avoid underflow;
  - **BF16 AMP**: exponent bits match FP32, no scaling needed;
  - **FP8 Hybrid**: E4M3 forward + E5M2 backward (H100+).
- **Measured**: DeepL 172B throughput 400→550 TFLOP/s (1.4×); LLM-jp 172B first 7000 steps BF16 then switched to FP8, loss stable.
- **Pitfalls**: FP16 must `unscale_()` before `scaler.step`; BF16 LR needs +10~20%.

## 2. KV Cache quantization & eviction

- **Role**: KV Cache is the **main memory bottleneck** of long-context inference—quantization compresses storage precision (2~4×), eviction cuts cached tokens (3~244×).
- **Representative**: vLLM/SGLang default FP8 KV (0.3% loss); DeepSeek V4 MLA+FP8 double compression needs only ~20GB for 1M context; RDKV (2026) unifies quantization+eviction under rate-distortion, 2.48% cache → 97.81% accuracy; ThinKV (ICLR 2026 Oral) <5% KV, nearly lossless, 5.8× throughput.
- **Gains**: FP8 2× compression 0.3% loss, INT4 4× 0.5% loss; combined eviction 50%+INT4 → 8× total.

## 3. PagedAttention

- **Mechanism**: manage KV Cache in **16-token blocks** (paging), fixing memory fragmentation and over-reservation in LLM serving.
- **Gains**: memory utilization 20~40% → 95%+, 2~4× concurrency on same hardware.
- **Representative**: vLLM (SOSP 2023), SGLang; also underpins "multi-robot concurrent VLA inference".

## 4. PD disaggregation (Prefill-Decode Disaggregation)

- **Mechanism**: split Prefill (compute-bound) and Decode (memory-bound) onto different instances, avoiding mutual drag, lowering TTFT.
- **Representative**: vLLM / SGLang / Mooncake / DistServe; KV transferred via NIXL/Mooncake.
- **Gains**: DeepSeek V3 full disaggregated stack ~545 output tok/s/GPU; **pays off only with long prompt + long output + high concurrency**.
- **Embodied link**: backbone for multi-robot concurrent inference (edge-cloud协同).

## 5. Speculative Decoding (EAGLE / MTP)

- **Mechanism**: draft-verify loop—a draft model guesses K tokens, the main model verifies in one forward pass, accepting the prefix consistent with the distribution.
- **Lossless property**: `p_main(x) == p_draft(x) + correction` ⇒ strictly preserves output distribution.
- **Typical gains**: DeepSeek V3 H200 batch=1 40→60 tok/s (1.8×); EAGLE-3 on Llama-3 8B up to 6.5×.
- **Integration**: SGLang shipped EAGLE-3 first, vLLM v0.25 supports NEXTN/MTP, Step JetSpec, AMD ROCm 1.25~2.11×.
- **Pitfall**: monitor acceptance rate (< 30% should disable); INT4 main + FP16 draft precision mismatch crashes throughput.

## 6. SwiGLU (gated FFN)

- **Mechanism**: two branches `SiLU(xW_gate) ⊙ (xW_val) · W_down`, adding element-wise multiplicative gating so the network adaptively "passes/suppresses" information.
- **Key point**: params match ReLU FFN, but `d_ff` must be `(8/3)d` (easy to miss when self-implementing).
- **Adopters**: all Llama, all Qwen, DeepSeek V4 (+Clamping [-10,10]), GLM-5, MiniMax, Gemma 4, Mistral.

## 7. Practical takeaways

- Inference cost = memory (GQA/MLA + quantized KV) + IO (FlashAttention) + scheduling (PagedAttention + PD disaggregation) + decode efficiency (speculative decoding), four layers stacked.
- When evaluating an inference stack, check whether "all four layers are on"; missing any one drags performance under long-context / high-concurrency.
