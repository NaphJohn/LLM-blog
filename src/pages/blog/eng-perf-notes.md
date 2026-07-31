---
title: 性能优化合集（已讲解话题记录）
description: 每日 AI 论文自动化去重用——已讲解性能优化话题一览。
pubDate: 2026-07-31
series: 大模型工程实践
lang: zh
layout: ../../layouts/BlogPost.astro
---

- PD 分离 (Prefill-Decode Disaggregation)
- Speculative Decoding (EAGLE / MTP, 投机解码)
- KV Cache 量化与驱逐 (KV Cache Quantization & Eviction)
- LoRA / QLoRA (参数高效微调 PEFT)
- 混合精度训练 (Mixed Precision Training, FP16/BF16/FP8 Hybrid)
- PD 分离 (Prefill-Decode Disaggregation)
- PagedAttention (vLLM KV Cache 分页管理, SOSP 2023)
PD分离（Prefill-Decode Disaggregation）
