---
title: 高效训练与推理系统：混合精度 / KV Cache / PagedAttention / PD 分离 / 投机解码 / SwiGLU
description: 把"够大的模型"跑得起、跑得快的六块基石——混合精度训练、KV Cache 量化与驱逐、PagedAttention、PD 分离、投机解码（草稿-验证）、SwiGLU。一篇讲清每块解决什么瓶颈、代表工作与关键坑点。
pubDate: 2026-07-31
series: AI 知识库
lang: zh
altLang: en
altHref: /en/blog/kb-efficient-training-inference
layout: ../../layouts/BlogPost.astro
---

## 1. 混合精度训练（FP16 / BF16 / FP8 Hybrid）

- **机制**：仅 GEMM 输入用低精度（显存减半），master weights 保持 FP32；三条路线：
  - **FP16 AMP**：需 Gradient Scaling 防下溢；
  - **BF16 AMP**：指数位同 FP32，无需 scaling；
  - **FP8 Hybrid**：E4M3 前向 + E5M2 反向（H100+）。
- **实测**：DeepL 172B 吞吐 400→550 TFLOP/s（1.4×）；LLM-jp 172B 前 7000 步 BF16、后切 FP8，loss 不崩。
- **坑**：FP16 必须在 `scaler.step` 前 `unscale_()`；BF16 学习率需调高 10~20%。

## 2. KV Cache 量化与驱逐

- **定位**：KV Cache 是长上下文推理的**主要显存瓶颈**——量化压缩存储精度（2~4×），驱逐削减缓存 token 数（3~244×）。
- **代表工作**：vLLM/SGLang 默认 FP8 KV（0.3% 损失）；DeepSeek V4 MLA+FP8 双重压缩，1M 上下文仅需 ~20GB；RDKV（2026）率失真统一量化+驱逐，2.48% 缓存 → 97.81% 准确率；ThinKV（ICLR 2026 Oral）<5% KV 近乎无损、5.8× 吞吐提升。
- **收益**：FP8 2× 压缩 0.3% 损失、INT4 4× 0.5% 损失；叠加淘汰 50%+INT4 → 总压缩比 8×。

## 3. PagedAttention

- **机制**：把 KV Cache 按 16-token block **分页管理**，解决 LLM 服务中的内存碎片与过度预留。
- **收益**：内存利用率 20~40% → 95%+，同等硬件并发提升 2~4×。
- **代表**：vLLM（SOSP 2023）、SGLang；也是"多机器人并发 VLA 推理"的底层支撑。

## 4. PD 分离（Prefill-Decode Disaggregation）

- **机制**：Prefill（算力密集）与 Decode（访存密集）拆到不同实例，互不拖累、降 TTFT。
- **代表**：vLLM / SGLang / Mooncake / DistServe；KV 通过 NIXL/Mooncake 传输。
- **收益**：DeepSeek V3 全量分离栈 ~545 output tok/s/GPU；**仅在长 prompt + 长输出 + 高并发时回本**。
- **具身关联**：支撑多机器人并发推理的端云协同基础设施。

## 5. 投机解码（Speculative Decoding，EAGLE / MTP）

- **机制**：草稿-验证循环——draft 模型先猜 K 个 token，主模型一次前向验证，顺序接受与分布一致的 prefix。
- **lossless 性质**：`p_main(x) == p_draft(x) + 修正项` ⇒ 严格不改变输出分布。
- **典型收益**：DeepSeek V3 H200 batch=1 40→60 tok/s（1.8×）；EAGLE-3 在 Llama-3 8B 最高 6.5×。
- **集成**：SGLang 首发 EAGLE-3、vLLM v0.25 支持 NEXTN/MTP、阶跃 JetSpec、AMD ROCm 1.25~2.11×。
- **坑**：接受率必须监控（< 30% 应关掉）；INT4 主 + FP16 草稿精度不匹配会暴跌。

## 6. SwiGLU（门控 FFN）

- **机制**：双支路 `SiLU(xW_gate) ⊙ (xW_val) · W_down`，引入逐元素乘法门控，让网络自适应"通过/抑制"信息。
- **关键点**：参数量持平 ReLU FFN，但 `d_ff` 须改为 `(8/3)d`（自实现易踩坑）。
- **在用**：Llama 全系、Qwen 全系、DeepSeek V4（+Clamping [-10,10]）、GLM-5、MiniMax、Gemma 4、Mistral。

## 7. 实践启示

- 推理成本 = 显存（GQA/MLA + 量化 KV）+ IO（FlashAttention）+ 调度（PagedAttention + PD 分离）+ 解码效率（投机解码）四层叠加。
- 评估推理栈时，看是否"四层全开"；缺任何一层都会在长上下文/高并发场景下显著掉队。
