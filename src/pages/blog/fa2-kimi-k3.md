---
title: 前沿架构解码手记（二）：Kimi K3——注意力与 MoE 的双重进化
description: 顺着总览的路线图，第一家拆解 Kimi K3。它不靠堆算力，而是把注意力机制"重设计"了一遍：Kimi Delta Attention（KDA）在 MLA 上加差分项，Gated MLA 按 3:1 混合，Attention Residuals 稳住训练；MoE 用 Stable LatentMoE（896 路由 / 激活 16）；视觉端用从零训练的 MoonViT-V2 做原生多模态。官方称相对 K2.5 缩放效率约 2.5×。
pubDate: 2026-08-06
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa2-kimi-k3
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么从 Kimi K3 讲起

在总览里我们说过，Kimi K3 选的是"重设计注意力"这条最温和、也最好懂的路。它相比 M2.7→M3 的"稀疏化"、DeepSeek V4 的"压缩"，改动更靠近现有架构，适合作为循序渐进的**第一站**。

## 1. 注意力：在 MLA 上加一个"差分项"

Kimi K3 的注意力核心是 **Kimi Delta Attention（KDA）**，建立在 MLA（Multi-head Latent Attention）之上：

- **MLA 基线**：把 Key/Value 压成低维潜向量，大幅省 KV Cache——这是 DeepSeek 带火、现在被广泛采用的做法。
- **KDA 的增量**：在 MLA 之上叠加一个**差分项（delta）**，让注意力对"当前 query 与历史位置的相对变化"更敏感，缓解长上下文下标准注意力"远端信息被近端淹没"的问题。
- **Gated MLA（3:1）**：把"标准注意力路径"与"MLA 潜路径"按 **3:1** 门控混合——既保留 MLA 的 KV 省显存优势，又用标准路径补回一部分长程建模能力。

配合 **Attention Residuals（注意力残差）**：把上一层注意力的输出作为残差直接并入当前层，缓解深层网络里注意力信号衰减，让 1M 上下文训练更稳。

<div class="keybox">
<strong>要点：</strong>KDA 不是推翻 MLA，而是在 MLA 上加"差分 + 门控(3:1) + 残差"三件套，目标是<strong>不增加太多 KV 的前提下，把长上下文建模做扎实</strong>。
</div>

## 2. MoE：Stable LatentMoE

Kimi K3 是 2.8T 参数的 MoE，关键在"稳"：

- **896 个路由专家，每 token 激活 16 个**——专家数多、激活少，容量大但单步算力可控。
- **SiTU-GLU 激活**：专家内部用 Situ 化的 GLU，兼顾表达力与数值稳定。
- **Quantile Balancing（分位数均衡）**：用分位数统计做负载均衡，替代传统的辅助 loss，缓解"专家旱涝不均"导致的训练抖动——这就是 "Stable" 的由来。

## 3. 视觉：MoonViT-V2 原生多模态

Kimi K3 的视觉不是"外接一个 CLIP"，而是 **MoonViT-V2**——一个**从零训练、以 next-token-prediction 为目标的视觉编码器**：

- 与语言模型共享同一套生成目标，图像 token 与文本 token 在同一空间里被统一预测；
- 因此"看"和"说"是原生融合，而非后期拼接，长上下文里图文交错更顺。

## 4. 基础设施：MoonEP / FlashKDA / AgentEnv

- **MoonEP**：专家并行（Expert Parallelism）通信优化，支撑 896 专家的大规模 MoE 训练/推理。
- **FlashKDA**：把 KDA 算子写成 Flash 风格融合核，降低显存带宽压力。
- **AgentEnv**：面向 Agent / 工具调用的工作负载优化，贴合 1M 上下文下的长程任务。

## 5. 效果与定位

官方口径：相对 K2.5，**缩放效率约 2.5×**——即同样算力/数据下，能力爬升更快；2.8T MoE + 1M 上下文 + 原生视觉，定位为"能读、能看、能长程推理"的通用前沿模型。

> 资料说明：本文核心架构点来自 Moonshot AI 官方技术报告（PDF）及官方微信公众号发布的解读；技术报告 PDF 在本站环境内未做机器解析，细节以官方发布为准。

## 6. 投资视角

- "重设计注意力 + 稳定 MoE + 原生视觉"代表了一条**不依赖极端硬件**的升级路径：同等能力下更省算力，利好**推理成本下降**与**国产算力适配**。
- 2.8T MoE 开源，意味着"前沿多模态能力"进一步商品化，应用层（Agent、具身、长文档/代码理解）的**壁垒从模型转向数据与场景**。
- 下一章看 MiniMax：它走的是和 KDA 不同的另一条路——直接把注意力**稀疏化**。
