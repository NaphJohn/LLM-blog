---
title: 生成模型：扩散与对抗——Stable Diffusion 与 GAN 两条路线
description: 图像生成的两大支柱：Stable Diffusion（潜空间扩散 LDM）把扩散搬到 VAE 压缩后的潜空间；GAN 用生成器与判别器的零和博弈绕开显式概率建模。二者都深刻影响今天的视觉生成与对齐技术。
pubDate: 2026-07-31
series: AI 知识库
lang: zh
altLang: en
altHref: /en/blog/kb-generative-diffusion-gan
layout: ../../layouts/BlogPost.astro
---

## 1. 两条路线解决同一件事：从噪声/随机向量生成逼真样本

- **GAN**：直接学一个映射 `G: z → x`，靠对抗训练让生成分布逼近真实分布。
- **扩散（Diffusion / DDPM）**：逐步往数据加噪再学"去噪"，从纯噪声迭代还原出样本。

二者思路迥异，但都已成为今天 SDXL / FLUX / StyleGAN / 视频生成等工业级模型的底层支柱。

## 2. Stable Diffusion（LDM, CVPR 2022）

- **核心洞察**：把扩散从像素空间（3×512² ≈ 78 万维）搬到 VAE 压缩后的**潜空间**（4×64² ≈ 1.6 万维，约 48×↓），计算量大幅下降而质量几乎不变。
- **三件套**：
  - **VAE**：感知压缩（像素 ↔ 潜变量）；
  - **U-Net**：去噪主干，靠 **Cross-Attention** 注入文本条件；
  - **DDPM/DDIM + CFG**：采样策略，Classifier-Free Guidance 控制"听话程度"。
- **关键公式**：
  - 训练损失：`L_LDM = E[‖ε − ε_θ(z_t, t, τ(c))‖²]`
  - CFG 预测：`ε̂ = ε_u + w·(ε_c − ε_u)`（w 越大越贴合 Prompt，但多样性下降）
- **影响**：SDXL / SD3 / HunyuanDiT / Wan-Video / FLUX 全沿用此模板；LoRA / ControlNet / IP-Adapter 整条下游微调链都站在 LDM 肩膀上。

## 3. GAN（Generative Adversarial Nets, Goodfellow 2014）

- **核心公式（minimax）**：
  `min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]`
- **关键洞察**：当 D 最优时，G 等价于最小化 JS 散度（p_g → p_data），从而绕开显式建模复杂概率分布。
- **训练技巧**：G 损失改用 `−log D(G(z))` 防梯度消失；Adam `β1=0.5`；Label Smoothing 稳定训练。
- **常见坑**：
  - **Mode Collapse**（模式崩溃）→ WGAN / WGAN-GP 用 Wasserstein 距离缓解；
  - **训练震荡** → SN-GAN（谱归一化）稳定。
- **影响**：被引 75000+ 次，衍生 DCGAN / StyleGAN / CycleGAN / pix2pix；更重要的是**对抗思想渗透进 RLHF 与安全对齐**（用判别器/奖励模型做偏好对齐）。

## 4. 实践启示

- 做视觉生成产品：扩散系（可控、易微调、生态成熟）是当前主流；GAN 系在"单图高保真/实时"场景仍有一席之地。
- 关注"扩散 + 对抗"的融合（对抗损失辅助扩散、扩散做 GAN 正则等），这是质量/速度权衡的前沿。
- 对齐（RLHF）本质是"GAN 式对抗训练"的变体——理解 GAN 的 Mode Collapse 有助于理解对齐中的"奖励黑客"。
