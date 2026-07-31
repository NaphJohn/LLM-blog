---
title: 多模态解码手记（三）：生成模型——扩散与对抗（Stable Diffusion & GAN）
description: 多模态不止"理解"，更要"生成"。本章拆解图像生成的两大支柱：Stable Diffusion 把扩散搬到 VAE 压缩后的潜空间（LDM）；GAN 用生成器与判别器的零和博弈绕开显式概率建模。二者都深刻影响今天的视觉生成与对齐技术。
pubDate: 2026-07-31
series: 多模态解码手记
lang: zh
altLang: en
altHref: /en/blog/mm3-generation
layout: ../../layouts/BlogPost.astro
---

## 1. 两条路线解决同一件事

> 从噪声 / 随机向量，生成逼真样本。

- **GAN**：直接学一个映射 `G: z → x`，靠对抗训练让生成分布逼近真实分布。
- **扩散（Diffusion / DDPM）**：逐步往数据加噪再学"去噪"，从纯噪声迭代还原出样本。

二者思路迥异，但都已成为今天 SDXL / FLUX / StyleGAN / 视频生成等工业级模型的底层支柱。

## 2. Stable Diffusion（LDM, CVPR 2022）

- **核心洞察**：把扩散从像素空间（3×512² ≈ 78 万维）搬到 VAE 压缩后的**潜空间**（4×64² ≈ 1.6 万维，约 48×↓），计算量大幅下降而质量几乎不变。
- **三件套**：
  - **VAE**：感知压缩（像素 ↔ 潜变量）；
  - **U-Net**：去噪主干，靠 **Cross-Attention** 注入文本条件；
  - **DDPM / DDIM + CFG**：采样策略，Classifier-Free Guidance 控制"听话程度"。
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
- **影响**：被引 75000+ 次，衍生 DCGAN / StyleGAN / CycleGAN / pix2pix；**更重要的是"对抗"思想渗透进 RLHF 与安全对齐**——用判别器 / 奖励模型做偏好对齐。

## 4. 为什么生成模型是多模态的关键一环

- **多模态生成闭环**：VLM（二章）负责"理解"，扩散 / GAN（本章）负责"画出来"——文生图、文生视频、图像编辑都是"理解→生成"的下游。
- **与对齐的暗线**：RLHF 本质是"GAN 式对抗训练"的变体（奖励模型 ≈ 判别器）。理解 GAN 的 **Mode Collapse**，有助于理解对齐中的"奖励黑客（reward hacking）"——模型为了骗过奖励模型而退化到单一模式。
- **扩散 + 对抗融合**：用对抗损失辅助扩散、扩散做 GAN 正则，是质量 / 速度权衡的前沿方向。

## 5. 实践启示

- 做视觉生成产品：扩散系（可控、易微调、生态成熟）是当前主流；GAN 系在"单图高保真 / 实时"场景仍有一席之地。
- 评估多模态公司时，关注其生成底座是自研还是套壳 SD；自研潜空间 + 控制网（ControlNet 类）能力，是图像 / 视频生成赛道的护城河之一。
- 下一章把"理解 + 生成"接到**动作**上：当模型不仅能看能说能画，还能把这一切转成机器人动作，就进入了 VLA 与世界模型。
