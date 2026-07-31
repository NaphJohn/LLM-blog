---
title: 'Multimodal Decoding Notes (3): Generative Models — Diffusion vs GAN (Stable Diffusion & GAN)'
description: 'Multimodality is not only "understanding" but also "generating". This chapter dissects the two pillars of image generation: Stable Diffusion moves diffusion into the VAE-compressed latent space (LDM); GAN uses a zero-sum game between generator and discriminator to bypass explicit probabilistic modeling. Both deeply influence today''s visual generation and alignment techniques.'
pubDate: 2026-07-31
series: Multimodal Decoding Notes
lang: en
altLang: zh
altHref: /blog/mm3-generation
layout: ../../../layouts/BlogPost.astro
---

## 1. Two Routes to the Same Goal

> Generate realistic samples from noise / a random vector.

- **GAN**: directly learn a mapping `G: z → x`, using adversarial training to make the generated distribution approach the real one.
- **Diffusion (Diffusion / DDPM)**: progressively add noise to data then learn to "denoise", iteratively recovering a sample from pure noise.

Different in spirit, but both are now the substrate of industrial models like SDXL / FLUX / StyleGAN / video generation.

## 2. Stable Diffusion (LDM, CVPR 2022)

- **Core insight**: move diffusion from pixel space (3×512² ≈ 780K-dim) into the VAE-compressed **latent space** (4×64² ≈ 16K-dim, ~48×↓), slashing compute with almost no quality loss.
- **The trio**:
  - **VAE**: perceptual compression (pixel ↔ latent);
  - **U-Net**: denoising backbone, injecting text conditions via **Cross-Attention**;
  - **DDPM / DDIM + CFG**: sampling strategy; Classifier-Free Guidance controls "how obedient".
- **Key formulas**:
  - Training loss: `L_LDM = E[‖ε − ε_θ(z_t, t, τ(c))‖²]`
  - CFG prediction: `ε̂ = ε_u + w·(ε_c − ε_u)` (larger w = more prompt-faithful but less diverse)
- **Impact**: SDXL / SD3 / HunyuanDiT / Wan-Video / FLUX all follow this template; the entire downstream fine-tuning chain (LoRA / ControlNet / IP-Adapter) stands on LDM's shoulders.

## 3. GAN (Generative Adversarial Nets, Goodfellow 2014)

- **Core formula (minimax)**:
  `min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]`
- **Key insight**: when D is optimal, G is equivalent to minimizing JS divergence (p_g → p_data), thus bypassing explicit modeling of a complex probability distribution.
- **Training tricks**: use `−log D(G(z))` for G to prevent vanishing gradients; Adam `β1=0.5`; Label Smoothing for stability.
- **Common pitfalls**:
  - *Mode Collapse* → WGAN / WGAN-GP ease it via Wasserstein distance;
  - *Training oscillation* → SN-GAN (spectral normalization) stabilizes.
- **Impact**: 75000+ citations, spawning DCGAN / StyleGAN / CycleGAN / pix2pix; **more importantly, the "adversarial" idea permeated RLHF and safety alignment** — using a discriminator / reward model for preference alignment.

## 4. Why Generative Models Are a Key Multimodal Piece

- **The multimodal generation loop**: VLM (Part 2) handles "understanding", diffusion / GAN (this part) handle "drawing" — text-to-image, text-to-video, image editing are all "understand → generate" downstream.
- **The hidden link to alignment**: RLHF is essentially a variant of "GAN-style adversarial training" (reward model ≈ discriminator). Understanding GAN's **Mode Collapse** helps understand "reward hacking" in alignment — the model degenerates to a single mode to fool the reward model.
- **Diffusion + adversarial fusion**: using adversarial loss to assist diffusion, or diffusion as GAN regularization, is a frontier in the quality / speed trade-off.

## 5. Practical Takeaways

- For visual-generation products: diffusion-based (controllable, easy to fine-tune, mature ecosystem) is the current mainstream; GAN-based still holds a place in "single-image high-fidelity / real-time" scenarios.
- When evaluating multimodal companies, check whether the generative substrate is self-developed or a SD wrapper; self-developed latent space + control-net (ControlNet-class) capability is a moat in the image / video generation track.
- Next chapter connects "understanding + generation" to **action**: when a model can not only see, speak, and draw, but also turn all this into robot actions, we enter VLA and world models.
