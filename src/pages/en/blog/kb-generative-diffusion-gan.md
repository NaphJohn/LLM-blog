---
title: "Generative Models: Diffusion vs GAN—Stable Diffusion and the Adversarial Route"
description: The two pillars of image generation—Stable Diffusion (Latent Diffusion, LDM) moves diffusion into the VAE-compressed latent space; GANs use a minimax game between generator and discriminator to bypass explicit probability modeling. Both deeply shape today's visual generation and alignment.
pubDate: 2026-07-31
series: AI Knowledge Base
lang: en
altLang: zh
altHref: /blog/kb-generative-diffusion-gan
layout: ../../../layouts/BlogPost.astro
---

## 1. Two routes to the same goal: generate realistic samples from noise / random vectors

- **GAN**: directly learns a mapping `G: z → x`, using adversarial training to push the generated distribution toward the real one.
- **Diffusion (DDPM)**: iteratively adds noise to data then learns to "denoise", reconstructing samples from pure noise.

Different in spirit, yet both are the foundation of today's industrial-grade models (SDXL / FLUX / StyleGAN / video generation).

## 2. Stable Diffusion (LDM, CVPR 2022)

- **Core insight**: move diffusion from pixel space (3×512² ≈ 780K dims) into the **latent space** after VAE compression (4×64² ≈ 16K dims, ~48×↓), slashing compute with almost no quality loss.
- **The trio**:
  - **VAE**: perceptual compression (pixel ↔ latent);
  - **U-Net**: denoising backbone, injecting text via **Cross-Attention**;
  - **DDPM/DDIM + CFG**: sampling; Classifier-Free Guidance controls "prompt adherence".
- **Key formulas**:
  - Training loss: `L_LDM = E[‖ε − ε_θ(z_t, t, τ(c))‖²]`
  - CFG prediction: `ε̂ = ε_u + w·(ε_c − ε_u)` (larger w → more on-prompt, less diversity)
- **Impact**: SDXL / SD3 / HunyuanDiT / Wan-Video / FLUX all reuse this template; the entire LoRA / ControlNet / IP-Adapter fine-tune chain stands on LDM's shoulders.

## 3. GAN (Generative Adversarial Nets, Goodfellow 2014)

- **Core formula (minimax)**: `min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]`
- **Key insight**: when D is optimal, G is equivalent to minimizing JS divergence (p_g → p_data), bypassing explicit complex density modeling.
- **Training tricks**: use `−log D(G(z))` for G to avoid vanishing gradients; Adam `β1=0.5`; Label Smoothing for stability.
- **Common pitfalls**:
  - **Mode Collapse** → WGAN / WGAN-GP use Wasserstein distance;
  - **Training oscillation** → SN-GAN (spectral normalization).
- **Impact**: 75K+ citations; spawned DCGAN / StyleGAN / CycleGAN / pix2pix; more importantly, the **adversarial idea permeates RLHF and safety alignment** (using a discriminator/reward model for preference alignment).

## 4. Practical takeaways

- For visual-gen product work: diffusion (controllable, easy to fine-tune, mature ecosystem) is the current mainstream; GANs still hold a niche in "single-image high-fidelity / real-time".
- Watch "diffusion + adversarial" hybrids (adversarial loss assisting diffusion, diffusion as GAN regularizer)—the frontier of quality/speed trade-offs.
- Alignment (RLHF) is essentially a variant of "GAN-style adversarial training"—understanding GAN mode collapse helps understand "reward hacking" in alignment.
