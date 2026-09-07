---
title: '(9) Wan2.2-I2V/T2V-A14B: The Dual-Expert MoE Video Model and vllm-omni Engine Dissected'
description: 'Wan2.2 dual-expert diffusion MoE architecture, I2V image conditioning, flow matching denoising, timestep-based expert switching, plus TP4 tensor parallelism and the code path of vllm-omni 0.21.0, with an I2V vs T2V comparison table.'
pubDate: 2026-08-11
series: vLLM & SGLang Serving Notes
lang: en
altLang: zh
altHref: /blog/fw9-wan22-vllm-omni
layout: ../../../layouts/BlogPost.astro
---

> **Document info** | Updated: 2026-08-11 | Author: Wang Hangkai | Contents: model architecture / generation principles / vllm-omni code paths / T2V architecture comparison | Version: v0.1
> Companion document: `Wan2.2-I2V-A14B-P800-vllm-omni-Deployment-Test-Report.md` (deployment / performance / precision). All component parameters here come from the actual configs of the weights at `69.75:/home/whk/Wan2.2-I2V-A14B-Diffusers`; all code paths were read from the source at `/workspace/vllm-omni` inside the deployment image `vllm_omni_kunlun:py310_torch2.9_0.21.0`.

## 1. Overview

Wan2.2-A14B is a video generation model from Alibaba's Wan team (official repo). Its core design is a **"dual-expert" diffusion MoE**: two structurally identical 14B DiTs (Diffusion Transformers), one handling the early denoising stage (high-noise segment, responsible for global composition and motion) and one handling the late stage (low-noise segment, responsible for detail refinement), switching at `boundary_ratio` along the denoising timestep — at any moment only one 14B expert is running (hence "A14B" = Active 14B; total parameters about 28B).

I2V (image-to-video) and T2V (text-to-video) share this skeleton and differ only in input conditioning: I2V encodes the reference image through the VAE and uses it as a per-step concatenated condition channel, so the first frame is "pinned" to the input image.

## 2. Wan2.2-I2V-A14B Architecture

> **Model architecture and data flow overview** (reference image to 20-channel condition, noise latent, UMT5-xxl text encoding, dual-expert DiT switching at boundary_timestep with CFG scheduling, 40-step UniPC flow matching, VAE decoding to 81 frames of h264 mp4):
>
> <img src="/wan22_i2v_arch.png" alt="Wan2.2-I2V-A14B architecture and data flow: dual-expert diffusion MoE (high/low-noise DiT, 14B each, ~28B total params, 14B active); the reference image is encoded by the VAE Encoder into a 20-channel condition (16 reference latent + 4 mask), the initial noise latent is 16 channels, and the two are concatenated into a 36-channel DiT input at every step; the prompt goes through the UMT5-xxl text encoder (24 layers / d_model 4096) to a 4096-dim text embedding injected via cross-attention; each DiT expert uses 3D RoPE self-attn + cross-attn + FFN with hidden 5120 (40 heads x 128); the high-noise expert handles t >= boundary_timestep (boundary_ratio=0.9, about 1/4 of the steps) with CFG=3.5, the low-noise expert handles t < boundary_timestep (about 3/4 of the steps) with CFG 2 to 3.5; after 40 steps of UniPC flow matching denoising the VAE Decoder (with slicing/tiling to save memory) produces 81 frames of 832x480 RGB to mp4. T2V-A14B differences: no I2V image condition branch (DiT input is 16ch instead of 36ch), boundary_ratio=0.875, flow_shift=12.0." style="max-width:100%;border:1px solid #eaeaea;border-radius:8px;" />

Diffusers layout (`model_index.json` with `_class_name = WanImageToVideoPipeline`, `boundary_ratio: 0.9`), five components:

| Component | Class | Key parameters (measured from weight configs) | Disk size |
|---|---|---|---|
| transformer (high-noise expert) | WanTransformer3DModel | 40 layers, 40 heads x head_dim 128 (hidden 5120), FFN 13824, in_channels=36, out_channels=16, patch_size=[1,2,2], QK RMSNorm, cross_attn_norm | 54GB (fp32 storage, ~14B params) |
| transformer_2 (low-noise expert) | WanTransformer3DModel | Structurally identical to transformer (separate weights) | 54GB |
| text_encoder | UMT5EncoderModel (UMT5-xxl encoder) | 24 layers, d_model 4096, 64 heads, d_ff 10240, vocab 256384 (multilingual, prompts in Chinese or English both work) | 11GB (~5.7B) |
| vae | AutoencoderKLWan (Wan2.1 VAE) | z_dim=16, temporal compression 4x (temperal_downsample=[F,T,T]), spatial compression 8x (dim_mult=[1,2,4,4]), with 16-dim latents_mean/std normalization statistics | 485MB |
| scheduler | UniPCMultistepScheduler | prediction_type=flow_prediction, use_flow_sigmas=true (flow matching), num_train_timesteps=1000, solver_order 2 (bh2) | — |

Key points:

1. **DiT input shape arithmetic**. The video goes through the VAE first: 81 frames to (81-1)/4+1 = 21 latent frames; 832x480 to 104x60 latent. Then a 3D patch embedding with patch_size=[1,2,2] gives a token sequence length of 21 x 52 x 30 = 32760 tokens (this is why video diffusion is more expensive per step than an LLM: every denoising step is a full forward pass over 30k tokens, times 40 steps, times two CFG branches).
2. **in_channels=36 is the I2V signature**: 16 (noise latent) + 20 (image condition: 4 mask channels + 16 reference latent channels), see section 3.3. The T2V transformer uses in_channels=16.
3. Each layer = self-attention (3D RoPE positional encoding, rope_max_seq_len=1024) + cross-attention (over UMT5 text embeddings, text_dim=4096) + FFN; the timestep is encoded sinusoidally with freq_dim=256 and fed into AdaLN modulation.
4. Weights are stored in fp32 (hence 118GB on disk); loading with `--dtype bfloat16` halves that.

## 3. Generation Principles

### 3.1 Flow matching denoising

Wan2.2 is trained with rectified flow / flow matching (not classic DDPM epsilon-prediction): the model learns the "straight-line" velocity field from the noise distribution to the data distribution, and the scheduler interprets the model output as flow_prediction. Sampling uses a second-order UniPC multistep solver over 40 steps.

`flow_shift` (sample_shift) controls the redistribution of the timestep sigma: the larger the shift, the more the sampling steps lean toward the high-noise segment (more steps allocated to the large-composition phase). The official I2V value is 5.0, and the official T2V 480p value is 12.0 (`wan_i2v_A14B.py` / `wan_t2v_A14B.py`). Note that `scheduler_config.json` in the weights says 3.0, and vllm-omni ignores it — the scheduler is rebuilt from the server-side `--flow-shift` argument via `resolve_wan_flow_shift()` (see 4.4).

### 3.2 Dual-expert switching by timestep (MoE)

`boundary_timestep = boundary_ratio x num_train_timesteps(1000)`, and with the official I2V `boundary_ratio=0.9` the boundary is t=900:

- `t >= 900` (roughly the first quarter of denoising steps): uses transformer (high-noise expert), CFG with guidance_scale;
- `t < 900` (the remaining steps): switches to transformer_2 (low-noise expert), CFG with guidance_scale_2.

Official I2V guidance is (3.5, 3.5). Each expert performs classifier-free guidance independently (two forward passes for positive and negative prompt), so every step is really 2 x 14B forwards.

### 3.3 I2V image conditioning (the essential difference from T2V)

The reference image does not go through a separate image encoder (`image_encoder: null` in `model_index.json`); instead the VAE encodes the first frame into latent space for channel-wise concatenation:

1. Resize the reference image to the target resolution, build a pseudo-video of "first frame + 80 zero frames," run VAE encode to get a 16-channel latent_condition, and normalize it with latents_mean/std;
2. Build a 4-channel mask (1 at the first-frame position, 0 elsewhere, folded by the temporal compression factor of 4), and concatenate it with latent_condition into a 20-channel condition;
3. At every denoising step, concatenate `[latents(16), condition(20)]` along channels into a 36-channel input for the DiT — the reference image information is injected continuously across all 40 steps, not just used for initialization. This keeps the first frame consistent with the input image while later frames evolve around it.

(Corresponding measurement in the deployment report: I2V is 14-17% slower than T2V, mostly the cost of this condition branch plus the VAE encode.)

### 3.4 VAE decoding

After denoising, the 16x21x104x60 latent is de-normalized and passed through AutoencoderKLWan decode to 81x480x832 RGB frames, which the server then encodes into h264 mp4. Enabling `--vae-use-slicing --vae-use-tiling` (sliced/tiled decoding) significantly lowers peak memory during decoding.

## 4. Code Path in vllm-omni 0.21.0

All paths below are relative to `/workspace/vllm-omni` inside the container; line numbers were measured against that image.

### 4.1 Request entry (API layer)

- `POST /v1/videos` (async task) and `POST /v1/videos/sync` (sync) share form parsing in `vllm_omni/entrypoints/openai/api_server.py:2511 _parse_video_form()`: either the file field `input_reference` (UploadFile) or the string field `image_reference` (URL/base64 JSON), mutually exclusive (line 2547 explicitly rejects both with a 400). This is the origin of "the field name is not image" in Appendix A-1 of the deployment report.
- The image is decoded to PIL by `decode_input_reference()` and put into the request's `multi_modal_data["image"]`, then together with the prompt and sampling parameters (width/height/num_frames/steps/guidance/boundary_ratio/seed...) an `OmniDiffusionRequest` is constructed and queued into the AsyncOmni engine.

### 4.2 Model identification and pipeline routing

- At server startup, `resolve_model_config_path()` in `vllm_omni/entrypoints/utils.py` only recognizes HF transformers or Diffusers layouts — the native Wan format (high_noise_model/ layout) has no `model_index.json`, so it raises `ValueError: Could not determine model_type ...` right there (the source of the error in Appendix A-5).
- `vllm_omni/diffusion/data.py:808` reads `model_index.json` from the weights and takes `_class_name = "WanImageToVideoPipeline"`;
- `vllm_omni/diffusion/registry.py:114` maps it to vllm-omni's own implementation `Wan22I2VPipeline` (`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py:137`), and attaches `get_wan22_i2v_pre_process_func` (registry.py:498, request preprocessing) and `get_wan22_i2v_post_process_func` (registry.py:459, frame post-processing/encoding).

In other words, vllm-omni does not use the diffusers `WanImageToVideoPipeline` implementation; it uses its own pipeline with the same semantics (for TP parallelism, CFG parallelism, cache backends, and so on).

### 4.3 Pipeline main flow (pipeline_wan2_2_i2v.py)

`Wan22I2VPipeline.forward()` (from line 376) does, in order:

1. Read parameters: prompt / image (from multi_modal_data) / width / height / num_frames / steps / guidance. `guidance_scale` goes to the high-noise expert, `guidance_scale_2` to the low-noise expert (lines 444-450).
2. Resolve `boundary_ratio` (lines 452-455):

   ```python
   boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else req.sampling_params.boundary_ratio
   if boundary_ratio is None:
       boundary_ratio = 0.875
       logger.warning("boundary_ratio is required for I2V generation. using default value 0.875")
   ```

   Priority = server-side `--boundary-ratio` > request parameter > hardcoded 0.875 fallback. It never reads the 0.9 from `model_index.json` — this is the code-level root cause of the Appendix A-2 pitfall (I2V must explicitly pass 0.9). There is also an optimization in `__init__` (line 242): when `boundary_ratio` is 1.0/0.0 only a single expert is loaded.
3. `check_inputs` (line 863): width and height must be divisible by 16 (patch_size 2 x vae_scale_factor_spatial 8), otherwise `ValueError: height and width have to be divisible by 16` (Appendix A-6, the reason the 854x480 stress test failed; the T2V pipeline has no such check).
4. `encode_prompt` (line 685): UMT5-xxl encodes the positive/negative prompts.
5. `prepare_latents` (line 760): generate the initial noise latent (seed controlled by the generator) and build the 20-channel condition as described in 3.3 (VAE encode the reference image + fold the mask + normalize with latents_mean/std, code at lines 820-833).
6. `diffuse` denoising loop (line 272):

   ```python
   current_model = self.transformer
   current_guidance_scale = guidance_low
   if boundary_timestep is not None and t < boundary_timestep and self.transformer_2 is not None:
       current_model = self.transformer_2
       current_guidance_scale = guidance_high
   ...
   latent_model_input = torch.cat([latents, condition], dim=1)   # 16+20=36 channels
   noise_pred = self.predict_noise_maybe_with_cfg(...)           # CFG positive/negative branches
   latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
   ```

   where `boundary_timestep = boundary_ratio * scheduler.num_train_timesteps` (lines 541-543) — expert switching, per-step condition concatenation, and CFG all live inside this single loop.
7. VAE decode + post-processing: de-normalize the latent, run `AutoencoderKLWan.decode` (controlled by `--vae-use-slicing/--vae-use-tiling`), then `get_wan22_i2v_post_process_func` converts frames to uint8, and the API layer encodes mp4 and returns.

### 4.4 Scheduler and flow-shift

`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py:44 build_wan_scheduler(sample_solver, flow_shift)` plus `:71 resolve_wan_flow_shift()`: rebuild the `UniPCMultistepScheduler` from the server-side `--flow-shift` (5.0 in this deployment) and request parameters, overriding the 3.0 in the weight directory's `scheduler_config.json`.

### 4.5 TP4 tensor parallelism (wan2_2_transformer.py)

vllm-omni rewrites the diffusers `WanTransformer3DModel` into vLLM-style true tensor parallelism (this is why it is "four cards cooperating on one instance" rather than the multi-instance round-robin used for text-to-image):

- Self-attention uses `QKVParallelLinear` (line 366) plus `RowParallelLinear` for the output projection (line 387), with 40 heads split as 10 heads per card under TP4;
- Cross-attention q/k/v use `ColumnParallelLinear` (lines 476-497);
- FFN `WanFeedForward` (line 103): `ColumnParallelGELU` (column split) + `RowParallelLinear` (row split + all-reduce), with the 13824 intermediate dimension split into 3456 per card;
- QK normalization uses the in-house `DistributedRMSNorm` (line 40): after the head dimension is split, it first does `tensor_model_parallel_all_reduce` to aggregate the sum of squares before normalizing (line 61).

Each card holds 1/4 of each expert's weights plus a full text_encoder/VAE, matching the measured 58.6-64.5GB per card. `--cache-backend cache_dit` adds DiT feature caching on top (reusing intermediate activations that change little across steps) to speed up denoising.

### 4.6 The complete path of one request (put together)

```
curl -F input_reference=@img -F prompt=... /v1/videos/sync
  -> api_server.py:_parse_video_form (2511)         # form/image parsing
  -> OmniDiffusionRequest -> AsyncOmni engine queue
  -> registry: model_index._class_name "WanImageToVideoPipeline" -> Wan22I2VPipeline
  -> forward(): parameter/boundary_ratio parsing -> check_inputs (divisible by 16)
  -> UMT5 encode_prompt -> VAE encode reference image (prepare_latents, 20ch condition)
  -> diffuse: 40 steps x [t>=900: transformer | t<900: transformer_2] x CFG,
              each step cat([latents16, condition20]) -> 36ch, TP4 parallel forward
  -> UniPC scheduler step (flow_prediction, shift=5.0)
  -> VAE decode (slicing/tiling) -> post_process -> h264 mp4 response
```

## 5. Wan2.2-T2V-A14B Architecture and Principles

T2V is the sister model of I2V on the same skeleton, corresponding to `Wan22Pipeline` in vllm-omni (`vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py:300`, `model_index.json _class_name=WanPipeline`):

- Structurally identical dual experts: also 2 x 14B `WanTransformer3DModel` (40 layers / 5120 hidden / FFN 13824), also switching on `boundary_timestep` inside the diffuse loop (line 483), same UMT5-xxl + Wan2.1 VAE + UniPC flow matching. The `boundary_ratio` resolution logic is word-for-word identical (lines 618-623, same 0.875 fallback plus WARNING — T2V's official value happens to be 0.875, so T2V deployments never notice this pitfall).
- No image condition branch: transformer in_channels=16 (pure noise latent), `prepare_latents` only generates noise, each diffuse step feeds 16 channels directly, with no mask/condition concatenation and no VAE encode of a reference image.
- Different official sampling parameters (`wan_t2v_A14B.py`): boundary=0.875, sample_shift=12.0 (480p; 5.0 for 720p), guidance (4.0, 3.0) — T2V uses stronger CFG in the high-noise segment to lock composition and relaxes it later; I2V, with a reference image anchoring the composition, uses a mild 3.5 in both segments.
- No divisibility-by-16 check: T2V `check_inputs` does not verify width/height divisibility (which is why the T2V SOP works at 854x480).
- Historical difference: the ModelScope native weights used in the T2V SOP era needed text_encoder key remapping (Wan native keys to HF UMT5 keys) and a VAE latents_mean/std fix; I2V uses the official HF Diffusers weights directly, so those problems do not exist (the VAE statistics in section 2 of this post are exactly the official values written in during that T2V fix).

### I2V vs T2V Quick Reference

| Dimension | I2V-A14B | T2V-A14B |
|---|---|---|
| Pipeline class (vllm-omni) | Wan22I2VPipeline (pipeline_wan2_2_i2v.py) | Wan22Pipeline (pipeline_wan2_2.py) |
| model_index _class_name | WanImageToVideoPipeline | WanPipeline |
| transformer in_channels | 36 (16 latent + 4 mask + 16 image latent) | 16 |
| Conditioning | reference image VAE latent concatenated per step | text cross-attention only |
| boundary_ratio (official) | 0.9 | 0.875 |
| flow_shift (official, 480p) | 5.0 | 12.0 |
| guidance (official) | (3.5, 3.5) | (4.0, 3.0) |
| Width/height check | must be divisible by 16 | none |
| Required in request | input_reference (image) | prompt only |
| Measured latency (P800 TP4, same load) | x1.14 to 1.17 | baseline |

## 6. References

- Wan2.2 official repo: https://github.com/Wan-Video/Wan2.2 (I2V/T2V configs: wan/configs/wan_i2v_A14B.py, wan_t2v_A14B.py)
- HF weights: Wan-AI/Wan2.2-I2V-A14B-Diffusers, Wan-AI/Wan2.2-T2V-A14B-Diffusers
- vllm-omni source: /workspace/vllm-omni inside the deployment image (vllm 0.21.0 / diffusers 0.38.0 Kunlun XPU edition)
- Deployment / performance / precision measurements: `Wan2.2-I2V-A14B-P800-vllm-omni-Deployment-Test-Report.md` in this directory
