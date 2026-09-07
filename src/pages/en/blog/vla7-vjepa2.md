---
title: 'VLA Notes (7): V-JEPA 2 — Self-Supervised Video World Models and Zero-Shot Robot Planning'
description: 'Meta FAIR V-JEPA 2 learns a latent-space world model from one million hours of unlabeled video, then fine-tunes it with only 62 hours of robot data to reach zero-shot manipulation on a Franka arm at 65-80 percent success. This post explains the core idea of predicting in latent space instead of reconstructing pixels, the two-stage training recipe, the Encoder plus Predictor plus planning-head structure, with an architecture diagram and PyTorch-style pseudocode.'
pubDate: 2026-08-26
series: VLA Notes
lang: en
altLang: zh
altHref: /blog/vla7-vjepa2
layout: ../../../layouts/BlogPost.astro
---

## 0. One-Sentence Positioning

> Learn a **latent-space world model** from **one million hours of unlabeled video**, then fine-tune it with **62 hours of robot data** to achieve **zero-shot robot manipulation (65-80% success)**.

V-JEPA 2 (Assran et al., Meta FAIR, 2025-06, arXiv:2506.09985, 1.2B parameters) follows LeCun's JEPA philosophy: an agent should first rehearse "how the world would change if I did this" inside its head before acting — instead of brute-force training on "internet image-text plus massive teleoperation data" the way RT-2, OpenVLA, and pi0 do.

## 1. Core Idea: Predict in Latent Space, Not Pixels

Given a video frame sequence `(x_1, x_2, ..., x_T)`:

```text
# 1) Encode: map every frame into a latent vector space
  z_t = Encoder(x_t)                 # Vision Transformer (ViT-L/H/G)

# 2) Predict: forecast future latents from past latents plus action conditioning
  z_{t+1} = Predictor(z_{t-k...t}, a_t)   # a_t optional; without it this is video self-supervision

# 3) Loss: L2 / Smooth-L1 only on the latent representation of the future frame,
#     never reconstructing pixels; targets come from an EMA teacher encoder.
  L = || z_{t+1} - stop_grad(Encoder_target(x_{t+1})) ||^2
```

Notation: `x_t` = video frame at time t; `z_t` = latent vector; `Encoder` = ViT; `a_t` = optional action; subscript `_target` = EMA teacher encoder.

## 2. Architecture: Encoder + Predictor + Planning Loop

<div class="fig">
<svg viewBox="0 0 680 430" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="V-JEPA 2 architecture: Encoder to latent vector to Predictor plus action to predicted future latent versus EMA target to L2 loss, and the latent MPC planning head">
  <rect x="0" y="0" width="680" height="430" fill="none"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">V-JEPA 2: latent-space prediction plus latent MPC planning</text>

  <rect x="40" y="56" width="190" height="42" rx="6" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="135" y="82" font-size="12.5" fill="#b45309" text-anchor="middle">Video frames x1 ... x_T</text>

  <rect x="300" y="56" width="170" height="42" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="385" y="82" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">Encoder (ViT)</text>
  <line x1="230" y1="77" x2="298" y2="77" stroke="#16a34a" stroke-width="2" marker-end="url(#vgEn)"/>

  <text x="540" y="73" font-size="13" font-weight="700" fill="#1a1a1a">z_t</text>
  <text x="540" y="90" font-size="10" fill="#6b7280">latent vector</text>
  <line x1="470" y1="77" x2="518" y2="77" stroke="#16a34a" stroke-width="2" marker-end="url(#vgEn)"/>

  <rect x="40" y="128" width="150" height="40" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="115" y="153" font-size="11.5" fill="#92400e" text-anchor="middle">frame x_{t+1}</text>
  <rect x="300" y="128" width="170" height="40" rx="6" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="5 3"/>
  <text x="385" y="153" font-size="11.5" fill="#475569" text-anchor="middle">EMA Teacher Encoder</text>
  <line x1="190" y1="148" x2="298" y2="148" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#vgEn)"/>
  <text x="540" y="153" font-size="12.5" font-weight="700" fill="#475569">z (stop_grad)</text>

  <rect x="40" y="216" width="150" height="34" rx="6" fill="#faf5ff" stroke="#a855f7"/>
  <text x="115" y="238" font-size="11" fill="#7e22ce" text-anchor="middle">action a_t (optional)</text>
  <rect x="280" y="206" width="210" height="46" rx="6" fill="#ecfdf5" stroke="#10b981"/>
  <text x="385" y="226" font-size="12" font-weight="700" fill="#047857" text-anchor="middle">Predictor (Transformer)</text>
  <text x="385" y="244" font-size="10.5" fill="#047857" text-anchor="middle">input: past latent z plus action a_t</text>
  <line x1="190" y1="233" x2="278" y2="229" stroke="#a855f7" stroke-width="1.5" marker-end="url(#vgEn)"/>
  <line x1="540" y1="97" x2="540" y2="196" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="4 3"/>
  <line x1="540" y1="196" x2="492" y2="206" stroke="#16a34a" stroke-width="1.5" marker-end="url(#vgEn)"/>
  <text x="560" y="234" font-size="12.5" font-weight="700" fill="#047857">predicted z_{t+1}</text>

  <rect x="300" y="292" width="240" height="40" rx="6" fill="#fef2f2" stroke="#ef4444"/>
  <text x="420" y="316" font-size="12" fill="#b91c1c" text-anchor="middle">L = || predicted z - sg(z) ||^2</text>
  <line x1="560" y1="252" x2="530" y2="290" stroke="#ef4444" stroke-width="1.5" marker-end="url(#vrEn)"/>
  <line x1="560" y1="171" x2="545" y2="290" stroke="#ef4444" stroke-width="1.5" marker-end="url(#vrEn)"/>

  <rect x="40" y="366" width="320" height="48" rx="6" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="200" y="386" font-size="11" font-weight="700" fill="#b45309" text-anchor="middle">Planning head: latent MPC imagines K steps</text>
  <text x="200" y="405" font-size="10.5" fill="#b45309" text-anchor="middle">pick the action chain closest to the goal image embedding</text>
  <line x1="360" y1="390" x2="420" y2="390" stroke="#f59e0b" stroke-width="2" marker-end="url(#vgEn)"/>
  <text x="470" y="386" font-size="12" font-weight="700" fill="#1a1a1a">Action sequence</text>
  <text x="470" y="404" font-size="11" fill="#6b7280">sent to the robot</text>

  <defs>
    <marker id="vgEn" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#16a34a"/></marker>
    <marker id="vrEn" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#ef4444"/></marker>
  </defs>
</svg>
  <p class="cap">Figure: the V-JEPA 2 training loop (Encoder + Predictor + EMA teacher + L2 latent loss) and inference loop (latent MPC planning head). The prediction target is the future frame in latent space, not pixels.</p>
</div>

## 3. Two-Stage Training

- **Stage 1 (action-free self-supervised pretraining)**: one million hours of video plus one million images, with no human labels at all, so the Encoder and Predictor learn "the physics of the world in latent space."
- **Stage 2 (V-JEPA 2-AC action-conditioned fine-tuning)**: fine-tune only the Predictor on **62 hours of the Droid robot dataset**, writing no task-specific reward function, so the model learns "what the future looks like under action a."

## 4. Key Modules

- **Encoder**: ViT-L/H/G with standard ViT blocks (LayerNorm + MHSA + MLP), encoding 16x16 patches into latent vectors; an EMA teacher encoder supplies stable prediction targets.
- **Predictor**: a lightweight Transformer whose input is the positional concatenation of past frame latents plus an optional action token, outputting the latent vector of the next frame.
- **Planning head**: given the current observation and candidate action sequences, "imagine" K steps forward in latent space, pick the action chain whose endpoint is closest to the goal image embedding, then dispatch it with model-predictive control (MPC).
- **Goal representation**: a single **goal image** serves as the task instruction, so the robot can "see" what to do without relying on natural language.

## 5. PyTorch-Style Pseudocode

```python
import torch
import torch.nn as nn

class VJEPA2(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder               # ViT-L/H/G
        self.target_encoder = encoder        # EMA teacher, stop_grad
        self.predictor = predictor           # Transformer
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, frames, actions=None):
        # frames: (B, T, 3, H, W) video clip
        B, T = frames.shape[:2]
        feats = self.encoder(frames.flatten(0, 1))        # (B*T, D)
        feats = feats.unflatten(0, (B, T))                # (B, T, D)
        with torch.no_grad():
            targets = self.target_encoder(frames[:, 1:].flatten(0, 1))
            targets = targets.unflatten(0, (B, T - 1))
        preds = self.predictor(feats[:, :-1], actions)   # (B, T-1, D)
        return preds, targets  # L2 / Smooth-L1 applied in D dimensions

# Planning: model-predictive control in latent space
@torch.no_grad()
def plan(model, obs, goal_embed, action_candidates, horizon=8):
    z = model.encoder(obs)                    # (D,)
    best, best_score = None, -1
    for traj in action_candidates:            # each (T_pred, A_dim)
        z_pred = z
        for a in traj:
            z_pred = model.predictor(z_pred.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
        score = torch.cosine_similarity(z_pred, goal_embed, dim=-1)
        if score > best_score:
            best, best_score = traj, score
    return best  # action sequence dispatched to the robot
```

## 6. Training and Optimization Notes

- **EMA teacher encoder**: the target encoder is updated as an exponential moving average of the encoder weights (momentum about 0.99 to 0.999), preventing representation collapse.
- **Masked latent prediction**: in the spirit of MAE, patches are randomly masked and the predictor only fills in the masked parts, cutting compute while learning a more robust representation.
- **Resolution curriculum**: train at 224x224 first, then fine-tune at 384x384.
- **Never reconstruct pixels**: the loss lives only in latent space, so the training objective aligns naturally with the high-level semantics humans care about (object motion, causality) instead of wasting capacity on texture and color details.

## 7. Complexity and Ablations

- With a ViT-H/16 encoder on 16 frames of 384x384 input, inference is **30x faster** than NVIDIA Cosmos (Meta's own benchmark).
- **77.3%** top-1 on Something-Something v2, beating supervised models of the same scale; PerceptionTest 84.0, TempCompass 76.9.
- Zero-shot robot pick-and-place at **65-80%** success (objects seen in Droid, on a Franka tabletop scene never seen before).
- Ablation highlights: removing action conditioning makes planning fail entirely; removing EMA collapses the representation; switching the prediction space to pixels degrades performance and costs more than 5x the compute.

## 8. Summary and Impact

V-JEPA 2 pushes "world models" from a paper concept to an engineering solution that can be bolted onto real hardware, with three layers of significance:

1. **Route significance** — it proves a robot can "understand" its environment without pixel reconstruction and without large-scale teleoperation;
2. **Data significance** — one million hours of video costs far less at the margin than 62 hours of teleoperation data;
3. **Ecosystem significance** — the full V-JEPA 2 weights are open (CC-BY); combined with NVIDIA Cosmos being closed-source and 30x slower, Meta now leads the world-model route.

For the embodied-AI main thread: bullish for upstream world models / video understanding / VLA trimodal stacks (the embodied "brain" camp), and bullish for video-multimodal pretraining infrastructure (video encoding, latent compression, action tokenization).
