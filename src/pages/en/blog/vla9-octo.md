---
title: 'VLA Notes (9): Octo — A 27M Generalist Robot Policy and What "Fine-Tunable" Really Means'
description: "Octo is the first fully open-source generalist robot policy that can be efficiently fine-tuned to new sensors, new action spaces, and new robot embodiments. A 27M / 93M Transformer backbone with a diffusion-style action head is pretrained on 800k Open X-Embodiment trajectories; Readout Tokens decouple input from output, and 4-step flow matching is enough to run on a robot. This post breaks down the three-stage architecture, action chunking and the Flow Matching loss, the fine-tuning recipe, and compares it against OpenVLA."
pubDate: 2026-09-11
series: VLA Notes
lang: en
altLang: zh
altHref: /blog/vla9-octo
layout: ../../../layouts/BlogPost.astro
---

## 0. One-Sentence Positioning

**Octo** = a **Generalist Robot Policy (GRP)** built from a 27M / 93M Transformer backbone plus a diffusion-style action head, pretrained on 800k multi-embodiment trajectories from Open X-Embodiment. It is the first work that is **fully open-source and can be efficiently fine-tuned to new sensors, new action spaces, and new robot embodiments** (RSS 2024).

It does not chase "bigger means more general." Instead it splits generality into four replaceable parts:

```text
generality = a swappable Tokenizer      (eat any observation)
           + decoupled Readout Tokens   (new task without retraining the backbone)
           + action chunking            (mitigate myopic planning)
           + a diffusion / flow head    (multimodal action distribution, few-step inference)
```

The Diffusion Policy and Flow Matching covered in `vla2` are the engine. Octo is the engineering scheme that mounts the engine in a car and **lets you swap the body**.

## 1. Why a Generalist Robot Policy

Robot foundation models exploded across 2023-2024, but each route kept a flaw:

| Work | Approach | Flaw |
|---|---|---|
| **RT-2** | 55B general VLM emitting actions | Language-only conditioning, no goal images; closed source |
| **RT-1-X** | Trained on many tasks | Locked to a single embodiment |
| **Diffusion Policy** | Diffusion over actions | Single task; retrain per task |
| **Octo** | 27M/93M + swappable tokenizer | Solves all three at once |

Octo targets three concrete problems:

1. **One model accepts either a language instruction or a goal image** as the task description.
2. **Multiple camera layouts + proprioception + optional force-torque** all unify into tokens.
3. **A new robot needs only a new tokenizer plus a little data** — a consumer GPU for a few hours.

Headline result: on WidowX, zero-shot performance beats RT-1-X (the strongest open GRP at the time) and matches or slightly exceeds 55B RT-2-X — **three orders of magnitude fewer parameters**.

## 2. Three Stages: Tokenizer, Transformer Backbone, Diffusion Head

Octo treats robot control as a sequence-to-sequence problem:

<div class="fig">
<svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Octo three-stage architecture: heterogeneous observations become tokens, the Transformer backbone emits Readout Tokens, and the action head decodes an action chunk with Flow Matching">
  <defs>
    <marker id="arrowOctoEn" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Octo architecture: decoupled input, decoupled output</text>

  <rect x="20" y="48" width="150" height="150" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="70" font-size="12" font-weight="700" fill="#1a1a1a">(1) Input Tokenizer</text>
  <rect x="34" y="82" width="122" height="26" rx="5" fill="#eff6ff" stroke="#2563eb"/>
  <text x="95" y="99" font-size="11" fill="#1d4ed8" text-anchor="middle">language - 1 token</text>
  <rect x="34" y="114" width="122" height="26" rx="5" fill="#eff6ff" stroke="#2563eb"/>
  <text x="95" y="131" font-size="11" fill="#1d4ed8" text-anchor="middle">image - patch tokens</text>
  <rect x="34" y="146" width="122" height="26" rx="5" fill="#eff6ff" stroke="#2563eb"/>
  <text x="95" y="163" font-size="11" fill="#1d4ed8" text-anchor="middle">proprio - 1 token</text>

  <rect x="200" y="48" width="180" height="150" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="215" y="70" font-size="12" font-weight="700" fill="#047857">(2) Transformer Backbone</text>
  <rect x="214" y="82" width="152" height="26" rx="5" fill="#d1fae5" stroke="#10b981"/>
  <text x="290" y="99" font-size="11" fill="#047857" text-anchor="middle">K Readout Tokens</text>
  <rect x="214" y="114" width="152" height="52" rx="5" fill="#ffffff" stroke="#10b981"/>
  <text x="290" y="134" font-size="11" fill="#047857" text-anchor="middle">N layers of MHSA + MLP</text>
  <text x="290" y="152" font-size="10" fill="#047857" text-anchor="middle">read-only mask: inputs never see readout</text>
  <text x="290" y="180" font-size="10" fill="#6b7280" text-anchor="middle">output gathered only at readout positions</text>

  <rect x="410" y="48" width="250" height="150" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="425" y="70" font-size="12" font-weight="700" fill="#b45309">(3) Action Head (Flow Matching)</text>
  <text x="425" y="94" font-size="11" fill="#b45309">noise a0 ~ N(0,I) plus conditioning o</text>
  <text x="425" y="116" font-size="11" fill="#b45309">FiLM injects global o into every layer, predicts v</text>
  <text x="425" y="138" font-size="11" fill="#b45309">Euler integration in 4-16 steps - action chunk (T_p)</text>
  <text x="425" y="162" font-size="11" fill="#b45309">execute first T_a steps, then replan (receding horizon)</text>
  <text x="425" y="186" font-size="10" fill="#6b7280">swappable for DDPM (~100 steps, 27x slower)</text>

  <line x1="172" y1="122" x2="196" y2="122" stroke="#6b7280" marker-end="url(#arrowOctoEn)"/>
  <line x1="382" y1="122" x2="406" y2="122" stroke="#6b7280" marker-end="url(#arrowOctoEn)"/>

  <rect x="20" y="222" width="640" height="158" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="246" font-size="13" font-weight="700" fill="#1a1a1a">Why split it this way: a new embodiment only touches the two ends</text>
  <text x="35" y="272" font-size="12" fill="#444">- New sensor or camera layout: change the Tokenizer only, backbone untouched;</text>
  <text x="35" y="294" font-size="12" fill="#444">- New action space or DoF: change the Action Head output width, backbone untouched;</text>
  <text x="35" y="316" font-size="12" fill="#444">- New task or instruction format: train the Readout Tokens, backbone can stay frozen;</text>
  <text x="35" y="338" font-size="12" fill="#444">- So fine-tuning becomes fitting an adapter, not retraining a model. That is the real claim.</text>
  <text x="35" y="364" font-size="11" fill="#6b7280">Contrast: OpenVLA puts capacity in a 7B VLM plus LoRA; Octo uses a small backbone trained from scratch plus readout.</text>
</svg>
<p class="cap">Figure: heterogeneous observations become unified tokens, Readout Tokens summarize them, and a flow-matching head decodes the action chunk. Both ends are swappable.</p>
</div>

### (1) Input Tokenizer

| Input | Encoding | Token count |
|---|---|---|
| Language instruction | Sentence-BERT | 1 sentence token |
| Goal image / wrist camera | small CNN (ResNet features) | 64-256 patch tokens |
| Proprioception | MLP | 1 token |
| Force / touch (optional) | MLP | by DoF |

### (2) Backbone plus Readout Tokens

An N-layer standard Transformer where every layer lets tokens attend to each other. The twist is **K Readout Tokens** prepended to the sequence with a **read-only attention mask**: readouts can see all input tokens, but input tokens never see the readouts.

Those K vectors become the summary of the whole input and feed the action head. The payoff is full decoupling of input and output: a new embodiment adds a tokenizer and trains readout, with no backbone retraining.

### (3) Action Head: Flow Matching or DDPM

The head takes the readout token (conditioning o) plus noise a0, injects o globally via **FiLM (Feature-wise Linear Modulation)** into every denoising layer, and emits an action chunk of Tp future steps. It executes Ta steps and replans.

## 3. Action Chunking and the Flow Matching Loss

Action chunking is the default posture: **Tp is the prediction horizon, Ta the execution horizon, with Ta < Tp**. This receding-horizon pattern is standard across ACT, π0, and Diffusion Policy.

The training objective in one formula:

```text
x_t = (1 - t) * eps + t * a_target          # t in [0,1], a straight interpolation path
v_pred = head(x_t, t, o)                    # predict the velocity field
L_FM = mean( (v_pred - (a_target - eps))^2 ) # regress the velocity, target is a_target - eps
```

At inference, start from noise and run a deterministic ODE integration: **4-16 steps is enough**, versus roughly 100 for a DDPM of the same architecture.

```python
import torch, torch.nn as nn

class OctoPolicy(nn.Module):
    def __init__(self, backbone, tokenizer, action_head, K, d):
        super().__init__()
        self.tokenizer   = tokenizer     # language / image / proprio -> tokens
        self.backbone    = backbone      # N-layer Transformer
        self.readout     = nn.Parameter(torch.randn(K, d))   # K learnable readout tokens
        self.action_head = action_head   # Flow Matching / DDPM head

    def encode_obs(self, lang, image, proprio):
        toks = []
        if lang     is not None: toks.append(self.tokenizer.language(lang))
        if image    is not None: toks.append(self.tokenizer.vision(image))
        if proprio is not None: toks.append(self.tokenizer.proprio(proprio))
        r   = self.readout[None].expand(toks[0].size(0), -1, -1)   # [B, K, d]
        seq = torch.cat([r, *toks], dim=1)                          # readout first
        return self.backbone(seq, readout_mask=True)                # gather only at readout

    def flow_loss(self, a_target, o_feat):
        B   = a_target.size(0)
        t   = torch.rand(B, 1, 1, device=a_target.device)
        eps = torch.randn_like(a_target)
        x_t = (1 - t) * eps + t * a_target
        v_pred = self.action_head(x_t, t, o_feat)
        return ((v_pred - (a_target - eps)) ** 2).mean()

    @torch.no_grad()
    def predict_chunk(self, o_feat, n_steps=8):
        a  = torch.randn(o_feat.size(0), T_p, action_dim, device=o_feat.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            a = a + self.action_head(a, i * dt, o_feat) * dt
        return a                                                    # [B, T_p, action_dim]

# closed loop: predict a full chunk, execute the first T_a steps, then replan
for episode in episodes:
    obs = env.reset()
    for t in range(horizon):
        o_feat = policy.encode_obs(**obs)
        chunk  = policy.predict_chunk(o_feat)
        for k in range(T_a):
            obs = env.step(chunk[:, k])
            if done: break
```

## 4. Fine-Tuning to a New Embodiment

This is what separates Octo from yet another VLA reproduction:

```text
freeze:  Transformer Backbone + Readout Tokens
train:   the new Action Head + a new Tokenizer adapter
data:    50-100 trajectories in the target domain
hardware: a single A6000, a few hours
```

Default settings from the paper, useful for reproduction:

| Item | Setting |
|---|---|
| Data | 25 Open X-Embodiment datasets, 800k episodes, weighted by size and task diversity |
| Optimizer | AdamW, weight decay 1e-4 |
| LR | cosine warmup, 5000 steps to 1e-4 |
| Batch | 256, one trajectory per GPU |
| EMA | decay 0.9999, roughly +0.5 to 1% success |

## 5. Ablations: What Actually Matters

| Design choice | Finding |
|---|---|
| **Action chunking** | On Stanford Coffee, Tp=16 lifts success from 0.45 to 0.75 (**+30pp**) and is the key fix for myopic planning |
| **Readout tokens** | Conditioning on the last token instead of readout costs **about 5pp**; readout compresses context into K vectors that transfer better |
| **Flow Matching vs DDPM** | Same architecture and data: FM at 4 steps scores **0.798**, DDPM at 100 steps scores 0.816 (2% gap), but latency is **23 ms vs 635 ms (27.5x)** |
| **Model size** | 27M to 93M averages 0.62 to 0.72 across 9 real robots, but after fine-tuning the gap shrinks to **about 0.02** |
| **Goal image vs language** | On WidowX, goal images win by about **25%** thanks to higher information density |

The last row is worth remembering: **a bigger model mainly buys zero-shot capability; once fine-tuning is allowed, small models close in fast.** That is very good news for on-device deployment.

## 6. Head-to-Head with OpenVLA

| Dimension | **Octo** | **OpenVLA** |
|---|---|---|
| Backbone | 27M / 93M, trained from scratch | 7B VLM (frozen LLaVA) |
| Adaptation | new Tokenizer + trained Readout + new Action Head | LoRA fine-tuning |
| Action decoding | Flow Matching (4 steps) | discrete action tokens, autoregressive |
| Openness | fully open source, including pretrained checkpoints | open weights |
| Inference cost | real time on a consumer GPU (<30 ms per chunk) | needs substantial VRAM |
| Shared conclusion | **a generalist policy no longer requires per-robot retraining** | same |

Two routes, one destination. The difference is small model with decoupled interfaces versus large model with parameter-efficient tuning.

## 7. What It Means for Embodied AI

1. **The algorithm stack is commoditizing.** Once a policy can be re-headed and its backbone frozen, differentiation stops coming from cleverer network design and shifts to **data scale, embodiment stability, and commercial scenarios**. That accelerates — and pressures — the moat-building speed of integrators like AgiBot, Unitree, and UBTech.
2. **Small model plus big data validates on-device feasibility.** A 27M model with 4-step FM inference under 30 ms supports 30 Hz control. This is the first time an on-device generalist VLA clears the **parameters / latency / performance triangle simultaneously**.
3. **Open checkpoints let domestic teams take off immediately.** With pretrained weights public, the engineering window compresses, and the race becomes whose scenario data flywheel spins faster — matching the competitive landscape in `mm4` and `vla4`.

## 8. Common Pitfalls

1. **Writing the readout mask backwards.** If input tokens attend back to readouts, output information leaks into the input; fine-tuning scores look inflated and collapse on a new task.
2. **Setting Ta too large.** The closer execution gets to Tp, the lower the replanning rate and the more likely a stale action chunk is executed in a dynamic scene. A practical value is Ta about half of Tp.
3. **FM is more LR-sensitive than DDPM.** Too high oscillates; for fine-tuning, start lower.
4. **Swapping only the Action Head.** If a new embodiment moves the cameras, token semantics break. Align the tokenizer first.

## 9. Summary

| Dimension | Octo's answer |
|---|---|
| Where generality comes from | Not parameter count, but **decoupled interfaces** across tokenizer, readout, and head |
| Why fine-tuning is cheap | Frozen backbone plus adapter layers; 50-100 trajectories, a few hours |
| Why it is fast | Flow Matching in 4 steps, 27.5x faster than DDPM for a 2% quality cost |
| Why chunking matters | Action chunking lifts myopic planning from 0.45 to 0.75 |
| Engineering takeaway | Algorithm commoditization pushes competition to data and scenarios; small models plus on-device inference open the deployment window |

> One line to remember: **Octo does not prove that small models win. It proves that once you cut the interfaces apart, a generalist robot policy becomes a maintainable engineering problem.**

---

**Series navigation**: previous post [`vla8-dreamer-v3`](/en/blog/vla8-dreamer-v3) covers world-model RL; action generation fundamentals are in [`vla2-action-generation`](/en/blog/vla2-action-generation); [`pp2-diffusion-policy`](/en/blog/pp2-diffusion-policy) is the paper primer on Diffusion Policy.
