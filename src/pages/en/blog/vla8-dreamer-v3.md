---
title: 'VLA Notes (8): Dreamer V3 — One Set of Hyperparameters Across 50+ Domains'
description: 'Dreamer V3 is the first world-model reinforcement learning algorithm to match or beat domain-specific SOTA on 150+ tasks with a single fixed set of hyperparameters. It learns environment dynamics with an RSSM, unifies reward scales with symlog, and rolls out policies inside imagination. This post breaks down its architecture, training loop, and why it matters for embodied AI.'
pubDate: 2026-08-26
series: VLA Notes
lang: en
altLang: zh
altHref: /blog/vla8-dreamer-v3
layout: ../../../layouts/BlogPost.astro
---

## 0. One-Sentence Positioning

**Dreamer V3** = learning policies by "dreaming inside a model's head," sweeping 150+ tasks including Atari, DMLab, Crafter, Minecraft, and robot arm control with one fixed set of hyperparameters.

It is the turning point where the world-model line of work moved from "tuning art" to "engineering system," and one of the technical ancestors of later physical world models such as Genie, Cosmos, and GR00T-Dreams.

## 1. Why a World Model?

The problem with classic RL:

```text
Interact with the real environment -> expensive, sample-inefficient, dangerous on real hardware
```

The world-model idea:

```text
First learn an internal model of the environment -> roll it out in imagination -> only verify with real interaction when necessary
```

For robots this is the difference between letting the model fall thousands of times inside a dream and letting it fall on a real production line.

## 2. RSSM: Modeling the World as a Recurrent State Space

The core of Dreamer V3 is the world model **RSSM (Recurrent State-Space Model)**, whose state is split in two:

| State | Meaning | Update |
|---|---|---|
| **h_t** | Deterministic recurrent state | GRU recursion |
| **z_t** | Discrete stochastic state | Categorical distribution |

```text
h_t = GRU(h_{t-1}, a_{t-1}, z_{t-1})              # deterministic path
z_t ~ p(z_t | h_t)       = Categorical(NN_prior(h_t))      # prior: pure prediction, no observation
z_t ~ q(z_t | h_t, o_t)  = Categorical(NN_post(h_t, o_t))  # posterior: corrected by observation
```

During training the posterior q updates the model; during imagined rollouts only the prior p is used, because future observations are not available.

## 3. Training Objective: Reconstruction + Reward + Continue + KL

The world model predicts three things at once:

```text
o_hat = Decoder(h_t, z_t)       # reconstruct observation
r_hat = RewardHead(h_t, z_t)    # predict reward
c_hat = ContinueHead(h_t, z_t)  # predict termination
L_WM  = L_recon + L_reward + L_continue + beta * KL(q || p)
```

## 4. symlog: The Key to One Hyperparameter Set Across Domains

Reward scales differ wildly across tasks:

- Atari: 0 to 1000
- Control tasks: -1 to 1
- Minecraft: sparse, delayed rewards

Dreamer V3 unifies the scale with the **symlog** transform:

```text
symlog(x) = sign(x) * ln(1 + |x|)
symexp(x) = sign(x) * (exp(|x|) - 1)   # inverse
```

Both the reward head and the value head now predict compressed values, critic training is more stable, and the same loss weights work in every domain.

## 5. Imagination Rollout: Training Actor-Critic Inside the Dream

<div class="fig">
<svg viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dreamer V3 training loop: world model plus actor-critic trained inside imagination">
  <defs>
    <marker id="arrow2En" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Dreamer V3 training loop</text>

  <rect x="20" y="55" width="140" height="70" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="90" y="82" font-size="12" fill="#047857" text-anchor="middle">Real trajectory</text>
  <text x="90" y="102" font-size="11" fill="#047857" text-anchor="middle">o_t, a_t, r_t</text>

  <rect x="200" y="55" width="160" height="70" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="280" y="82" font-size="12" fill="#1d4ed8" text-anchor="middle">RSSM world model</text>
  <text x="280" y="102" font-size="11" fill="#1d4ed8" text-anchor="middle">learns h_t, z_t, o, r, c</text>

  <rect x="400" y="55" width="140" height="70" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="470" y="82" font-size="12" fill="#b45309" text-anchor="middle">Imagination</text>
  <text x="470" y="102" font-size="11" fill="#b45309" text-anchor="middle">roll out h, z</text>

  <rect x="580" y="55" width="80" height="70" rx="8" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="620" y="82" font-size="12" fill="#6b21a8" text-anchor="middle">Actor</text>
  <text x="620" y="102" font-size="11" fill="#6b21a8" text-anchor="middle">+ Critic</text>

  <line x1="160" y1="90" x2="198" y2="90" stroke="#6b7280" marker-end="url(#arrow2En)"/>
  <line x1="360" y1="90" x2="398" y2="90" stroke="#6b7280" marker-end="url(#arrow2En)"/>
  <line x1="540" y1="90" x2="578" y2="90" stroke="#6b7280" marker-end="url(#arrow2En)"/>

  <rect x="20" y="160" width="640" height="170" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="185" font-size="13" font-weight="700" fill="#1a1a1a">Optimizing the policy inside imagination</text>
  <text x="35" y="210" font-size="12" fill="#444">1. Start from the last real step (h_T, z_T), let the Actor produce action a_T;</text>
  <text x="35" y="230" font-size="12" fill="#444">2. The RSSM prior predicts the next state (h_{T+1}, z_{T+1}) and reward r_{T+1};</text>
  <text x="35" y="250" font-size="12" fill="#444">3. Repeat for H steps to obtain the imagined trajectory;</text>
  <text x="35" y="270" font-size="12" fill="#444">4. The Critic estimates the lambda-return, the Actor maximizes return plus entropy regularization.</text>
  <text x="35" y="300" font-size="12" fill="#6b7280">Loss: L_actor = -E[ symlog(V_lambda) ] + lambda_entropy * H(a); L_critic = MSE(symlog(v), symlog(V_lambda))</text>
</svg>
<p class="cap">Figure: Dreamer V3 trains the world model on real data, then trains Actor-Critic on imagined trajectories.</p>
</div>

## 6. Why Can the Hyperparameters Stay Fixed?

Before Dreamer V3, every domain needed its own tuning: learning rate, discount factor, reward scaling, KL weight, and so on. V3 fixes them thanks to four mechanisms:

1. **symlog/symexp**: unify reward and value scales;
2. **Discrete categorical state**: 32 categories x 32 groups, more stable than a continuous Gaussian;
3. **KL balancing**: dynamically adjusts the prior/posterior weight to prevent model collapse;
4. **Normalization and initialization**: observations, rewards, and gradients are all normalized, reducing sensitivity to task statistics.

## 7. Relation to Embodied AI

Dreamer V3 directly inspired:

- **Genie** (Google): learning interactive world models from video;
- **Cosmos** (NVIDIA): physical world foundation model;
- **GR00T-Dreams** (NVIDIA humanoid): using world models to generate synthetic training data;
- **AgiBot WITA-Omni**, **Unitree GR00T-Dreams line**: leading Chinese labs also use "world models generate data" to relieve the shortage of real teleoperation data.

The core logic:

```text
World model generates synthetic rollouts -> cheaply expand training data -> VLA is more stable on real hardware
```

This is one of the key paths out of the "data hunger" problem in embodied AI.

## 8. Simplified PyTorch Training Loop

```python
for batch in dataloader:
    # 1. Encode the real trajectory
    h, z = rssm.observe(batch.obs, batch.act)

    # 2. World-model loss
    recon  = mse(decoder(h, z), batch.obs)
    reward = mse(reward_head(h, z), symlog(batch.reward))
    cont   = bce(continue_head(h, z), batch.cont)
    kl     = kl_divergence(rssm.posterior(h, batch.obs), rssm.prior(h))
    loss_wm = recon + reward + cont + 0.1 * kl

    # 3. Imagine rollout
    h_imag, z_imag, a_imag, r_imag = rssm.imagine(h[-1], z[-1], actor, horizon=15)

    # 4. Actor-Critic
    values = critic(h_imag, z_imag)
    returns = lambda_return(r_imag, values, gamma=0.997, lambda_=0.95)
    loss_actor = -symlog(returns).mean() + 1e-4 * entropy(a_imag)
    loss_critic = mse(symlog(values.detach()), symlog(returns))

    (loss_wm + loss_actor + loss_critic).backward()
    optimizer.step()
```

## 9. Summary

| Dimension | Dreamer V3's breakthrough |
|---|---|
| Sample efficiency | Rolls out in imagination, cutting real interaction |
| Cross-domain generalization | Fixed hyperparameters fit 150+ different tasks |
| State representation | Discrete categorical + GRU recursion, stable and scalable |
| Reward handling | symlog unifies scales, resolving cross-domain differences |
| Embodied impact | Becomes the base of the VLA + World Model fusion route |

> Remember it in one line: **Dreamer V3 teaches a model to "dream," then uses the experience from those dreams to guide real action.** For a robot, that means falling enough times in the virtual world before it ever touches real hardware.
