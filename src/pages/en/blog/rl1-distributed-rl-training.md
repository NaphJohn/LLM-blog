---
title: 'RL Training Notes (1): Distributed RL Training — Running PPO / GRPO From One GPU to a Cluster'
description: 'Distributed RL training splits the Actor, Critic, Reference and Reward roles onto separate GPU pools, puts vLLM or SGLang behind the rollout stage, and broadcasts trained weights back to the inference engine over NCCL. It turns RLHF and RLAIF from one pot of everything into a pipeline. This post covers the four-role architecture, the six stages of a PPO step, the PPO and GRPO objectives, the two weight-sync paths, measured speedups, and how it connects to VLA preference alignment.'
pubDate: 2026-09-11
series: RL Training Notes
lang: en
altLang: zh
altHref: /blog/rl1-distributed-rl-training
layout: ../../../layouts/BlogPost.astro
---

## 0. One-Sentence Positioning

**Distributed RL training** = Ray for orchestration, with **Actor (policy) / Critic (value) / Reference (frozen, for KL) / Reward (scoring)** split onto separate GPU pools, a **vLLM or SGLang rollout engine** sitting in the middle to generate, and a high-speed **weight sync** broadcasting parameters from training back to inference. It is the shared **engineering substrate** behind RLHF, RLAIF, preference tuning, and robot VLA preference alignment.

It is not a new algorithm. It is the decision to stop letting one model act as both performer and judge on a single GPU.

## 1. Why RLHF Cannot Be One Pot

The textbook PPO-RLHF implementation (early TRL / HF Trainer) runs in a single process:

```text
same weights: fp forward to generate -> reward scoring -> same weights: backward update
```

That survives below 1B parameters and collapses above 7B, for three reasons:

1. **One set of weights plays two roles.** It is the Actor (needs gradients) and the Reference (frozen, computes KL). Both copies must stay resident, doubling memory.
2. **Inference and training fight over the same SMs.** Generating a 256-token response is a bandwidth-bound inference workload where bigger batches win. Gradient descent is a compute-bound workload. Jammed together, neither runs at its operating point.
3. **The time distribution is wildly uneven.** Within a PPO step, **rollout (generation) accounts for roughly 80% of the wall clock**; DeepSpeed training accounts for 20%. Running them sequentially on the same cards means those cards idle during the training phase.

In one line: **rollout and training are two different workloads and must be asynchronous.**

## 2. Four Roles Decoupled Plus a Rollout Engine

<div class="fig">
<svg viewBox="0 0 680 390" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Distributed RL training architecture: Ray orchestrates Actor, Critic, Reference and Reward GPU pools, vLLM or SGLang handles rollout, and weight sync broadcasts trained parameters back to inference">
  <defs>
    <marker id="arrowRLEn" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Distributed RL training: four decoupled roles plus a dedicated rollout engine</text>

  <rect x="20" y="46" width="150" height="112" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="35" y="68" font-size="12" font-weight="700" fill="#b45309">vLLM / SGLang</text>
  <text x="35" y="88" font-size="11" fill="#b45309">Rollout Engine</text>
  <text x="35" y="108" font-size="11" fill="#b45309">PagedAttention generation</text>
  <text x="35" y="128" font-size="11" fill="#b45309">about 80% of a PPO step</text>
  <text x="35" y="148" font-size="10" fill="#6b7280">enable_sleep to hand VRAM back</text>

  <rect x="196" y="46" width="150" height="112" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="211" y="68" font-size="12" font-weight="700" fill="#1d4ed8">Actor GPUs</text>
  <text x="211" y="88" font-size="11" fill="#1d4ed8">main policy, needs grads</text>
  <text x="211" y="108" font-size="11" fill="#1d4ed8">DeepSpeed ZeRO-3 / FSDP</text>
  <text x="211" y="128" font-size="11" fill="#1d4ed8">PPO or GRPO update</text>

  <rect x="372" y="46" width="140" height="112" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="387" y="68" font-size="12" font-weight="700" fill="#047857">Critic GPUs</text>
  <text x="387" y="88" font-size="11" fill="#047857">value head</text>
  <text x="387" y="108" font-size="11" fill="#047857">GAE advantage</text>
  <text x="387" y="128" font-size="10" fill="#6b7280">GRPO removes this pool entirely</text>

  <rect x="538" y="46" width="122" height="52" rx="8" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="599" y="68" font-size="12" font-weight="700" fill="#6b21a8">Reference</text>
  <text x="599" y="86" font-size="11" fill="#6b21a8">frozen, computes logp</text>

  <rect x="538" y="106" width="122" height="52" rx="8" fill="#fef2f2" stroke="#e24b4a"/>
  <text x="599" y="128" font-size="12" font-weight="700" fill="#a32d2d">Reward</text>
  <text x="599" y="146" font-size="11" fill="#a32d2d">RM or rule verifier</text>

  <line x1="172" y1="102" x2="192" y2="102" stroke="#6b7280" marker-end="url(#arrowRLEn)"/>
  <line x1="348" y1="102" x2="368" y2="102" stroke="#6b7280" marker-end="url(#arrowRLEn)"/>
  <line x1="514" y1="102" x2="534" y2="102" stroke="#6b7280" marker-end="url(#arrowRLEn)"/>

  <rect x="20" y="182" width="640" height="86" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="204" font-size="13" font-weight="700" fill="#1a1a1a">Weight Sync: training side to inference side (not optional)</text>
  <text x="35" y="226" font-size="12" fill="#444">(1) Ray Object Store: GPU to CPU, serialize, CPU to GPU. Traffic O(N x params). Simple, slow.</text>
  <text x="35" y="246" font-size="12" fill="#444">(2) NCCL broadcast: direct GPU to GPU via a StatelessProcessGroup side channel. Zero copy.</text>
  <text x="35" y="262" font-size="11" fill="#6b7280">OpenRLHF, veRL, NeMo-RL and Miles all take the second path.</text>

  <rect x="20" y="284" width="640" height="86" rx="8" fill="#fef2f2" stroke="#e24b4a"/>
  <text x="35" y="306" font-size="13" font-weight="700" fill="#a32d2d">What happens without sync: policy mismatch</text>
  <text x="35" y="328" font-size="12" fill="#444">The Actor updates but never broadcasts to vLLM, so the engine keeps generating with the previous</text>
  <text x="35" y="348" font-size="12" fill="#444">weights. The sampling policy is not the training policy, the importance ratio is wrong, and PPO diverges.</text>
</svg>
<p class="cap">Figure: four roles on independent GPU pools sharing an experience buffer through Ray Object Store; after every update the weights must go back to the inference engine.</p>
</div>

None of this is algorithmic novelty. It is all engineering, and it answers one question: **how do you spend 80% of the time on the thing that matters (generation) instead of idling cards?**

## 3. Six Stages of a PPO Step

```text
(1) Rollout generation       : vLLM generates responses (the longest stage)
(2) Reward scoring           : reward model or rule verifier
(3) Reference forward + KL   : the frozen model computes logp_ref per sample
(4) GAE advantage            : the Critic estimates advantages
(5) Critic / Policy gradient : the backward update
(6) Weight sync back to vLLM : broadcast parameters to the engine, next round
```

The hard parts are (3) and (6). Stage (3) dictates your memory budget (Reference weights plus KV), and stage (6) dictates whether the pipeline stalls waiting for a barrier.

## 4. The PPO and GRPO Objectives

The core of PPO in four lines:

```text
A_t    = R_t - V(s_t)                                   # advantage (GAE in practice)
rho_t  = pi_theta(a_t|s_t) / pi_old(a_t|s_t)            # importance ratio, must be >= 0
L_CLIP = E[ min( rho_t*A_t, clip(rho_t, 1-eps, 1+eps)*A_t ) ]
L_KL   = -beta * E[ log pi_theta(a_t) - log pi_ref(a_t) ]
L_PPO  = L_CLIP - c1 * L_VF + c2 * L_KL
```

Here `eps` defaults to 0.2, `beta` is raised adaptively as measured KL grows (too much drift means pulling back), and `c1` / `c2` weight the value and KL terms. The Reference must be frozen while computing `logp_ref`.

**GRPO's contribution is deleting the Critic.** Instead of a value head, sample a group of answers for the same prompt and use within-group relative ranking as the advantage:

```text
A_i = (r_i - mean(r_group)) / std(r_group)      # group-normalized, replaces the Critic
```

One fewer GPU pool, one fewer model in memory, one fewer objective. The cost is that **group size must be at least 4**, otherwise the within-group variance estimate is unstable.

## 5. Weight Sync: Two Paths

```python
def sync_weights_to_vllm(actor, vllm):
    for name, param in actor.named_parameters():
        if param.requires_grad:
            # zero-copy GPU-to-GPU broadcast over a side-channel NCCL group
            vllm.collective_rpc("load_weights", param.data)
```

| Path | Mechanism | Verdict |
|---|---|---|
| **Ray Object Store** | GPU to CPU, serialize, CPU to GPU | Traffic O(N x params); easy to configure |
| **NCCL broadcast** | direct GPU-to-GPU tensor broadcast | the default in every serious framework |

vLLM 0.7+ exposes `WorkerExtension` and `collective_rpc`, so the training side can build a side-channel NCCL group with `StatelessProcessGroup` and broadcast zero-copy.

## 6. Measured Speedups

| Framework | Result |
|---|---|
| **OpenRLHF** | Full PPO RLHF convergence at 70B, roughly **3x faster** than the original TRL |
| **veRL** (ByteDance Seed) | PPO RLHF iteration on Qwen2.5-32B drops from 1.5 hours to **25 minutes** |
| **DeepSeek V4** in-house CANN-GRPO | KL correction and reward modeling fused into a single forward pass; RLHF cycle **7 days to 19 hours**, SWE-bench 42.1 to 58.2 (+38.2%) |
| **DeepSeek V4 + Miles** | MegaMoE fuses communication and compute into one GPU kernel, **1.96x** on the latency-sensitive rollout path |
| **AReaL** (Ant Group) | fully asynchronous rollout plus replay buffer pushes 70B RL training utilization above **80%** |
| **NeMo-RL** (NVIDIA) | 70B+ full-stack RLHF training, RTX-only pipeline on GB300 |

Read as a group, the trend is clear: **almost all of the efficiency gain comes from decoupling rollout from training, not from a new loss function.**

## 7. Four Common Pitfalls

1. **Sync gap.** If weight sync is slow, training uses v1 weights while inference uses v2, and the policy gradient becomes meaningless. Sync right after every mini-batch.
2. **Actor memory blowup.** The Reference cannot be offloaded to CPU, or blocked I/O during KL computation stalls the whole pipeline.
3. **vLLM without sleep.** When switching between rollout and training phases, call `enable_sleep` on the vLLM engine to release VRAM, or you OOM.
4. **GRPO group variance.** With group size below 4, the within-group advantage estimate is too noisy and training oscillates. Enlarge the group or fall back to PPO with a Critic.

## 8. How It Connects to VLA

This infrastructure serves more than text RLHF. Robot policy **RLAIF** runs on exactly the same stack:

```text
VLA generates trajectories -> reward model scores -> within-group ranking (GRPO) -> policy update -> sync back to inference
```

- Both Octo and π0 already support RLAIF against a custom reward model.
- **GRPO is the efficient reference algorithm for VLA fine-tuning**: dropping the Critic can cut Octo-Small (27M) fine-tuning from 2 hours to **about 20 minutes**.
- It is a hard requirement for **parallel multi-scenario training in a factory**: train and serve robots by day, collect rollouts by night, all on one pool of 64 GPUs. That is precisely the mainline OpenRLHF / veRL scenario, and robotics companies can inherit most of the boilerplate for free.

## 9. Summary

| Dimension | Answer |
|---|---|
| The problem | Rollout and training are different workloads; co-locating them wastes 80% of the cards |
| The fix | Four decoupled roles, a dedicated rollout engine, NCCL weight sync |
| Biggest gain source | The decoupling itself, not a new loss (veRL: 1.5h to 25min) |
| Do you need a Critic | Optional. GRPO removes it, at the cost of requiring group size >= 4 |
| Link to VLA | RLAIF and text RLHF share one stack; GRPO transfers directly to VLA fine-tuning |

> One line to remember: **the bottleneck in RLHF was never the algorithm. It was letting generation and training each sit in their proper place. Distributed RL training is that idea, engineered.**

---

**Series navigation**: this is the opening post of the series. Related background lives in [`fa4-deepseek-v4`](/en/blog/fa4-deepseek-v4) (V4 training and compression) and [`sys8-parallel-strategies-pd-disaggregation`](/en/blog/sys8-parallel-strategies-pd-disaggregation) (parallelism strategies and PD disaggregation).
