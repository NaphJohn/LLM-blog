---
title: 'Inference Systems Infrastructure Notes (4): Knowledge Distillation — Passing the Dark Knowledge of a Large Model to a Small One'
description: 'Knowledge distillation is the foundational technique for compressing and passing on large-model capability. From Hinton classic KL distillation, to DeepSeek R1 distilling reasoning into a 7B model, to on-device VLA deployment: one diagram plus a snippet of pseudocode covering the principle, the gains and the common pitfalls.'
pubDate: 2026-08-26
series: Inference Systems Infrastructure Notes
lang: en
altLang: zh
altHref: /blog/sys4-knowledge-distillation
layout: ../../../layouts/BlogPost.astro
---

## 0. One-Line Positioning

**Knowledge distillation (KD)** = let a large model (the teacher) pass its "dark knowledge" to a small model (the student), so the student approaches the teacher's performance with far fewer parameters.

The problem it solves: **model capability is not the same as deployment cost.** Large models are strong but expensive to serve; small models are cheap but, without enough training data, never learn that well. Distillation lets the small model "stand on the giant's shoulders."

## 1. Why Call It "Dark Knowledge"?

Take an image classification task where the teacher outputs:

```text
cat:   0.70
dog:   0.20
table: 0.05
chair: 0.05
```

A hard label only tells the model "this is a cat." The teacher's soft label additionally implies "this looks like a cat but also somewhat like a dog, and not much like furniture." Those **inter-class similarity signals** are the dark knowledge. A student that learns them generalizes better than one trained on one-hot labels alone.

## 2. Classic Hinton Distillation

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Knowledge distillation: the teacher emits soft targets, the student learns from soft and hard labels">
  <defs>
    <marker id="arrEn4" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Knowledge distillation: transferring dark knowledge</text>

  <rect x="20" y="60" width="160" height="90" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.2"/>
  <text x="100" y="92" font-size="14" font-weight="700" fill="#1e40af" text-anchor="middle">Teacher</text>
  <text x="100" y="115" font-size="12" fill="#1e40af" text-anchor="middle">large model</text>
  <text x="100" y="135" font-size="11" fill="#444" text-anchor="middle">emits soft labels</text>

  <rect x="220" y="75" width="120" height="60" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="280" y="100" font-size="12" font-weight="700" fill="#b45309" text-anchor="middle">Temperature T</text>
  <text x="280" y="120" font-size="11" fill="#b45309" text-anchor="middle">softens the distribution</text>

  <rect x="380" y="60" width="160" height="90" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="1.2"/>
  <text x="460" y="92" font-size="14" font-weight="700" fill="#047857" text-anchor="middle">Student</text>
  <text x="460" y="115" font-size="12" fill="#047857" text-anchor="middle">small model</text>
  <text x="460" y="135" font-size="11" fill="#444" text-anchor="middle">learns soft + hard</text>

  <rect x="580" y="60" width="80" height="90" rx="8" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="620" y="92" font-size="13" font-weight="700" fill="#6b21a8" text-anchor="middle">Loss</text>
  <text x="620" y="115" font-size="11" fill="#6b21a8" text-anchor="middle">a * KL +</text>
  <text x="620" y="132" font-size="11" fill="#6b21a8" text-anchor="middle">(1-a) * CE</text>

  <line x1="180" y1="105" x2="218" y2="105" stroke="#6b7280" marker-end="url(#arrEn4)"/>
  <line x1="340" y1="105" x2="378" y2="105" stroke="#6b7280" marker-end="url(#arrEn4)"/>
  <line x1="540" y1="105" x2="578" y2="105" stroke="#6b7280" marker-end="url(#arrEn4)"/>

  <rect x="20" y="180" width="640" height="100" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="205" font-size="12.5" fill="#1a1a1a" font-family="monospace">p_T = softmax(z_T / T)</text>
  <text x="35" y="225" font-size="12.5" fill="#1a1a1a" font-family="monospace">p_S = softmax(z_S / T)</text>
  <text x="35" y="245" font-size="12.5" fill="#1a1a1a" font-family="monospace">L_distill = T^2 * KL(p_T || p_S)</text>
  <text x="35" y="265" font-size="12.5" fill="#1a1a1a" font-family="monospace">L_total = a * L_distill + (1-a) * CE(y, p_S_hard)</text>
</svg>
<p class="cap">Figure: distillation softens the teacher's soft target with temperature T and uses it as the student's training objective.</p>
</div>

## 3. What Temperature T Does

```text
T = 1:      close to the raw distribution, little dark knowledge
T = 4 to 10: smoother distribution, inter-class relations clearer, dark knowledge rich
T -> inf:   the distribution approaches uniform and loses discriminative power
```

Empirical values: **T = 4 to 10**, with a distillation weight of **alpha = 0.5 to 0.9**.

## 4. Why Better Than Training the Small Model Directly?

| Training method | Information source | Generalization |
|---|---|---|
| Direct training | Hard label (one-hot) | Weak: inter-class relations are lost |
| Knowledge distillation | Teacher soft label + hard label | Strong: inherits the teacher's dark knowledge and ranking |

Typical gain: student parameters cut 10x to 50x, accuracy loss 1% to 3%.

## 5. Representative Work and Measured Gains

| Work | Teacher | Student | Key result |
|---|---|---|---|
| **DeepSeek R1 distillation** | R1 671B MoE | Qwen2.5 / Llama 7B-70B | 7B reaches 92.8% on MATH-500 (up from 58.8%) |
| **Meta Muse Glimmer** | Muse Spark 1.2 | 30B dense | SWE-Bench Verified 76.0; 3.1x speedup on RTX 5090 with DFlash |
| **DistilBERT** | BERT-base | 40% of the parameters | 60% faster, retains 97% of performance |
| **MiniMax on-device distillation** | Large MoE | Small dense | For on-device deployment |

## 6. Relationship to DeepSeek V4's OPD

DeepSeek V4 uses **OPD (On-Policy Distillation)**: the main model distills "domain expert" small models online, compressing specialized capability into one unified model.

How it differs from conventional offline distillation:

```text
Offline:  teacher is frozen -> generates soft labels -> student learns
Online:   teacher and student train together; the student's input distribution
          keeps updating as the policy updates
```

OPD avoids distribution shift, which suits unified models spanning many tasks and domains.

## 7. Embodied Intelligence: On-Device VLA Deployment

Robots have limited onboard compute (a Jetson Orin is on the order of tens to just over a hundred TOPS), so a 7B+ VLA cannot run locally. Distillation is the key path:

```text
Large cloud VLA (teacher)
    | distill
    v
Small on-device VLA (student)
    | deploy
    v
Real-time action inference on the robot
```

In NVIDIA GR00T's "three-computer architecture," DGX trains the large model and the distilled result goes to Jetson. Domestic VLA companies (Zhiyuan, Unitree, Fourier, and others) have to walk the same road.

## 8. Common Pitfalls

1. **T too small** — close to a hard label, so the dark knowledge never transfers;
2. **T too large** — the distribution flattens and the student learns no discrimination;
3. **Student and teacher architectures too different** — the capability gap is too wide and distillation becomes inefficient;
4. **Wrong distillation data distribution** — it must cover the teacher's capability boundary, not be randomly sampled;
5. **Distilling only the final output** — intermediate feature distillation transfers more information;
6. **Ignoring the hard label** — using soft labels alone removes the strong ground-truth constraint.

## 9. Pseudocode

```python
import torch.nn.functional as F

def distill_loss(teacher_logits, student_logits, labels, T=4.0, alpha=0.7):
    # soft target
    p_t = F.softmax(teacher_logits / T, dim=-1)
    p_s = F.log_softmax(student_logits / T, dim=-1)
    loss_kl = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)

    # hard label
    loss_ce = F.cross_entropy(student_logits, labels)

    return alpha * loss_kl + (1 - alpha) * loss_ce
```

## 10. Summary

| Problem | Distillation's answer |
|---|---|
| Large models are too expensive | Distill a small model, cost down 10x to 50x |
| Small models are not strong enough | Inherit the teacher's dark knowledge, generalization up |
| On-device deployment | Large VLA to small VLA to Jetson / robot chip |
| Unifying many tasks | OPD online distillation avoids distribution shift |

> One line to remember: **distillation does not shrink a large model; it teaches the small model the things the large model "knows but never said out loud."**
