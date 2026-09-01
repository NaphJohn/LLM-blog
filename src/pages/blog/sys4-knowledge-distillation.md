---
title: 推理系统基础设施手记（四）：知识蒸馏 —— 把大模型的「暗知识」传给小模型
description: 知识蒸馏是大模型压缩与传承的基石技术。从 Hinton 的经典 KL 蒸馏，到 DeepSeek R1 把推理能力蒸馏进 7B 模型，再到 VLA 端侧部署——本文用一张图 + 一段伪代码讲清原理、收益与常见坑。
pubDate: 2026-08-26
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys4-knowledge-distillation
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话定位

**知识蒸馏（Knowledge Distillation, KD）** = 让大模型（Teacher）把「暗知识」教给小模型（Student），让小模型在参数量大幅减少的情况下逼近大模型性能。

它解决的核心问题：**模型能力 ≠ 部署成本**。大模型很强，但推理贵；小模型便宜，但训练数据不够时学不到那么好。蒸馏让小模型「站在巨人肩膀上」。

## 1. 为什么叫「暗知识」？

假设一张图分类任务，Teacher 输出：

```text
猫：0.70
狗：0.20
桌子：0.05
椅子：0.05
```

Hard label 只会告诉模型「这是猫」。但 Teacher 的 soft label 还隐含了「这张图像猫又像狗，不太像家具」。这些**类间相似性信息**就是暗知识。Student 学这些暗知识，比只学 one-hot 标签泛化更好。

## 2. 经典 Hinton 蒸馏

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="知识蒸馏：Teacher 输出软目标，Student 学习 soft label 与 hard label">
  <defs>
    <marker id="arr3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">知识蒸馏：暗知识传递</text>

  <!-- teacher -->
  <rect x="20" y="60" width="160" height="90" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.2"/>
  <text x="100" y="92" font-size="14" font-weight="700" fill="#1e40af" text-anchor="middle">Teacher</text>
  <text x="100" y="115" font-size="12" fill="#1e40af" text-anchor="middle">大模型</text>
  <text x="100" y="135" font-size="11" fill="#444" text-anchor="middle">输出 soft label</text>

  <!-- temperature -->
  <rect x="220" y="75" width="120" height="60" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="280" y="100" font-size="12" font-weight="700" fill="#b45309" text-anchor="middle">Temperature T</text>
  <text x="280" y="120" font-size="11" fill="#b45309" text-anchor="middle">软化概率分布</text>

  <!-- student -->
  <rect x="380" y="60" width="160" height="90" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="1.2"/>
  <text x="460" y="92" font-size="14" font-weight="700" fill="#047857" text-anchor="middle">Student</text>
  <text x="460" y="115" font-size="12" fill="#047857" text-anchor="middle">小模型</text>
  <text x="460" y="135" font-size="11" fill="#444" text-anchor="middle">学习 soft + hard</text>

  <!-- loss -->
  <rect x="580" y="60" width="80" height="90" rx="8" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="620" y="92" font-size="13" font-weight="700" fill="#6b21a8" text-anchor="middle">Loss</text>
  <text x="620" y="115" font-size="11" fill="#6b21a8" text-anchor="middle">α·KL +</text>
  <text x="620" y="132" font-size="11" fill="#6b21a8" text-anchor="middle">(1-α)·CE</text>

  <line x1="180" y1="105" x2="218" y2="105" stroke="#6b7280" marker-end="url(#arr3)"/>
  <line x1="340" y1="105" x2="378" y2="105" stroke="#6b7280" marker-end="url(#arr3)"/>
  <line x1="540" y1="105" x2="578" y2="105" stroke="#6b7280" marker-end="url(#arr3)"/>

  <!-- formula -->
  <rect x="20" y="180" width="640" height="100" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="205" font-size="12.5" fill="#1a1a1a" font-family="monospace">p_T = softmax(z_T / T)</text>
  <text x="35" y="225" font-size="12.5" fill="#1a1a1a" font-family="monospace">p_S = softmax(z_S / T)</text>
  <text x="35" y="245" font-size="12.5" fill="#1a1a1a" font-family="monospace">L_distill = T² · KL(p_T || p_S)</text>
  <text x="35" y="265" font-size="12.5" fill="#1a1a1a" font-family="monospace">L_total = α · L_distill + (1-α) · CE(y, p_S_hard)</text>
</svg>
<p class="cap">图：知识蒸馏把 Teacher 的 soft target 用温度 T 软化后，作为 Student 的训练目标。</p>
</div>

## 3. 温度 T 的作用

```text
T = 1:  接近原始分布，暗知识较少
T = 4~10: 分布更平滑，类间关系更明显，暗知识丰富
T → ∞: 分布趋于均匀，失去区分度
```

经验值：**T = 4~10**，配合蒸馏权重 **α = 0.5~0.9**。

## 4. 为什么比直接训练小模型好？

| 训练方式 | 信息来源 | 泛化能力 |
|---|---|---|
| 直接训练 | Hard label（one-hot）| 弱：丢失类间关系 |
| 知识蒸馏 | Teacher soft label + hard label | 强：继承 Teacher 的暗知识与排序 |

典型收益：Student 参数减 10–50×，精度损失 1–3%。

## 5. 代表工作与实测收益

| 工作 | Teacher | Student | 关键结果 |
|---|---|---|---|
| **DeepSeek R1 蒸馏** | R1 671B MoE | Qwen2.5 / Llama 7B–70B | 7B 在 MATH-500 达 92.8%（原 58.8%） |
| **Meta Muse Glimmer** | Muse Spark 1.2 | 30B dense | SWE-Bench Verified 76.0，RTX 5090 + DFlash 3.1× 加速 |
| **DistilBERT** | BERT-base | 40% 参数 | 快 60%，保留 97% 性能 |
| **MiniMax 端侧蒸馏** | MoE 大模型 | Dense 小模型 | 用于端侧部署 |

## 6. 与 DeepSeek V4 OPD 的关系

DeepSeek V4 用了 **OPD（On-Policy Distillation，同策略蒸馏）**：主模型在线蒸馏「领域专家」小模型，把专项能力压缩进统一模型。

与传统离线蒸馏的区别：

```text
离线蒸馏：Teacher 固定 → 生成 soft label → Student 学
在线蒸馏：Teacher 和 Student 同时训练，Student 的输入分布随策略更新而更新
```

OPD 避免分布偏移，适合多任务、多领域统一模型。

## 7. 具身智能关联：VLA 端侧部署

机器人 onboard 算力有限（如 Jetson Orin 几十到一百多 TOPS），无法跑 7B+ VLA。蒸馏是关键路径：

```text
云端大 VLA（Teacher）
    ↓ 蒸馏
端侧小 VLA（Student）
    ↓ 部署到机器人
实时动作推理
```

NVIDIA GR00T 的「三计算机架构」中，DGX 训练大模型，蒸馏后部署到 Jetson。这也是国产 VLA 公司（智元、宇树、傅利叶等）必须走的路。

## 8. 常见坑

1. **T 太小**：接近 hard label，暗知识没传过去；
2. **T 太大**：分布太平，Student 学不到区分度；
3. **Student 架构与 Teacher 差异过大**：能力 gap 太大，蒸馏效率低；
4. **蒸馏数据分布不对**：要覆盖 Teacher 的能力边界，而不是随便抽数据；
5. **只蒸馏最终输出**：中间层特征蒸馏（feature distillation）能传递更多信息；
6. **忽视 hard label**：完全用 soft label 会让 Student 失去对 ground truth 的强约束。

## 9. 伪代码

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

## 10. 总结

| 问题 | 知识蒸馏的答案 |
|---|---|
| 大模型太贵 | 蒸馏出小模型，成本 ↓10×–50× |
| 小模型不够强 | 继承 Teacher 暗知识，泛化 ↑ |
| 端侧部署 | VLA 大模型 → 小模型 → Jetson/机器人芯片 |
| 多任务统一 | OPD 在线蒸馏避免分布偏移 |

> 一句话记住：**蒸馏不是把大模型变小，而是把大模型「知道但没说出口的东西」教给小模型。**
