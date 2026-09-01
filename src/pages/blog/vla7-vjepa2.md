---
title: VLA 解码手记（七）：V-JEPA 2——视频自监督世界模型与零样本机器人规划
description: Meta FAIR 的 V-JEPA 2 从 100 万小时无标注视频学一个潜在空间世界模型，仅用 62 小时机器人数据做动作条件微调，就能在 Franka 上零样本操控（65–80% 成功率）。本文讲清「潜在空间预测而非像素重建」的核心思想、两阶段训练、Encoder+Predictor+规划头结构，并附架构图与 PyTorch 风格伪代码。
pubDate: 2026-08-26
series: VLA 解码手记
lang: zh
altLang: en
altHref: /en/blog/vla7-vjepa2
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话定位

> 从 **100 万小时无标注视频**里学一个「潜在空间世界模型」，再用 **62 小时机器人数据**做动作条件微调，实现 **零样本机器人操控（65–80% 成功率）**。

V-JEPA 2（Assran et al., Meta FAIR, 2025-06, arXiv:2506.09985，1.2B 参数）走的是 LeCun 的 JEPA 哲学：智能体应该先在脑内预演「做这件事世界会怎么变」，再去行动——而不是像 RT-2 / OpenVLA / π0 那样靠「互联网图文 + 大量遥操数据」硬训。

## 1. 核心思想：潜在空间预测，不是像素重建

给定视频帧序列 `(x_1, x_2, ..., x_T)`：

```text
# 1) 编码：把每一帧映射到潜在向量空间
  z_t = Encoder(x_t)                 # Vision Transformer (ViT-L/H/G)

# 2) 预测：用过去潜在向量 + 动作条件 预测未来潜在向量
  z_{t+1} = Predictor(z_{t-k...t}, a_t)   # a_t 可选，无动作时为视频自监督

# 3) 损失：只对「未来帧的潜在表示」做 L2 / Smooth-L1，
#    不重建像素；目标也由 EMA teacher encoder 产生。
  L = || z_{t+1} - stop_grad(Encoder_target(x_{t+1})) ||^2
```

符号：`x_t` = t 时刻视频帧；`z_t` = 潜在向量；`Encoder` = ViT；`a_t` = 可选动作；下标 `_target` = EMA 教师编码器。

## 2. 架构图：Encoder + Predictor + 规划回环

<div class="fig">
<svg viewBox="0 0 680 430" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="V-JEPA 2 架构：Encoder → 潜在向量 → Predictor + 动作 → 预测未来潜在 ↔ EMA 目标 → L2 损失，以及 latent MPC 规划头">
  <rect x="0" y="0" width="680" height="430" fill="none"/>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">V-JEPA 2 架构：潜在空间预测 + latent MPC 规划</text>

  <!-- Input frames -->
  <rect x="40" y="56" width="190" height="42" rx="6" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="135" y="82" font-size="12.5" fill="#b45309" text-anchor="middle">视频帧 x₁ … x_T</text>

  <!-- Encoder -->
  <rect x="300" y="56" width="170" height="42" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="385" y="82" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">Encoder (ViT)</text>
  <line x1="230" y1="77" x2="298" y2="77" stroke="#16a34a" stroke-width="2" marker-end="url(#vg)"/>

  <!-- z_t -->
  <text x="540" y="73" font-size="13" font-weight="700" fill="#1a1a1a">z_t</text>
  <text x="540" y="90" font-size="10" fill="#6b7280">潜在向量</text>
  <line x1="470" y1="77" x2="518" y2="77" stroke="#16a34a" stroke-width="2" marker-end="url(#vg)"/>

  <!-- EMA teacher -->
  <rect x="40" y="128" width="150" height="40" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="115" y="153" font-size="11.5" fill="#92400e" text-anchor="middle">帧 x_{t+1}</text>
  <rect x="300" y="128" width="170" height="40" rx="6" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="5 3"/>
  <text x="385" y="153" font-size="11.5" fill="#475569" text-anchor="middle">EMA Teacher Encoder</text>
  <line x1="190" y1="148" x2="298" y2="148" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#vg)"/>
  <text x="540" y="153" font-size="12.5" font-weight="700" fill="#475569">ẑ (stop_grad)</text>

  <!-- Predictor -->
  <rect x="40" y="216" width="150" height="34" rx="6" fill="#faf5ff" stroke="#a855f7"/>
  <text x="115" y="238" font-size="11" fill="#7e22ce" text-anchor="middle">动作 a_t (可选)</text>
  <rect x="280" y="206" width="210" height="46" rx="6" fill="#ecfdf5" stroke="#10b981"/>
  <text x="385" y="226" font-size="12" font-weight="700" fill="#047857" text-anchor="middle">Predictor (Transformer)</text>
  <text x="385" y="244" font-size="10.5" fill="#047857" text-anchor="middle">输入：过去潜在 z + 动作 a_t</text>
  <line x1="190" y1="233" x2="278" y2="229" stroke="#a855f7" stroke-width="1.5" marker-end="url(#vg)"/>
  <line x1="540" y1="97" x2="540" y2="196" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="4 3"/>
  <line x1="540" y1="196" x2="492" y2="206" stroke="#16a34a" stroke-width="1.5" marker-end="url(#vg)"/>
  <text x="560" y="234" font-size="12.5" font-weight="700" fill="#047857">ẑ_{t+1}</text>

  <!-- Loss -->
  <rect x="300" y="292" width="240" height="40" rx="6" fill="#fef2f2" stroke="#ef4444"/>
  <text x="420" y="316" font-size="12" fill="#b91c1c" text-anchor="middle">L = ‖ ẑ_{t+1} − sg(z) ‖²</text>
  <line x1="560" y1="252" x2="530" y2="290" stroke="#ef4444" stroke-width="1.5" marker-end="url(#vr)"/>
  <line x1="560" y1="171" x2="545" y2="290" stroke="#ef4444" stroke-width="1.5" marker-end="url(#vr)"/>

  <!-- Planning head -->
  <rect x="40" y="366" width="320" height="48" rx="6" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="200" y="388" font-size="11.5" font-weight="700" fill="#b45309" text-anchor="middle">规划头：latent MPC 想象 K 步</text>
  <text x="200" y="405" font-size="10.5" fill="#b45309" text-anchor="middle">选最接近「目标图像 embedding」的动作链</text>
  <line x1="360" y1="390" x2="420" y2="390" stroke="#f59e0b" stroke-width="2" marker-end="url(#vg)"/>
  <text x="470" y="386" font-size="12" font-weight="700" fill="#1a1a1a">动作序列</text>
  <text x="470" y="404" font-size="11" fill="#6b7280">→ 下发机器人</text>

  <defs>
    <marker id="vg" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#16a34a"/></marker>
    <marker id="vr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#ef4444"/></marker>
  </defs>
</svg>
  <p class="cap">图：V-JEPA 2 的训练闭环（Encoder + Predictor + EMA 教师 + L2 潜在损失）与推理闭环（latent MPC 规划头）。预测目标是潜在空间里的未来帧，而不是像素。</p>
</div>

## 3. 两阶段训练

- **阶段 1（无动作自监督预训练）**：100 万小时视频 + 100 万张图像，完全无人工标注，让 Encoder + Predictor 学会「潜在空间里的物理动力学」。
- **阶段 2（V-JEPA 2-AC 动作条件微调）**：仅用 **62 小时 Droid 机器人数据集**微调 Predictor，不写任何 task-specific 奖励函数，让模型学会「在动作 a 下未来长啥样」。

## 4. 关键模块

- **Encoder**：ViT-L/H/G，标准 ViT 块（LayerNorm + MHSA + MLP），把 16×16 patch 编码为潜在向量；EMA 教师编码器提供稳定预测目标。
- **Predictor**：轻量 Transformer，输入是过去帧潜在向量的位置拼接 + 可选动作 token，输出下一帧的潜在向量。
- **规划头（planning head）**：给定当前观测 + 候选动作序列，在潜在空间「想象」未来 K 步，挑出与目标图像 embedding 最近的那条动作链，再用模型预测控制（MPC）下发。
- **目标表示**：用一张**目标图像**作为任务指令，让机器人不依赖自然语言就能「看出」要做什么。

## 5. PyTorch 风格伪代码

```python
import torch
import torch.nn as nn

class VJEPA2(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder               # ViT-L/H/G
        self.target_encoder = encoder        # EMA 教师，stop_grad
        self.predictor = predictor           # Transformer
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, frames, actions=None):
        # frames: (B, T, 3, H, W) 视频片段
        B, T = frames.shape[:2]
        feats = self.encoder(frames.flatten(0, 1))        # (B*T, D)
        feats = feats.unflatten(0, (B, T))                # (B, T, D)
        with torch.no_grad():
            targets = self.target_encoder(frames[:, 1:].flatten(0, 1))
            targets = targets.unflatten(0, (B, T - 1))
        preds = self.predictor(feats[:, :-1], actions)   # (B, T-1, D)
        return preds, targets  # L2 / Smooth-L1 在 D 维做

# 规划：Model-Predictive Control in latent space
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
    return best  # 下发给机器人的动作序列
```

## 6. 训练 / 优化要点

- **EMA 教师编码器**：目标编码器用 encoder 权重的指数滑动平均更新（momentum ≈ 0.99→0.999），避免表征塌缩。
- **Masked latent prediction**：仿 MAE 思路对 patch 随机 mask，让预测器只补全被遮掉的部分，降低计算量同时学更鲁棒表示。
- **分辨率梯度提升**：先在 224×224 训，再 fine-tune 到 384×384。
- **不重建像素**：loss 只落在潜在空间 → 训练目标与人类关心的高层语义（物体运动、因果）天然对齐，避免浪费容量在纹理/颜色细节。

## 7. 复杂度与消融

- Encoder ViT-H/16 在 16 帧 384² 输入下，推理速度是英伟达 Cosmos 的 **30 倍**（Meta 官方口径）。
- Something-Something v2 上 **77.3%** top-1，超过同规模有监督模型；PerceptionTest 84.0、TempCompass 76.9。
- 零样本机器人 pick-and-place **65–80%** 成功率（在 Droid 见过、在 Franka 全新桌面未见的物体）。
- 消融要点：移除动作条件 → 规划完全失败；移除 EMA → 表征塌缩；预测空间改成像素 → 性能下降 + 算力翻 5 倍以上。

## 8. 总结与影响

V-JEPA 2 把「世界模型」从论文概念推进到可直接挂到真机的工程方案，三层意义：

1. **路线意义**——证明不依赖像素重建、不依赖大规模遥操也能让机器人「看懂」环境；
2. **数据意义**——100 万小时视频成本远低于 62 小时遥操数据的边际成本；
3. **生态意义**——V-JEPA 2 全套权重开源（CC-BY），叠加 NVIDIA Cosmos 闭源 + 30× 速度差，世界模型路线 Meta 已经领跑。

对具身智能主线：利好上游世界模型 / 视频理解 / VLA 三模态（具身「大脑」派）、利好视频-多模态预训练基础设施（视频编码、潜在向量压缩、动作 token 化）。
