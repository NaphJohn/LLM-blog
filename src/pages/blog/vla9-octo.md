---
title: VLA 解码手记（九）：Octo —— 27M 参数的通用机器人策略，以及"能微调"这件事
description: Octo 是首个完全开源、可高效微调到新传感器 / 新动作空间 / 新机器人本体的通用机器人策略。它用 27M / 93M 参数的 Transformer 主干加扩散式动作头，在 Open X-Embodiment 80 万条轨迹上预训练，靠 Readout Tokens 把"输入"与"输出"解耦，推理 4 步流匹配即可上机。本文拆解它的三段式架构、动作分块与 Flow Matching 损失、微调配方，并与 OpenVLA 的路线做对照。
pubDate: 2026-09-11
series: VLA 解码手记
lang: zh
altLang: en
altHref: /en/blog/vla9-octo
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话定位

**Octo** = 27M / 93M 参数的 Transformer 主干 + 扩散式动作头构成的**通用机器人策略（Generalist Robot Policy, GRP）**，在 Open X-Embodiment 的 80 万条多本体轨迹上预训练，是首个**完全开源、且能高效微调到新传感器 / 新动作空间 / 新机器人本体**的工作（RSS 2024）。

它不追求"参数越大越通用"，而是把通用性拆成四件可以单独替换的零件：

```text
通用性 = 可换的 Tokenizer（吃任意观测）
       + 解耦的 Readout Tokens（换任务不用重训主干）
       + 动作分块（缓解短视规划）
       + 扩散/流匹配动作头（多模态动作分布 + 少步推理）
```

上一篇 `vla2` 讲的 Diffusion Policy 和 Flow Matching 是"发动机"，Octo 是把发动机装上车、并且**允许你换车身**的那套工程方案。

## 1. 为什么需要"通用机器人策略"

2023–2024 年机器人基础模型集中爆发，但每条路线都留着一个硬伤：

| 工作 | 做法 | 硬伤 |
|---|---|---|
| **RT-2** | 55B 通用 VLM 直接输出动作 | 只能接语言指令，不能接目标图像；闭源 |
| **RT-1-X** | 在多任务上训练 | 限定单一本体，换机器人要重训 |
| **Diffusion Policy** | 扩散生成动作 | 单任务；每个新任务重训 |
| **Octo** | 27M/93M + 可替换 tokenizer | ——把上面三件事同时解掉 |

Octo 要解决的是三个具体问题：

1. **同一模型既能接语言指令，也能接目标图像**作为任务描述；
2. **支持多种相机配置 + 本体状态 + 可选力矩观测**（异构输入统一成 token）；
3. **改 tokenizer + 训少量数据就能微调到新机器人**（消费级 GPU、数小时）。

代表成绩：在 WidowX 上零样本性能超过 RT-1-X（当时最强开源 GRP），与 55B 的 RT-2-X 持平或略优——**参数少了三个数量级**。

## 2. 三段式架构：Tokenizer → Transformer Backbone → Diffusion Head

Octo 把机器人控制当成"序列到序列"问题：

<div class="fig">
<svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Octo 三段式架构：异构观测经 Tokenizer 统一为 token，Transformer 主干输出 Readout Tokens，动作头用 Flow Matching 解码出动作块">
  <defs>
    <marker id="arrowOcto" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Octo 架构：输入解耦 / 输出解耦</text>

  <rect x="20" y="48" width="150" height="150" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="70" font-size="12" font-weight="700" fill="#1a1a1a">① Input Tokenizer</text>
  <rect x="34" y="82" width="122" height="26" rx="5" fill="#eff6ff" stroke="#2563eb"/>
  <text x="95" y="99" font-size="11" fill="#1d4ed8" text-anchor="middle">语言 → 1 token</text>
  <rect x="34" y="114" width="122" height="26" rx="5" fill="#eff6ff" stroke="#2563eb"/>
  <text x="95" y="131" font-size="11" fill="#1d4ed8" text-anchor="middle">图像 → patch tokens</text>
  <rect x="34" y="146" width="122" height="26" rx="5" fill="#eff6ff" stroke="#2563eb"/>
  <text x="95" y="163" font-size="11" fill="#1d4ed8" text-anchor="middle">本体状态 → 1 token</text>
  <rect x="34" y="178" width="122" height="0" rx="5" fill="none"/>

  <rect x="200" y="48" width="180" height="150" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="215" y="70" font-size="12" font-weight="700" fill="#047857">② Transformer Backbone</text>
  <rect x="214" y="82" width="152" height="26" rx="5" fill="#d1fae5" stroke="#10b981"/>
  <text x="290" y="99" font-size="11" fill="#047857" text-anchor="middle">K 个 Readout Tokens</text>
  <rect x="214" y="114" width="152" height="52" rx="5" fill="#fff" stroke="#10b981"/>
  <text x="290" y="134" font-size="11" fill="#047857" text-anchor="middle">N 层 MHSA + MLP</text>
  <text x="290" y="152" font-size="10" fill="#047857" text-anchor="middle">read-only 掩码：主 token 不看 readout</text>
  <text x="290" y="180" font-size="10" fill="#6b7280" text-anchor="middle">输出只从 readout 位置 gather</text>

  <rect x="410" y="48" width="250" height="150" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="425" y="70" font-size="12" font-weight="700" fill="#b45309">③ Action Head（Flow Matching）</text>
  <text x="425" y="94" font-size="11" fill="#b45309">噪声 a₀ ~ N(0,I) + 条件 o</text>
  <text x="425" y="116" font-size="11" fill="#b45309">FiLM 注入 o 的全局特征 → 预测速度场 v</text>
  <text x="425" y="138" font-size="11" fill="#b45309">Euler 积分 4–16 步 → 动作块 (T_p 步)</text>
  <text x="425" y="162" font-size="11" fill="#b45309">执行前 T_a 步 → 重规划（receding horizon）</text>
  <text x="425" y="186" font-size="10" fill="#6b7280">可替换为 DDPM（~100 步，慢 27×）</text>

  <line x1="172" y1="122" x2="196" y2="122" stroke="#6b7280" marker-end="url(#arrowOcto)"/>
  <line x1="382" y1="122" x2="406" y2="122" stroke="#6b7280" marker-end="url(#arrowOcto)"/>

  <rect x="20" y="222" width="640" height="158" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="246" font-size="13" font-weight="700" fill="#1a1a1a">为什么这样拆：新本体只需要动两头</text>
  <text x="35" y="272" font-size="12" fill="#444">· 换传感器 / 换相机 → 只改 Tokenizer，主干不动；</text>
  <text x="35" y="294" font-size="12" fill="#444">· 换动作空间 / 换自由度 → 只改 Action Head 的输出维度，主干不动；</text>
  <text x="35" y="316" font-size="12" fill="#444">· 换任务 / 换指令形式 → 只训 Readout Tokens，主干可以冻结；</text>
  <text x="35" y="338" font-size="12" fill="#444">· 于是"微调"从"重训一个模型"变成"训一个适配器"，这才是它敢叫通用策略的底气。</text>
  <text x="35" y="364" font-size="11" fill="#6b7280">对照：OpenVLA 走 7B VLM + LoRA 的路线，参数在 VLM 侧；Octo 走小主干 + 从头训 + readout 的路线。</text>
</svg>
<p class="cap">图：异构观测 → 统一 token → Readout Tokens 摘要 → 流匹配动作头。两头可换，主干可冻结。</p>
</div>

### ① Input Tokenizer：异构观测 → 统一 token 序列

| 输入 | 编码方式 | token 数 |
|---|---|---|
| 语言指令 | Sentence-BERT | 1 个 sentence token |
| 目标图像 / 腕部相机 | 小型 CNN（ResNet 特征） | 64–256 个 patch token |
| 本体状态（proprio） | MLP | 1 个 token |
| 力矩 / 触觉（可选） | MLP | 按自由度 |

### ② Transformer Backbone + Readout Tokens（核心创新）

N 层标准 Transformer（MHSA + MLP），每一层允许 token 互相 attend。关键是在序列最前面拼上 **K 个 Readout Tokens**，并给它们加 **read-only 注意力掩码**——readout 可以看所有输入 token，但输入 token 不看 readout。

于是这 K 个向量就成了整段输入的"摘要"，再交给下游动作头。好处是**输入与输出完全解耦**：新本体只需加 tokenizer + 训 readout，不需要重训主干。

### ③ Action Head：流匹配 / DDPM 二选一

动作头的输入是 readout token（条件 o）+ 噪声 a₀，通过 **FiLM（Feature-wise Linear Modulation）** 把 o 的全局特征注入每一层去噪网络，输出未来 Tp 步的动作轨迹块，执行 Ta 步后重规划。

## 3. 动作分块与 Flow Matching 损失

动作分块（action chunking）是 Octo 的默认姿态：**Tp 是预测视界，Ta 是执行视界（Ta < Tp）**，到期重新预测。这套 receding horizon 范式是 ACT / π0 / Diffusion Policy 的标配。

流匹配的训练目标用一个公式概括：

```text
x_t = (1 - t) * eps + t * a_target        # t ∈ [0,1]，线性插值路径
v_pred = head(x_t, t, o)                  # 预测速度场
L_FM = mean( (v_pred - (a_target - eps))² ) # 速度场回归，目标是 a_target - eps
```

推理时从噪声出发做确定性 ODE 积分，**4–16 步即可**（DDPM 同架构要 ~100 步）。

```python
import torch, torch.nn as nn

class OctoPolicy(nn.Module):
    def __init__(self, backbone, tokenizer, action_head, K, d):
        super().__init__()
        self.tokenizer   = tokenizer     # 语言 / 图像 / 本体 → tokens
        self.backbone    = backbone      # N 层 Transformer
        self.readout     = nn.Parameter(torch.randn(K, d))   # K 个可学习 readout token
        self.action_head = action_head   # Flow Matching / DDPM head

    def encode_obs(self, lang, image, proprio):
        toks = []
        if lang     is not None: toks.append(self.tokenizer.language(lang))
        if image    is not None: toks.append(self.tokenizer.vision(image))
        if proprio is not None: toks.append(self.tokenizer.proprio(proprio))
        r   = self.readout[None].expand(toks[0].size(0), -1, -1)   # [B, K, d]
        seq = torch.cat([r, *toks], dim=1)                          # readout 放最前
        return self.backbone(seq, readout_mask=True)                # 只在 readout 上 gather

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

# 闭环：预测一整块 → 只执行前 T_a 步 → 重新预测
for episode in episodes:
    obs = env.reset()
    for t in range(horizon):
        o_feat = policy.encode_obs(**obs)
        chunk  = policy.predict_chunk(o_feat)
        for k in range(T_a):
            obs = env.step(chunk[:, k])
            if done: break
```

## 4. 微调：怎么把 27M 模型装到新本体上

这是 Octo 区别于"又一个 VLA 复现"的地方。它的微调配方大致是：

```text
冻结：Transformer Backbone + Readout Tokens
训练：新的 Action Head + 新的 Tokenizer 适配器
数据：50–100 条目标域轨迹
硬件：单卡 A6000，数小时
```

训练主干时的配置（论文默认值，供复现参考）：

| 项 | 设置 |
|---|---|
| 数据 | Open X-Embodiment 25 个数据集混合 800k episodes，按数据集大小 + 任务多样性加权 |
| 优化器 | AdamW，weight decay 1e-4 |
| 学习率 | cosine warmup，5000 步升到 1e-4 |
| batch | 256，每张 GPU 跑 1 条 trajectory |
| EMA | decay 0.9999，稳定评估，约 +0.5~1% success |

## 5. 消融：哪些设计真的有用

| 设计选择 | 结论 |
|---|---|
| **动作分块** | Tp=16 时 Stanford Coffee 任务从 0.45 → 0.75（**+30pp**），是缓解"短视规划"的关键 |
| **Readout tokens** | 直接用末端 token 当条件比 readout 差 **约 5pp**；readout 把上下文压成 K 个向量，更适配新任务 |
| **Flow Matching vs DDPM** | 同架构同数据：FM 4 步拿 **0.798**、DDPM 100 步拿 0.816（仅差 2%），但时延 **23 ms vs 635 ms（27.5×）** |
| **模型规模** | 27M → 93M 在 9 个真实机器人上平均 0.62 → 0.72；但**微调后差距缩小到约 0.02**（小模型微调更稳） |
| **目标图像 vs 语言** | WidowX 系列上目标图更优（信息密度大），平均 **+25%** |

最后一行值得记住：**大模型的优势主要体现在零样本，一旦允许微调，小模型追得很近**。这对端侧部署是极好的消息。

## 6. 与 OpenVLA 的路线对照

| 维度 | **Octo** | **OpenVLA** |
|---|---|---|
| 主干 | 27M / 93M，从头训 | 7B VLM（冻结 LLaVA） |
| 适配方式 | 换 Tokenizer + 训 Readout + 换 Action Head | LoRA 微调 |
| 动作解码 | Flow Matching（4 步） | 离散动作 token 自回归 |
| 开源程度 | 完全开源，含预训练 checkpoint | 开源权重 |
| 推理成本 | 消费级 GPU 可实时（<30 ms/块） | 需较大显存 |
| 共同结论 | **通用机器人策略不再需要每个机器人重训** | 同 |

两条路线殊途同归：都验证了"一个策略服务多个本体"这件事成立。差别在于一个是"小模型 + 解耦接口"，一个是"大模型 + 参数高效微调"。

## 7. 对具身智能的含义

1. **算法栈开始同质化**。当策略可以换头微调、主干可冻结，差异化就不再来自"谁的网络设计得更妙"，而转向 **数据规模 + 本体稳定性 + 商业场景**。这对智元 / 宇树 / 优必选这类整机厂的"算法护城河构建速度"是加速器，也是压力。
2. **小模型 + 大数据 = 端侧可行性被验证**。27M 模型 + 4 步 FM 推理时延 < 30 ms，足以支撑 30 Hz 控制。这是"端侧通用 VLA"第一次在**参数 / 时延 / 性能三角中同时达标**。
3. **开源 checkpoint 让国内团队可以直接 take-off**。预训练权重公开，工程落地的窗口期被压缩，比拼的是谁的场景数据闭环更快——对应 `mm4`、`vla4` 里提到的国内玩家竞争格局。

## 8. 常见坑

1. **Readout 掩码写错**：如果让输入 token 反向 attend readout，信息会从输出泄漏回输入，微调时表现虚高、换任务立刻崩。
2. **Ta 设得太大**：执行步数越接近 Tp，重规划频率越低，越容易在动态场景里"撞上"过期动作块。经验值 Ta ≈ Tp/2。
3. **FM 学习率比 DDPM 敏感**：调高容易震荡，微调场景建议从更小的学习率起手。
4. **只换 Action Head 不换 Tokenizer**：新本体的相机布局一变，token 语义就错了，必须先对齐 tokenizer。

## 9. 总结

| 维度 | Octo 的答案 |
|---|---|
| 通用性怎么来 | 不是靠参数大，是靠**接口解耦**（tokenizer / readout / head 三段可换） |
| 为什么能微调 | 主干冻结 + 只训适配层，50–100 条数据、数小时 |
| 为什么快 | Flow Matching 4 步，比 DDPM 快 27.5×，只掉 2% 性能 |
| 为什么要分块 | 动作分块把"短视规划"从 0.45 提到 0.75 |
| 投资/工程含义 | 算法同质化 → 竞争转向数据与场景；小模型 + 端侧推理打开落地窗口 |

> 一句话记住：**Octo 证明的不是"小模型能打"，而是"把接口切开之后，通用机器人策略变成了一个可维护的工程问题"。**

---

**系列导航**：上一篇 [`vla8-dreamer-v3`](/blog/vla8-dreamer-v3) 讲世界模型 RL；动作生成原理见 [`vla2-action-generation`](/blog/vla2-action-generation)；[`pp2-diffusion-policy`](/blog/pp2-diffusion-policy) 是 Diffusion Policy 的精读原文笔记。
