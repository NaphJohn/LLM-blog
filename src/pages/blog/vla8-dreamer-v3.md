---
title: VLA 解码手记（八）：Dreamer V3 —— 一套超参横扫 50+ 域的世界模型 RL
description: Dreamer V3 是首个用单一固定超参数在 150+ 任务上达到或超越专用 SOTA 的世界模型强化学习算法。它用 RSSM 学习环境动力学、用 symlog 统一奖励尺度、在 imagination 中 rollout 策略。本文拆解其架构、训练循环与具身智能落地意义。
pubDate: 2026-08-26
series: VLA 解码手记
lang: zh
altLang: en
altHref: /en/blog/vla8-dreamer-v3
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话定位

**Dreamer V3** = 用大模型在「脑子里做梦」的方式学策略，一套固定超参数横扫 Atari、DMLab、Crafter、Minecraft、机械臂控制等 150+ 任务。

它是世界模型（World Model）路线从「调参艺术」走向「工程系统」的关键节点，也是后来 Genie、Cosmos、GR00T-Dreams 等物理世界模型的技术源头之一。

## 1. 为什么需要世界模型？

传统 RL 的问题：

```text
真实环境交互 → 成本高、样本效率低、真机试错危险
```

世界模型的思路：

```text
先学一个环境的内部模型 → 在 imagination 里 rollout → 只在必要时用真实交互验证
```

对机器人来说，这相当于让模型在「梦境」里摔无数次跤，而不是在真实产线上摔。

## 2. RSSM：把世界建模成循环状态空间

Dreamer V3 的核心是世界模型 **RSSM（Recurrent State-Space Model）**，状态拆成两部分：

| 状态 | 含义 | 更新方式 |
|---|---|---|
| **h_t** | 确定性循环状态 | GRU 递归更新 |
| **z_t** | 离散随机状态 | 分类分布（Categorical）|

```text
h_t = GRU(h_{t-1}, a_{t-1}, z_{t-1})           # 确定性路径
z_t ~ p(z_t | h_t) = Categorical(NN_prior(h_t)) # 先验：不用观测，纯预测
z_t ~ q(z_t | h_t, o_t) = Categorical(NN_post(h_t, o_t)) # 后验：用观测修正
```

训练时，后验 q 用来更新模型；想象 rollout 时，只用先验 p（因为看不到未来观测）。

## 3. 训练目标：重建 + 奖励 + 终止 + KL

世界模型同时预测三件事：

```text
o_hat = Decoder(h_t, z_t)       # 重建观测
r_hat = RewardHead(h_t, z_t)    # 预测奖励
c_hat = ContinueHead(h_t, z_t)  # 预测是否终止
L_WM = L_recon + L_reward + L_continue + beta * KL(q || p)
```

## 4. symlog：让一套超参跨域通用的关键

不同任务的奖励尺度天差地别：

- Atari：0 ~ 1000
- 控制任务：-1 ~ 1
- Minecraft：稀疏、延迟奖励

Dreamer V3 用 **symlog** 变换统一尺度：

```text
symlog(x) = sign(x) * ln(1 + |x|)
symexp(x) = sign(x) * (exp(|x|) - 1)   # 反变换
```

这样 reward head 和 value head 都预测压缩后的值，Critic 训练更稳定，同一套损失权重能适应所有域。

## 5. Imagination Rollout：在梦里训练 Actor-Critic

<div class="fig">
<svg viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dreamer V3 训练循环：世界模型 + Actor-Critic 在 imagination 中训练">
  <defs>
    <marker id="arrow2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Dreamer V3 训练循环</text>

  <!-- real trajectory -->
  <rect x="20" y="55" width="140" height="70" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="90" y="82" font-size="12" fill="#047857" text-anchor="middle">真实交互轨迹</text>
  <text x="90" y="102" font-size="11" fill="#047857" text-anchor="middle">o_t, a_t, r_t</text>

  <!-- world model -->
  <rect x="200" y="55" width="160" height="70" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="280" y="82" font-size="12" fill="#1d4ed8" text-anchor="middle">RSSM 世界模型</text>
  <text x="280" y="102" font-size="11" fill="#1d4ed8" text-anchor="middle">学 h_t, z_t, ô, r̂, ĉ</text>

  <!-- imagination -->
  <rect x="400" y="55" width="140" height="70" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="470" y="82" font-size="12" fill="#b45309" text-anchor="middle">Imagination</text>
  <text x="470" y="102" font-size="11" fill="#b45309" text-anchor="middle">h,z 想象 rollout</text>

  <!-- actor critic -->
  <rect x="580" y="55" width="80" height="70" rx="8" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="620" y="82" font-size="12" fill="#6b21a8" text-anchor="middle">Actor</text>
  <text x="620" y="102" font-size="11" fill="#6b21a8" text-anchor="middle">+ Critic</text>

  <!-- arrows -->
  <line x1="160" y1="90" x2="198" y2="90" stroke="#6b7280" marker-end="url(#arrow2)"/>
  <line x1="360" y1="90" x2="398" y2="90" stroke="#6b7280" marker-end="url(#arrow2)"/>
  <line x1="540" y1="90" x2="578" y2="90" stroke="#6b7280" marker-end="url(#arrow2)"/>

  <!-- actor-critic detail -->
  <rect x="20" y="160" width="640" height="170" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="185" font-size="13" font-weight="700" fill="#1a1a1a">在 imagination 中优化策略</text>
  <text x="35" y="210" font-size="12" fill="#444">1. 从真实轨迹最后一步 (h_T, z_T) 出发，用 Actor 生成动作 a_T；</text>
  <text x="35" y="230" font-size="12" fill="#444">2. RSSM 先验预测下一步 (h_{T+1}, z_{T+1}) 和奖励 r̂_{T+1}；</text>
  <text x="35" y="250" font-size="12" fill="#444">3. 重复 H 步，得到 imagined trajectory；</text>
  <text x="35" y="270" font-size="12" fill="#444">4. Critic 估计 lambda-return，Actor 最大化 return + entropy 正则。</text>
  <text x="35" y="300" font-size="12" fill="#6b7280">损失：L_actor = -E[ symlog(V_λ) ] + λ_entropy * H(a)；L_critic = MSE(symlog(v̂), symlog(V_λ))</text>
</svg>
<p class="cap">图：Dreamer V3 在真实数据上训练世界模型，再在想象轨迹上训练 Actor-Critic。</p>
</div>

## 6. 关键超参为什么能固定？

Dreamer V3 之前，每个域都需要单独调：学习率、折扣因子、奖励缩放、KL 权重等。V3 能固定超参，主要靠四点：

1. **symlog/symexp**：统一奖励与 value 尺度；
2. **离散分类状态**：32 类 × 32 组，比连续高斯更稳定；
3. **KL 平衡**：动态调整先验/后验权重，防止模型崩塌；
4. **归一化与初始化**：观测、奖励、梯度都做了归一化，降低对任务统计量的敏感。

## 7. 与具身智能的关系

Dreamer V3 直接启发了：

- **Genie**（Google）：从视频中学可交互世界模型；
- **Cosmos**（NVIDIA）：物理世界基础模型；
- **GR00T-Dreams**（NVIDIA 人形机器人）：用世界模型生成合成训练数据；
- **智元 WITA-Omni**、**宇树 GR00T-Dreams 路线**：国内头部也在用「世界模型生成数据」缓解真机遥操数据不足。

核心逻辑：

```text
世界模型生成 synthetic rollout → 低成本扩大训练数据 → VLA 在真机上更稳
```

这是解决具身智能「数据饥渴」的关键路径之一。

## 8. 简化版 PyTorch 训练循环

```python
for batch in dataloader:
    # 1. 真实轨迹编码
    h, z = rssm.observe(batch.obs, batch.act)

    # 2. 世界模型损失
    recon = mse(decoder(h, z), batch.obs)
    reward = mse(reward_head(h, z), symlog(batch.reward))
    cont = bce(continue_head(h, z), batch.cont)
    kl = kl_divergence(rssm.posterior(h, batch.obs), rssm.prior(h))
    loss_wm = recon + reward + cont + 0.1 * kl

    # 3. 想象 rollout
    h_imag, z_imag, a_imag, r_imag = rssm.imagine(h[-1], z[-1], actor, horizon=15)

    # 4. Actor-Critic
    values = critic(h_imag, z_imag)
    returns = lambda_return(r_imag, values, gamma=0.997, lambda_=0.95)
    loss_actor = -symlog(returns).mean() + 1e-4 * entropy(a_imag)
    loss_critic = mse(symlog(values.detach()), symlog(returns))

    (loss_wm + loss_actor + loss_critic).backward()
    optimizer.step()
```

## 9. 总结

| 维度 | Dreamer V3 的突破 |
|---|---|
| 样本效率 | 在 imagination 中 rollout，减少真实交互 |
| 跨域泛化 | 固定超参适配 150+ 不同任务 |
| 状态表示 | 离散分类 + GRU 循环，稳定可扩展 |
| 奖励处理 | symlog 统一尺度，解决多域差异 |
| 具身影响 | 成为 VLA + World Model 融合路线的底座 |

> 一句话记住：**Dreamer V3 让模型学会「做梦」，再用梦里的经验指导现实行动。** 对机器人来说，这等于先在虚拟世界里摔够跤，再上真机。
