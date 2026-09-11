---
title: 强化学习训练手记（一）：分布式 RL 训练 —— PPO / GRPO 怎么从单卡跑到多机
description: 分布式 RL 训练把 Actor / Critic / Reference / Reward 四个角色解耦到不同 GPU 池，再用 vLLM 或 SGLang 承担 Rollout、用 NCCL 把训练侧权重同步回推理引擎，把 RLHF / RLAIF 从"一锅炖"变成流水线。本文讲清四角色架构、一个 PPO step 的六阶段、PPO 与 GRPO 的损失公式、Weight Sync 的两条路径与实测提速，以及和机器人 VLA 偏好对齐的关系。
pubDate: 2026-09-11
series: 强化学习训练手记
lang: zh
altLang: en
altHref: /en/blog/rl1-distributed-rl-training
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话定位

**分布式 RL 训练** = 用 Ray 做编排，把 **Actor（策略）/ Critic（价值）/ Reference（参考模型，算 KL）/ Reward（打分）** 四个角色拆到不同 GPU 池，中间插一个 **vLLM 或 SGLang 的 Rollout Engine** 专门负责生成，再用高速 **Weight Sync** 把参数从训练侧广播回推理侧——这是 RLHF、RLAIF、偏好微调、机器人 VLA 偏好对齐共同的**工程基座**。

它不是什么新算法，而是把"一个模型在一张卡上又当演员又当裁判"这件蠢事彻底拆开。

## 1. 为什么 RLHF 不能"一锅炖"

PPO-RLHF 的教科书实现（TRL / HF Trainer 早期版本）是单进程顺序跑：

```text
同一份参数 fp 前向生成 → reward 打分 → 同一份参数反向更新
```

这在 <1B 模型上还能扛，7B 以上必须拆，原因有三个浪费：

1. **同一份参数被当成两个角色用**：既是 Actor（要梯度），又是 Reference（要冻结算 KL）。两份都要常驻显存，浪费一倍。
2. **推理与训练抢同一组 SM**：生成 256 个 token 的 response 是纯推理负载（吃带宽、batch 越大越好），梯度下降是计算负载（吃算力）。硬塞在一起，两边都不在最佳工作点。
3. **时间分布极度不均**：一个 PPO step 里 **rollout（生成）占大约 80% 的时间**，DeepSpeed 训练只占 20%。用同一批卡顺序做，等于让 80% 的卡在训练阶段闲着。

一句话：**rollout 和训练是两种完全不同的负载，必须异步。**

## 2. 四角色解耦 + vLLM Rollout Engine

<div class="fig">
<svg viewBox="0 0 680 390" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="分布式 RL 训练架构：Ray 编排 Actor、Critic、Reference、Reward 四个 GPU 池，vLLM 或 SGLang 负责 rollout，Weight Sync 把训练侧参数广播回推理引擎">
  <defs>
    <marker id="arrowRL" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">分布式 RL 训练：四角色解耦 + 独立 Rollout 引擎</text>

  <rect x="20" y="46" width="150" height="112" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="35" y="68" font-size="12" font-weight="700" fill="#b45309">vLLM / SGLang</text>
  <text x="35" y="88" font-size="11" fill="#b45309">Rollout Engine</text>
  <text x="35" y="108" font-size="11" fill="#b45309">PagedAttention 生成</text>
  <text x="35" y="128" font-size="11" fill="#b45309">占一个 PPO step 的 ~80% 时间</text>
  <text x="35" y="148" font-size="10" fill="#6b7280">切换阶段时 enable_sleep 让出显存</text>

  <rect x="196" y="46" width="150" height="112" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="211" y="68" font-size="12" font-weight="700" fill="#1d4ed8">Actor GPUs</text>
  <text x="211" y="88" font-size="11" fill="#1d4ed8">主策略，要梯度</text>
  <text x="211" y="108" font-size="11" fill="#1d4ed8">DeepSpeed ZeRO-3 / FSDP</text>
  <text x="211" y="128" font-size="11" fill="#1d4ed8">PPO 或 GRPO 更新</text>

  <rect x="372" y="46" width="140" height="112" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="387" y="68" font-size="12" font-weight="700" fill="#047857">Critic GPUs</text>
  <text x="387" y="88" font-size="11" fill="#047857">value head</text>
  <text x="387" y="108" font-size="11" fill="#047857">GAE 优势估计</text>
  <text x="387" y="128" font-size="10" fill="#6b7280">GRPO 可整个去掉</text>

  <rect x="538" y="46" width="122" height="52" rx="8" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="599" y="68" font-size="12" font-weight="700" fill="#6b21a8">Reference</text>
  <text x="599" y="86" font-size="11" fill="#6b21a8">冻结，算 logp</text>

  <rect x="538" y="106" width="122" height="52" rx="8" fill="#fef2f2" stroke="#e24b4a"/>
  <text x="599" y="128" font-size="12" font-weight="700" fill="#a32d2d">Reward</text>
  <text x="599" y="146" font-size="11" fill="#a32d2d">RM 或规则校验</text>

  <line x1="172" y1="102" x2="192" y2="102" stroke="#6b7280" marker-end="url(#arrowRL)"/>
  <line x1="348" y1="102" x2="368" y2="102" stroke="#6b7280" marker-end="url(#arrowRL)"/>
  <line x1="514" y1="102" x2="534" y2="102" stroke="#6b7280" marker-end="url(#arrowRL)"/>

  <rect x="20" y="182" width="640" height="86" rx="8" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="35" y="204" font-size="13" font-weight="700" fill="#1a1a1a">Weight Sync：训练侧 → 推理侧（不能省的一步）</text>
  <text x="35" y="226" font-size="12" fill="#444">① Ray Object Store：GPU → CPU → 序列化 → CPU → GPU，通信量 O(N × 参数量)，配置简单但慢；</text>
  <text x="35" y="246" font-size="12" fill="#444">② NCCL broadcast：GPU → GPU 直通，用 StatelessProcessGroup 建 side-channel 一把广播，零拷贝。</text>
  <text x="35" y="262" font-size="11" fill="#6b7280">OpenRLHF、veRL、NeMo-RL、Miles 走的都是第二条。</text>

  <rect x="20" y="284" width="640" height="86" rx="8" fill="#fef2f2" stroke="#e24b4a"/>
  <text x="35" y="306" font-size="13" font-weight="700" fill="#a32d2d">不同步会怎样：policy 不一致</text>
  <text x="35" y="328" font-size="12" fill="#444">Actor 更新完参数却没广播回 vLLM，推理引擎还在用上一版权重生成 rollout，</text>
  <text x="35" y="348" font-size="12" fill="#444">于是"训练用的策略"和"采样用的策略"不是同一个，importance ratio 失真，PPO 直接不收敛。</text>
</svg>
<p class="cap">图：四个角色各占独立 GPU 池，靠 Ray Object Store 共享 experience buffer；每轮更新后必须把权重广播回推理引擎。</p>
</div>

这个结构没有任何一行是"算法创新"，全是工程。它要回答的问题是：**怎么让 80% 的时间花在最有价值的那件事（生成）上，而不浪费卡。**

## 3. 一个 PPO step 的六阶段流水线

```text
① Rollout generation        : vLLM 生成 response（最长的一段）
② Reward scoring            : RM 或规则校验器打分
③ Reference forward + KL    : 冻结模型对每条样本算 logp_ref
④ GAE advantage             : Critic 估计优势
⑤ Critic / Policy gradient  : 反向更新
⑥ Weight sync back to vLLM  : 参数广播回推理引擎，进入下一轮
```

真正的难点在 ③ 和 ⑥：③ 决定了你需要多少显存（Reference 的权重和 KV），⑥ 决定了你的流水线会不会"卡在等同步"。

## 4. PPO 与 GRPO 的损失

PPO 的核心四行：

```text
A_t      = R_t - V(s_t)                                  # 优势（实际用 GAE）
rho_t    = pi_theta(a_t|s_t) / pi_old(a_t|s_t)            # importance ratio，必须 >= 0
L_CLIP   = E[ min( rho_t*A_t, clip(rho_t, 1-eps, 1+eps)*A_t ) ]
L_KL     = -beta * E[ log pi_theta(a_t) - log pi_ref(a_t) ]
L_PPO    = L_CLIP - c1 * L_VF + c2 * L_KL
```

其中 `eps` 默认 0.2；`beta` 随实测 KL 自适应上调（KL 大了说明漂移太多，要拉回来）；`c1`、`c2` 是 value 与 KL 的系数；Reference 在算 `logp_ref` 时必须处于冻结状态。

**GRPO 的价值在于把 Critic 整个删掉**：不训 value head，而是对同一个 prompt 采样一组（group）答案，用组内相对优劣当优势：

```text
A_i = (r_i - mean(r_group)) / std(r_group)      # 组内标准化，替代 Critic
```

少一整个 GPU 池、少一份显存、少一个训练目标。代价是 **group size 要 >= 4**，否则组内方差估计不稳。

## 5. Weight Sync：两条路径

```python
def sync_weights_to_vllm(actor, vllm):
    for name, param in actor.named_parameters():
        if param.requires_grad:
            # 走 side-channel NCCL process group，GPU -> GPU 零拷贝广播
            vllm.collective_rpc("load_weights", param.data)
```

| 路径 | 机制 | 评价 |
|---|---|---|
| **Ray Object Store** | GPU → CPU → 序列化 → CPU → GPU | 通信量 O(N × 参数量)，好在配置简单 |
| **NCCL broadcast** | GPU → GPU 直通，广播 tensor | 所有 serious 框架的默认选择 |

vLLM 0.7+ 暴露了 `WorkerExtension` 与 `collective_rpc`，训练侧用 `StatelessProcessGroup` 建一条旁路 NCCL 组，即可做到零拷贝广播。

## 6. 实测收益

| 框架 | 结果 |
|---|---|
| **OpenRLHF** | 70B 模型 RLHF 全程 PPO 收敛，相对 TRL 原版约 **3× 提速** |
| **veRL**（字节 Seed） | Qwen2.5-32B 上 PPO RLHF 一个迭代从 1.5 小时降到 **25 分钟** |
| **DeepSeek V4** 自研 CANN-GRPO | KL 校正与奖励建模融合进单次前向，RLHF 迭代周期 **7 天 → 19 小时**，SWE-bench 42.1 → 58.2（+38.2%） |
| **DeepSeek V4 + Miles** | MegaMoE 把通信与计算融成单 GPU kernel，rollout 路径 **1.96× 提速** |
| **AReaL**（蚂蚁） | 完全异步 rollout + replay buffer，70B 推理模型 RL 训练利用率 **>80%** |
| **NeMo-RL**（NVIDIA） | 70B+ 全栈 RLHF 训练，GB300 上的 RTX-only pipeline |

把这些放在一起看，能读出一个趋势：**RL 训练的效率提升，绝大部分来自"把 rollout 和训练解耦"这一件事，而不是新的损失函数。**

## 7. 四个常见坑

1. **Sync gap（同步空窗）**：Weight sync 太久，就会训练用 v1 权重、推理用 v2 权重，policy gradient 直接失真。建议每个 mini-batch 后立即同步。
2. **Actor 显存爆炸**：Reference 不能 unload 到 CPU，否则算 KL 时 I/O 阻塞，整条流水线等它。
3. **vLLM 不 sleep**：切换 rollout / train 阶段时要把 vLLM engine `enable_sleep`，把显存让给训练，否则 OOM。
4. **GRPO 组内方差爆炸**：group size < 4 时组内优势估计方差过大，训练抖动。要么加大 group，要么退回 PPO + Critic。

## 8. 与机器人 VLA 的关系

这套基础设施不止服务文本 RLHF，机器人策略的 **RLAIF（用 reward model 做偏好对齐）** 走的是同一套：

```text
VLA 生成轨迹 → reward model 打分 → 组内相对优劣（GRPO）→ 更新策略 → 同步回推理侧
```

- Octo、π0 都已支持在自定义 reward model 上做 RLAIF；
- **GRPO 是 VLA 微调的高效算法参考**：去掉 Critic 后，27M 的 Octo-Small 微调时间可从 2 小时压到 **约 20 分钟**；
- 对**工厂侧的多场景并行训练**更是刚需：白天训练 + 推理跑机器人，晚上集中 rollout 收数据，同一组 64 张卡做到"一鱼多吃"。这套用法正是 OpenRLHF / veRL 的主线场景，机器人公司可以白嫖大部分 boilerplate。

## 9. 总结

| 维度 | 答案 |
|---|---|
| 要解决什么 | rollout 与训练是两种负载，硬塞在一起浪费 80% 的卡 |
| 怎么解决 | 四角色解耦 + 独立 Rollout Engine + NCCL Weight Sync |
| 最大收益来自 | 解耦本身，而不是新损失函数（veRL 1.5h → 25min） |
| 要不要 Critic | 可选。GRPO 去掉 Critic 换更简单，代价是 group size >= 4 |
| 与 VLA 的关系 | RLAIF 与文本 RLHF 共用同一套基础设施，GRPO 可直接用于 VLA 微调 |

> 一句话记住：**RLHF 的瓶颈从来不在算法，而在"让生成和训练各就各位"。分布式 RL 训练做的就是把这件事工程化。**

---

**系列导航**：本篇是本系列开篇。相关背景见 [`fa4-deepseek-v4`](/blog/fa4-deepseek-v4)（V4 的训练与压缩）与 [`sys8-parallel-strategies-pd-disaggregation`](/blog/sys8-parallel-strategies-pd-disaggregation)（并行切分与 PD 分离）。
