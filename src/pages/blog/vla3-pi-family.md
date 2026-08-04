---
title: VLA 解码手记（三）：π 系列演进——Physical Intelligence 的 VLA 路线
description: 从 π0 到 π0.7，Physical Intelligence 如何把 VLA 一步步推到"超越专精策略"。本文按时间线拆解 PaliGemma/ Gemma3 底座、Flow Matching 50Hz、跨本体、RL 专精，以及 π0.7 的记忆与组合泛化。
pubDate: 2026-08-04
series: VLA 解码手记
lang: zh
altLang: en
altHref: /en/blog/vla3-pi-family
layout: ../../layouts/BlogPost.astro
---

## 1. π0：把 VLA 做成"通用接口"（2024-10）

Physical Intelligence 的 **π0** 定了基线：

- **底座**：PaliGemma（约 3B VLM）+ 一个独立的**动作专家（action expert）**，动作专家用 **Flow Matching** 生成连续动作，输出 **50Hz** 高频控制。
- **数据**：在 OXE + 自有演示上训练，覆盖 **7 种机器人平台、68 项任务**。
- **统一接口**：语言/图像进，连续动作出。证明"一个通用 VLA 能在多本体上干活"。

## 2. π0-Fast：离散化提速

在 π0 上加**动作离散化 + 自回归**，换取推理速度。是"连续 vs 离散"两条路线的工程折中，适合延迟敏感场景。

## 3. π0.5：跨本体 + 任务分解（2025-04）

- **跨本体协同训练（cross-embodiment co-training）**：多台不同机器人共享一个"通用大脑"，学到与硬件无关的操作常识。
- **任务分解（task decomposition）**：高层把复杂指令拆成子目标，低层策略依次执行——让长时程任务可控。

## 4. π0.6*：RL 专精（RECAP）

用**强化学习**在已训 VLA 上做专精打磨（RECAP 思路），把特定任务成功率再往上推。是"通用预训练 + 专精后训练"的典型后半段。

## 5. π0.7：记忆 + 组合泛化（2026-04）

π0.7 是系列目前的顶点，几个关键升级：

- **底座升级**：Gemma3 4B + 860M 动作专家。
- **MEM 记忆机制**：从过往经验中学习、携带上下文记忆，处理长程多步任务更稳。
- **多模态提示**：支持图像 / 语言 / 演示混合提示，会"照着示范做"。
- **涌现的组合泛化（compositional generalization）**：未见过的指令组合也能泛化——这是迈向"真正通用"的信号。
- **效果**：在多项基准上**超越专精策略（specialist policies）**，而不只是追平。

<div class="fig">
  <svg viewBox="0 0 680 150" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="20" y="30" font-size="13" fill="#1a1a1a" font-weight="bold">π 系列演进主线</text>
    <line x1="40" y1="80" x2="640" y2="80" stroke="#ccc" stroke-width="2" marker-end="url(#c)"/>
    <g font-size="11" text-anchor="middle">
      <circle cx="90" cy="80" r="7" fill="#4285F4"/><text x="90" y="110">π0</text><text x="90" y="125" font-size="9" fill="#555">FlowMatch 50Hz</text>
      <circle cx="200" cy="80" r="7" fill="#4285F4"/><text x="200" y="110">π0-Fast</text><text x="200" y="125" font-size="9" fill="#555">离散化提速</text>
      <circle cx="320" cy="80" r="7" fill="#34A853"/><text x="320" y="110">π0.5</text><text x="320" y="125" font-size="9" fill="#555">跨本体+分解</text>
      <circle cx="440" cy="80" r="7" fill="#FBBC04"/><text x="440" y="110">π0.6*</text><text x="440" y="125" font-size="9" fill="#555">RL 专精</text>
      <circle cx="560" cy="80" r="7" fill="#EA4335"/><text x="560" y="110">π0.7</text><text x="560" y="125" font-size="9" fill="#555">记忆+组合泛化</text>
    </g>
    <defs><marker id="c" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#ccc"/></marker></defs>
  </svg>
  <p class="fig-cap">图：π 系列的演进——底座升级、跨本体、RL 专精、记忆与组合泛化</p>
</div>

## 6. 演进主线（一句话）

> VLM 底座升级 → 跨本体复用 → RL 专精打磨 → 记忆 + 组合泛化。**通用 VLA 正从"能干活"走向"比专精策略更强"。**

## 7. 承上启下

π 系列是海外标杆。下一篇看国内：蚂蚁灵波 LingBot-VLA 2.0 的"一脑多机"、小米的开源实时 VLA、腾讯 HyVLA 的 FlowPRO——以及一处常见混淆的澄清。
