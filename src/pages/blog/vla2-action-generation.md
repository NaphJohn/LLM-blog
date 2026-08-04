---
title: VLA 解码手记（二）：动作生成的发动机——Diffusion Policy、Flow Matching 与低延迟
description: 为什么高性能 VLA 不用离散 token 而用连续生成？本文拆解 Diffusion Policy、Flow Matching、Action Chunking，以及小米 Xiaomi-Robotics-0 如何用 MoT + 异步推理把延迟压到 80ms、在 4090 上实时跑。
pubDate: 2026-08-04
series: VLA 解码手记
lang: zh
altLang: en
altHref: /en/blog/vla2-action-generation
layout: ../../layouts/BlogPost.astro
---

## 1. 为什么离散 token 不够用

上篇说到，把连续动作量化成 token 自回归生成，会**截断精度**、轨迹抖动。机器人需要的是高频、平滑、对扰动敏感的连续控制——这正是扩散/流匹配擅长的。

## 2. Diffusion Policy：把动作当"图像"去去噪

Diffusion Policy（Chi et al., 2023）借用了图像生成的扩散思想：

- 训练时给动作序列加噪声，让网络学"从噪声恢复动作"；
- 推理时从随机噪声出发，多步去噪，得到一条**多模态、鲁棒**的动作轨迹。

好处是天然支持多峰分布（同一画面可以有多种合理动作），对示范数据里的噪声也更能扛。

## 3. Flow Matching：比扩散更快更稳

Flow Matching 不直接建模复杂的概率路径，而是学习**概率流（probability flow）的映射**——把简单分布直线"推"向目标动作分布。

- 训练目标更简单、梯度更稳；
- 推理采样步数远少于传统 DDPM（常需数十~数百步），可压到 **5 步左右**；
- π0、小米都把 Flow Matching 当动作专家的核心。

> 本站（推测解码手记）里讲过的"流匹配"在这里落地成了**物理动作**，不是文字 token——同一个数学工具，跨模态复用。

## 4. Action Chunking：一次预测一整段

高频决策每毫秒都出动作，既慢又抖。Action Chunking（源自 ACT / π0）让模型**一次预测一小段动作块（chunk）**：

- 降低决策频率，动作更连贯；
- 块内自回归/扩散生成，块间用历史动作做前缀衔接；
- π0 用 Flow Matching + 50Hz 动作块，是"平滑 + 高频"的解法。

## 5. 小米的打法：MoT 松耦合 + 异步推理压延迟

Xiaomi-Robotics-0（2026-02 开源，4.7B）是低延迟 VLA 的范本：

- **MoT 架构**：VLM"大脑"负责理解；16 层 **DiT（Diffusion Transformer）"小脑"**负责生成动作块。两者用 **KV Cache 松耦合**——大脑输出直接喂给小脑，避免重复计算。
- **Flow Matching 训练**：采样步数从 DDPM 的数十步压到 5 步。
- **异步推理**：模型推理与机器人执行**解耦**——推理延迟不再卡住真机连续性，从机制上消灭"动作断层"。
- **Clean Action Prefix + Λ 型注意力**：用上一刻动作做输入保证时间连续；特殊注意力掩码让模型更盯当前视觉反馈，对环境突变反应敏捷。

结果：**80ms 推理延迟、30Hz 控制频率、RTX 4090 实时**，LIBERO / CALVIN / SimplerEnv 全基准 SOTA。

<div class="fig">
  <svg viewBox="0 0 680 190" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <rect x="10" y="60" width="150" height="70" rx="10" fill="#E8F0FE" stroke="#4285F4"/>
    <text x="85" y="95" text-anchor="middle" font-size="13" fill="#1a1a1a">VLM 大脑</text>
    <text x="85" y="113" text-anchor="middle" font-size="11" fill="#555">理解+决策</text>
    <rect x="200" y="60" width="150" height="70" rx="10" fill="#FEF7E0" stroke="#FBBC04"/>
    <text x="275" y="95" text-anchor="middle" font-size="13" fill="#1a1a1a">KV Cache</text>
    <text x="275" y="113" text-anchor="middle" font-size="11" fill="#555">松耦合传递</text>
    <rect x="390" y="60" width="150" height="70" rx="10" fill="#E6F4EA" stroke="#34A853"/>
    <text x="465" y="95" text-anchor="middle" font-size="13" fill="#1a1a1a">DiT 小脑</text>
    <text x="465" y="113" text-anchor="middle" font-size="11" fill="#555">生成动作块</text>
    <rect x="560" y="60" width="110" height="70" rx="10" fill="#FCE8E6" stroke="#EA4335"/>
    <text x="615" y="95" text-anchor="middle" font-size="13" fill="#1a1a1a">真机</text>
    <text x="615" y="113" text-anchor="middle" font-size="11" fill="#555">30Hz 异步</text>
    <line x1="160" y1="95" x2="198" y2="95" stroke="#888" stroke-width="2" marker-end="url(#b)"/>
    <line x1="350" y1="95" x2="388" y2="95" stroke="#888" stroke-width="2" marker-end="url(#b)"/>
    <line x1="540" y1="95" x2="558" y2="95" stroke="#888" stroke-width="2" marker-end="url(#b)"/>
    <defs><marker id="b" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#888"/></marker></defs>
  </svg>
  <p class="fig-cap">图：小米 MoT——大脑与小脑经 KV Cache 松耦合，异步驱动真机</p>
</div>

## 6. 承上启下

动作生成机制清楚了。下一篇看 π 系列如何用这套机制一步步进化：从 π0 的 Flow Matching 50Hz，到 π0.7 的记忆与组合泛化。
