---
title: VLA 解码手记（一）：从「看懂」到「动手」——什么是视觉-语言-动作模型
description: 具身智能的核心是把"理解"延伸到"行动"。本文拆解 VLA（Vision-Language-Action）模型的定义、它为何出现、从任务专用策略到原生 VLA 的范式跃迁，以及动作表征的两条路线（离散 token vs 连续生成）。
pubDate: 2026-08-04
series: VLA 解码手记
lang: zh
altLang: en
altHref: /en/blog/vla1-basics
layout: ../../layouts/BlogPost.astro
---

## 1. 为什么需要 VLA

前面的「多模态解码手记」讲完了 VLM——模型已经能**看**图、**说**出图里有什么。但真实世界里有大量任务要求的不只是"说"，而是**动手**：把碗放到沥水架、叠好毛巾、把零件装进卡槽。

**VLA（Vision-Language-Action，视觉-语言-动作）模型**就是把这三件事接到一条链路里：

> 输入 = 图像 / 视频（看到了什么）+ 语言指令（要做什么）
> 输出 = **机器人的连续动作**（去哪、怎么动、用多大力）

它是「具身智能（Embodied AI）」的操作内核——让模型不仅生成文字，还生成能改变物理世界的**控制指令**。

<div class="fig">
  <svg viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <rect x="10" y="70" width="120" height="60" rx="10" fill="#E8F0FE" stroke="#4285F4"/>
    <text x="70" y="105" text-anchor="middle" font-size="14" fill="#1a1a1a">图像 / 视频</text>
    <rect x="160" y="70" width="120" height="60" rx="10" fill="#E6F4EA" stroke="#34A853"/>
    <text x="220" y="105" text-anchor="middle" font-size="14" fill="#1a1a1a">语言指令</text>
    <rect x="320" y="60" width="140" height="80" rx="10" fill="#FEF7E0" stroke="#FBBC04"/>
    <text x="390" y="100" text-anchor="middle" font-size="14" fill="#1a1a1a">VLA 模型</text>
    <text x="390" y="120" text-anchor="middle" font-size="11" fill="#555">VLM 大脑 + 动作头</text>
    <rect x="510" y="70" width="150" height="60" rx="10" fill="#FCE8E6" stroke="#EA4335"/>
    <text x="585" y="100" text-anchor="middle" font-size="14" fill="#1a1a1a">机器人动作</text>
    <text x="585" y="120" text-anchor="middle" font-size="11" fill="#555">连续控制序列</text>
    <line x1="130" y1="100" x2="158" y2="100" stroke="#888" stroke-width="2" marker-end="url(#a)"/>
    <line x1="280" y1="100" x2="318" y2="100" stroke="#888" stroke-width="2" marker-end="url(#a)"/>
    <line x1="460" y1="100" x2="508" y2="100" stroke="#888" stroke-width="2" marker-end="url(#a)"/>
    <defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#888"/></marker></defs>
  </svg>
  <p class="fig-cap">图：VLA 把"看到 + 说清"升级为"看到 + 说清 + 做对"</p>
</div>

## 2. 范式跃迁：从手写策略到原生 VLA

机器人控制不是新课题。它走过几条路：

1. **任务专用策略（Task-specific policy）**：每个任务单独写运动规划 + 感知模块。泛化差，长尾任务写不过来。
2. **模仿学习（BC / DAgger）**：采集人类示范，让模型学"看状态→出动作"。能处理复杂任务，但换本体/换指令就崩。
3. **VLM 改造**：在现成 VLM 后面接一个"动作头"，把图像理解变成动作。便宜，但理解和动作是两段式，容易脱节。
4. **原生 VLA（联合预训练 V+L+A）**：把视觉、语言、**动作**在同一套表征里一起预训练。这是当前主流方向（π0、OpenVLA、Xiaomi-Robotics-0 都属此类）。

> 关键判断：VLA 不是"再训一个大模型"，而是把 LLM/VLM 已经具备的**常识与泛化**，顺着（一）（二）里讲的对齐范式，延伸到物理动作空间。

## 3. 动作怎么表示：两条路线

连续动作（机械臂 6–7 自由度关节角、移动底座速度）不能直接当文本 token。主流有两种编码：

- **离散 token 化**：把连续动作量化成一串离散 token，用自回归方式逐个生成（类似语言模型）。简单、能复用 LLM 训练栈，但量化会**截断精度**、轨迹不连续，高频控制下像"反应迟钝的木头人"。
- **连续生成（Diffusion / Flow Matching）**：直接回归**连续动作分布**，一步产出平滑的动作向量。轨迹自然、帧率高，是 π0、小米等高性能模型的选择。下一篇专讲这套机制。

## 4. 训练数据的命门：OXE 与跨本体

VLA 的"食材"是**机器人演示数据**。最重要的公开集是 **Open X-Embodiment（OXE）**——把几十家实验室、多种机器人本体的演示凑成跨本体（cross-embodiment）语料，让模型学到"与硬件无关"的通用操作常识。

但真实瓶颈在这里：语言模型吞了数万亿 token、视频模型吞了数十亿 clip，而头部公司的**高质量物理交互数据只有几十万小时**——离验证 scaling law 还差至少一个数量级（WAIC 2026 的共识）。

> 这恰好呼应（二）里"数据构造常是天花板"的判断：VLA 的竞争，短期看架构，长期看**谁先跑通数据飞轮**。

## 5. 承上启下

本篇给了 VLA 的骨架。下一篇深入动作生成的"发动机"——Diffusion Policy、Flow Matching、Action Chunking，以及小米如何把推理延迟压到 80ms。再之后是 π 系列的完整演进，以及国内玩家（蚂蚁灵波、小米、腾讯）的落地打法。

> 投资视角（呼应你的持仓框架）：具身智能正从"炫技"进入"拼大脑"阶段，真正壁垒是**模型体系 + 真实场景数据闭环**。关注有工程闭环能力的公司，比追单一参数更稳。
