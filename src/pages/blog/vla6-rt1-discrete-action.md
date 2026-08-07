---
title: VLA 解码手记（六）：RT-1 的离散动作路线——TokenLearner、因果掩码与交叉熵
description: RT-1 是第一条"把机器人控制当成语言生成"的路线。本文拆它怎么用 EfficientNet 提取视觉 token、TokenLearner 把 6 帧 486 个 token 压到 48 个、Decoder-only Transformer 用因果掩码 + 分类交叉熵逐 token 预测动作，以及掩码与损失各有哪些常见变体。
pubDate: 2026-08-07
series: VLA 解码手记
lang: zh
altLang: en
altHref: /en/blog/vla6-rt1-discrete-action
layout: ../../layouts/BlogPost.astro
---

## 1. 为什么 RT-1 选"离散 token"

VLA 系列第一篇说过，机器人动作的表示有两条路：

- **连续生成**：直接回归关节角、速度向量，轨迹平滑，是现在 π0 / 小米的主战场。
- **离散 token 化**：把连续动作切成一个个离散符号，用自回归方式逐个生成，像 LLM 写句子一样写动作。

**RT-1（Google Robotics, 2022）**走的是第二条路。它的核心假设很朴素：既然 LLM 能把数万亿 token 的语言规律学得很好，那把动作也变成 token，就能直接复用整套语言模型训练栈——自回归、Transformer、交叉熵，全都能搬过来。

代价也明显：动作是连续的，强行量化会损失精度，轨迹容易不连贯。但 RT-1 证明了这条路的**工程简洁性**和**可扩展性**，也为后来的 RT-2、OpenVLA 打下了基础。

> 本文要回答的，正是 RT-1 里两个让新手最卡壳的问题：
> 1. **6 帧图像、486 个视觉 token 是怎么被 TokenLearner 压到 48 个的？**
> 2. **Decoder-only Transformer 的动作解码里，"标准因果掩码 + 分类交叉熵损失"到底分哪几种？**

---

## 2. 视觉侧：EfficientNet + TokenLearner

### 2.1 从图像到 token：每帧 81 个

RT-1 的输入是**6 张连续的机器人视角图像**。

为什么是 6 帧？

> 机器人控制频率是 **3 Hz**，6 帧就是 **最近约 2 秒** 的历史观测。模型需要这段"短期记忆"来判断物体怎么动、手爪当前在哪、上一帧动作执行到哪一步。

每张 300×300 的 RGB 图像，先过一个 **EfficientNet-B3** 视觉编码器：

- EfficientNet-B3 最后一层输出特征图尺寸是 **9×9×512**。
- 把 9×9 空间位置拉平，就得到 **81 个视觉 token**，每个 token 是 512 维向量。

于是单帧：

```text
300×300 图像  →  EfficientNet-B3  →  9×9×512  →  81 tokens（512-d）
```

### 2.2 6 帧的 token 爆炸

6 帧图像，每帧 81 个 token，总共就是：

```text
6 × 81 = 486 个视觉 token
```

这还没算语言指令 token 和特殊 token。486 个视觉 token 对 3 Hz 实时推理来说太贵了——自回归 Transformer 的计算量大致随序列长度平方增长，token 越多越慢。

### 2.3 TokenLearner：把 81 个 token 压到 8 个

RT-1 的解法是 **TokenLearner（Google, 2021）**。它的思路不是"等比例压缩特征图"，而是**让模型自己决定哪些空间位置最重要**。

具体做法（元素级注意力）：

1. 对 9×9 特征图里的每个空间位置，都算一个**重要性分数**；
2. 这个分数是"元素级"的——不只看空间位置，也看 512 维通道里哪些响应强；
3. 用这组分数做**加权聚合**，从 81 个 token 里**自适应地选出 8 个代表 token**。

直观类比：

> 一张桌上有一堆物品，TokenLearner 就像一位摄影师，不是把整张照片平均缩小，而是自动把镜头对准"和手爪任务最相关的几个区域"，其他背景虚化掉。

效果：

```text
6 帧 × 81 token  →  TokenLearner  →  6 帧 × 8 token = 48 个视觉 token
```

论文里 RT-1 的推理速度因此提升了约 **2.4×**。

<div class="fig">
  <svg viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="340" y="22" text-anchor="middle" font-size="15" font-weight="bold" fill="#1a1a1a">RT-1 视觉压缩：从 486 到 48 个 token</text>

    <!-- 6 frames -->
    <g transform="translate(30,50)">
      <text x="0" y="0" font-size="12" fill="#555">输入：6 帧历史图像（3 Hz ≈ 2 秒）</text>
      <rect x="0" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
      <rect x="47" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
      <rect x="94" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
      <rect x="141" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
      <rect x="188" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
      <rect x="235" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
    </g>

    <!-- EfficientNet -->
    <rect x="30" y="125" width="130" height="55" rx="8" fill="#E6F4EA" stroke="#34A853"/>
    <text x="95" y="150" text-anchor="middle" font-size="12" fill="#1a1a1a">EfficientNet-B3</text>
    <text x="95" y="168" text-anchor="middle" font-size="10" fill="#555">300×300 → 9×9×512</text>

    <!-- arrow -->
    <line x1="95" y1="102" x2="95" y2="123" stroke="#888" stroke-width="2" marker-end="url(#rta)"/>

    <!-- per-frame tokens -->
    <g transform="translate(180,125)">
      <text x="0" y="18" font-size="11" fill="#555">每帧 81 token</text>
      <text x="0" y="36" font-size="11" fill="#555">6 帧共 486 token</text>
      <text x="0" y="54" font-size="11" fill="#EA4335">对 3Hz 实时推理太贵</text>
    </g>

    <!-- TokenLearner -->
    <rect x="330" y="125" width="130" height="55" rx="8" fill="#FEF7E0" stroke="#FBBC04"/>
    <text x="395" y="150" text-anchor="middle" font-size="12" fill="#1a1a1a">TokenLearner</text>
    <text x="395" y="168" text-anchor="middle" font-size="10" fill="#555">元素级注意力 → 8 token/帧</text>

    <line x1="270" y1="152" x2="328" y2="152" stroke="#888" stroke-width="2" marker-end="url(#rta)"/>

    <!-- compressed -->
    <g transform="translate(480,125)">
      <text x="0" y="18" font-size="11" fill="#555">压缩后 48 token</text>
      <text x="0" y="36" font-size="11" fill="#555">+ 语言条件 + &lt;BOS&gt;</text>
      <text x="0" y="54" font-size="11" fill="#34A853">推理速度 ↑ 2.4×</text>
    </g>

    <line x1="460" y1="152" x2="478" y2="152" stroke="#888" stroke-width="2" marker-end="url(#rta)"/>

    <!-- equation bar -->
    <rect x="30" y="220" width="620" height="45" rx="6" fill="#f8f9fa" stroke="#dadce0"/>
    <text x="340" y="248" text-anchor="middle" font-size="14" fill="#1a1a1a" font-family="monospace">300×300 × 6 frames  →  9×9×512 × 6 = 486 tokens  →  TokenLearner  →  8×6 = 48 tokens</text>

    <text x="340" y="310" text-anchor="middle" font-size="12" fill="#1a73e8">6 帧 = 最近 2 秒历史观测；TokenLearner 不是平均下采样，而是自适应保留任务相关区域。</text>

    <defs>
      <marker id="rta" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="#888"/>
      </marker>
    </defs>
  </svg>
  <p class="fig-cap">图：RT-1 先用 EfficientNet 把每帧图像变成 81 个 token，再用 TokenLearner 压到 8 个，6 帧共 48 个视觉 token。</p>
</div>

---

## 3. 动作侧：Decoder-only Transformer 动作解码

视觉 token 被压缩后，会和**语言指令 token**、一个**特殊起始 token `<BOS>`** 一起，送进一个 Decoder-only Transformer。

### 3.1 输入与输出长什么样

**输入序列**（从左到右）：

```text
<BOS>  [语言指令 token]  [视觉 token × 48]
```

**输出序列**（自回归逐 token 预测）：

```text
[模式选择 token]  [手臂动作 token …]  [底座动作 token …]
```

RT-1 把动作拆成三类离散 token：

| 类型 | 含义 | 例子 |
|---|---|---|
| **模式选择** | 末端执行器开合 / 运动模式 | open / close / move |
| **手臂动作** | 机械臂 6–7 个自由度的离散化关节角 | 每个维度量化成若干 bin |
| **底座动作** | 移动底盘速度 / 转向 | 前/后/左转/右转等 |

模型一次生成一串动作 token，最后再把它们解码回机器人的实际控制命令。

### 3.2 标准因果掩码：只能看左边

Decoder-only Transformer 的招牌是**因果掩码（Causal Mask）**：

> 位置 t 只能关注位置 ≤ t 的信息，不能偷看未来。

这保证了自回归的合法性——预测第 t 个动作 token 时，只能用已经生成的 t-1、t-2 … 以及输入侧的历史信息。

<div class="fig">
  <svg viewBox="0 0 680 160" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="340" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a1a1a">标准因果掩码：位置 t 只能看 ≤ t</text>

    <!-- matrix -->
    <g transform="translate(220,40)">
      <text x="-20" y="15" font-size="12" fill="#555">query</text>
      <text x="100" y="-8" text-anchor="middle" font-size="12" fill="#555">key</text>
      <!-- grid 6x6 -->
      <!-- row 1 -->
      <rect x="0" y="0" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="24" y="0" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="48" y="0" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="72" y="0" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="96" y="0" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="120" y="0" width="24" height="24" fill="#dadce0" stroke="#fff"/>
      <!-- row 2 -->
      <rect x="0" y="24" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="24" y="24" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="48" y="24" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="72" y="24" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="96" y="24" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="120" y="24" width="24" height="24" fill="#dadce0" stroke="#fff"/>
      <!-- row 3 -->
      <rect x="0" y="48" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="24" y="48" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="48" y="48" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="72" y="48" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="96" y="48" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="120" y="48" width="24" height="24" fill="#dadce0" stroke="#fff"/>
      <!-- row 4 -->
      <rect x="0" y="72" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="24" y="72" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="48" y="72" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="72" y="72" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="96" y="72" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="120" y="72" width="24" height="24" fill="#dadce0" stroke="#fff"/>
      <!-- row 5 -->
      <rect x="0" y="96" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="24" y="96" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="48" y="96" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="72" y="96" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="96" y="96" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="120" y="96" width="24" height="24" fill="#dadce0" stroke="#fff"/>

      <!-- labels -->
      <text x="12" y="-15" text-anchor="middle" font-size="10" fill="#555">1</text>
      <text x="36" y="-15" text-anchor="middle" font-size="10" fill="#555">2</text>
      <text x="60" y="-15" text-anchor="middle" font-size="10" fill="#555">3</text>
      <text x="84" y="-15" text-anchor="middle" font-size="10" fill="#555">4</text>
      <text x="108" y="-15" text-anchor="middle" font-size="10" fill="#555">5</text>
      <text x="132" y="-15" text-anchor="middle" font-size="10" fill="#555">6</text>

      <text x="-12" y="16" text-anchor="end" font-size="10" fill="#555">1</text>
      <text x="-12" y="40" text-anchor="end" font-size="10" fill="#555">2</text>
      <text x="-12" y="64" text-anchor="end" font-size="10" fill="#555">3</text>
      <text x="-12" y="88" text-anchor="end" font-size="10" fill="#555">4</text>
      <text x="-12" y="112" text-anchor="end" font-size="10" fill="#555">5</text>
    </g>

    <!-- legend -->
    <g transform="translate(450,60)">
      <rect x="0" y="0" width="16" height="16" fill="#34A853"/>
      <text x="24" y="13" font-size="12" fill="#555">可见（允许关注）</text>
      <rect x="0" y="28" width="16" height="16" fill="#dadce0"/>
      <text x="24" y="41" font-size="12" fill="#555">遮蔽（不可见）</text>
    </g>

    <text x="340" y="150" text-anchor="middle" font-size="12" fill="#1a73e8">下三角绿色：预测第 t 个 token 时，只能使用第 1…t 个位置的信息。</text>
  </svg>
  <p class="fig-cap">图：标准因果掩码是一个下三角矩阵，保证自回归生成时不能偷看未来 token。</p>
</div>

---

## 4. 因果掩码的几种形式

实际实现里，"只能看左边"这件事有几种变体，适用场景不同。

### 4.1 严格因果掩码（标准自回归）

最常用。位置 t 只能关注 ≤ t 的位置。训练和推理都如此，对应上图的下三角。

### 4.2 带起始 token 的掩码

如果序列开头有一个特殊的 `<BOS>` 或 `<PAD>`，有时希望**所有位置都能关注这个起始 token**（它不包含未来信息，只是提供全局上下文），而其他位置仍保持严格因果。实现上等于第一列全绿，其余下三角。

### 4.3 块因果掩码（Block Causal Mask）

动作通常按"组"生成：比如先出"模式"，再出一组"手臂关节"，最后出"底座"。块因果掩码允许：

- 同一块内严格因果；
- 当前块可以关注前面所有块；
- 但不能关注后面块。

这在 RT-1 这类"多字段动作 token"的模型里很自然——底座 token 生成时可以看模式和手臂，但手臂生成时不能偷看底座。

### 4.4 缓存式掩码（KV Cache Mask）

推理加速时，已经算过的 token 的 Key/Value 会被缓存。此时掩码表现为：

- 历史 token（已缓存）：不需要再算，标为 `*`；
- 当前新 token：只和缓存 + 自己这一行做注意力；
- 未来 token：仍遮蔽。

<div class="fig">
  <svg viewBox="0 0 680 210" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="340" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a1a1a">因果掩码的四种常见形式</text>

    <!-- helper to draw a mini matrix -->
    <g transform="translate(30,40)">
      <text x="60" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">1. 严格因果</text>
      <g transform="translate(20,10)">
        <rect x="0" y="0" width="14" height="14" fill="#34A853"/><rect x="14" y="0" width="14" height="14" fill="#dadce0"/><rect x="28" y="0" width="14" height="14" fill="#dadce0"/><rect x="42" y="0" width="14" height="14" fill="#dadce0"/><rect x="56" y="0" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="14" width="14" height="14" fill="#34A853"/><rect x="14" y="14" width="14" height="14" fill="#34A853"/><rect x="28" y="14" width="14" height="14" fill="#dadce0"/><rect x="42" y="14" width="14" height="14" fill="#dadce0"/><rect x="56" y="14" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="28" width="14" height="14" fill="#34A853"/><rect x="14" y="28" width="14" height="14" fill="#34A853"/><rect x="28" y="28" width="14" height="14" fill="#34A853"/><rect x="42" y="28" width="14" height="14" fill="#dadce0"/><rect x="56" y="28" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="42" width="14" height="14" fill="#34A853"/><rect x="14" y="42" width="14" height="14" fill="#34A853"/><rect x="28" y="42" width="14" height="14" fill="#34A853"/><rect x="42" y="42" width="14" height="14" fill="#34A853"/><rect x="56" y="42" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="56" width="14" height="14" fill="#34A853"/><rect x="14" y="56" width="14" height="14" fill="#34A853"/><rect x="28" y="56" width="14" height="14" fill="#34A853"/><rect x="42" y="56" width="14" height="14" fill="#34A853"/><rect x="56" y="56" width="14" height="14" fill="#34A853"/>
      </g>
      <text x="60" y="90" text-anchor="middle" font-size="10" fill="#555">最常用，纯自回归</text>
    </g>

    <g transform="translate(180,40)">
      <text x="60" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">2. 带 &lt;BOS&gt;</text>
      <g transform="translate(20,10)">
        <rect x="0" y="0" width="14" height="14" fill="#34A853"/><rect x="14" y="0" width="14" height="14" fill="#dadce0"/><rect x="28" y="0" width="14" height="14" fill="#dadce0"/><rect x="42" y="0" width="14" height="14" fill="#dadce0"/><rect x="56" y="0" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="14" width="14" height="14" fill="#34A853"/><rect x="14" y="14" width="14" height="14" fill="#34A853"/><rect x="28" y="14" width="14" height="14" fill="#dadce0"/><rect x="42" y="14" width="14" height="14" fill="#dadce0"/><rect x="56" y="14" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="28" width="14" height="14" fill="#34A853"/><rect x="14" y="28" width="14" height="14" fill="#34A853"/><rect x="28" y="28" width="14" height="14" fill="#34A853"/><rect x="42" y="28" width="14" height="14" fill="#dadce0"/><rect x="56" y="28" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="42" width="14" height="14" fill="#34A853"/><rect x="14" y="42" width="14" height="14" fill="#34A853"/><rect x="28" y="42" width="14" height="14" fill="#34A853"/><rect x="42" y="42" width="14" height="14" fill="#34A853"/><rect x="56" y="42" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="56" width="14" height="14" fill="#34A853"/><rect x="14" y="56" width="14" height="14" fill="#34A853"/><rect x="28" y="56" width="14" height="14" fill="#34A853"/><rect x="42" y="56" width="14" height="14" fill="#34A853"/><rect x="56" y="56" width="14" height="14" fill="#34A853"/>
      </g>
      <text x="60" y="90" text-anchor="middle" font-size="10" fill="#555">第 1 列全绿，其余下三角</text>
    </g>

    <g transform="translate(330,40)">
      <text x="60" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">3. 块因果</text>
      <g transform="translate(20,10)">
        <!-- block 1 -->
        <rect x="0" y="0" width="14" height="14" fill="#34A853"/><rect x="14" y="0" width="14" height="14" fill="#dadce0"/><rect x="28" y="0" width="14" height="14" fill="#dadce0"/><rect x="42" y="0" width="14" height="14" fill="#dadce0"/><rect x="56" y="0" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="14" width="14" height="14" fill="#34A853"/><rect x="14" y="14" width="14" height="14" fill="#34A853"/><rect x="28" y="14" width="14" height="14" fill="#dadce0"/><rect x="42" y="14" width="14" height="14" fill="#dadce0"/><rect x="56" y="14" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="28" width="14" height="14" fill="#34A853"/><rect x="14" y="28" width="14" height="14" fill="#34A853"/><rect x="28" y="28" width="14" height="14" fill="#34A853"/><rect x="42" y="28" width="14" height="14" fill="#dadce0"/><rect x="56" y="28" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="42" width="14" height="14" fill="#34A853"/><rect x="14" y="42" width="14" height="14" fill="#34A853"/><rect x="28" y="42" width="14" height="14" fill="#34A853"/><rect x="42" y="42" width="14" height="14" fill="#34A853"/><rect x="56" y="42" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="56" width="14" height="14" fill="#34A853"/><rect x="14" y="56" width="14" height="14" fill="#34A853"/><rect x="28" y="56" width="14" height="14" fill="#34A853"/><rect x="42" y="56" width="14" height="14" fill="#34A853"/><rect x="56" y="56" width="14" height="14" fill="#34A853"/>
      </g>
      <text x="60" y="90" text-anchor="middle" font-size="10" fill="#555">同块内因果，块间可见</text>
    </g>

    <g transform="translate(480,40)">
      <text x="60" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">4. 缓存式 (KV Cache)</text>
      <g transform="translate(20,10)">
        <rect x="0" y="0" width="14" height="14" fill="#FBBC04"/><rect x="14" y="0" width="14" height="14" fill="#dadce0"/><rect x="28" y="0" width="14" height="14" fill="#dadce0"/><rect x="42" y="0" width="14" height="14" fill="#dadce0"/><rect x="56" y="0" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="14" width="14" height="14" fill="#FBBC04"/><rect x="14" y="14" width="14" height="14" fill="#FBBC04"/><rect x="28" y="14" width="14" height="14" fill="#dadce0"/><rect x="42" y="14" width="14" height="14" fill="#dadce0"/><rect x="56" y="14" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="28" width="14" height="14" fill="#FBBC04"/><rect x="14" y="28" width="14" height="14" fill="#FBBC04"/><rect x="28" y="28" width="14" height="14" fill="#FBBC04"/><rect x="42" y="28" width="14" height="14" fill="#dadce0"/><rect x="56" y="28" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="42" width="14" height="14" fill="#FBBC04"/><rect x="14" y="42" width="14" height="14" fill="#FBBC04"/><rect x="28" y="42" width="14" height="14" fill="#FBBC04"/><rect x="42" y="42" width="14" height="14" fill="#FBBC04"/><rect x="56" y="42" width="14" height="14" fill="#dadce0"/>
        <rect x="0" y="56" width="14" height="14" fill="#FBBC04"/><rect x="14" y="56" width="14" height="14" fill="#FBBC04"/><rect x="28" y="56" width="14" height="14" fill="#FBBC04"/><rect x="42" y="56" width="14" height="14" fill="#FBBC04"/><rect x="56" y="56" width="14" height="14" fill="#FBBC04"/>
      </g>
      <text x="60" y="90" text-anchor="middle" font-size="10" fill="#555">黄色=已缓存，无需重算</text>
    </g>

    <text x="340" y="195" text-anchor="middle" font-size="12" fill="#1a73e8">严格因果最常用；带 &lt;BOS&gt; 和块因果适合多字段动作；缓存式是推理加速的工程技巧。</text>
  </svg>
  <p class="fig-cap">图：四种因果掩码变体。绿色=可见，灰色=遮蔽，黄色=已缓存。</p>
</div>

---

## 5. 分类交叉熵损失：从标准到变体

动作 token 是离散类别，所以训练目标就是**分类交叉熵（Cross-Entropy Loss）**：

```text
L = - Σ  y_t,v · log( p_hat_t,v )
```

其中 `y_t` 是真实动作 token 的 one-hot 向量，`p_hat_t` 是模型预测的概率分布。简单说：模型预测一个概率分布，损失只看真实类别对应的那个概率有没有被推高。

### 5.1 标准交叉熵（单标签）

真实标签是 one-hot：真实位置为 1，其余为 0。损失 = -log(真实位置的概率)。

这是 RT-1 训练的基础。

### 5.2 标签平滑（Label Smoothing）

标准交叉熵会让模型对答案过于"自信"，容易过拟合。标签平滑把 one-hot 改造成：

```text
真实位置 = 1 - ε
其余位置 = ε / (V - 1)
```

模型不再被要求把真实类别概率推到 1，而是推到 1-ε，同时给其他类别留一点概率。相当于"允许模型犯错一点点"，泛化更稳。

### 5.3 加权交叉熵（Weighted Cross-Entropy）

不同动作类别可能样本不均衡——比如"底座动作"出现次数远少于"手臂动作"。加权交叉熵给稀少类别更高的惩罚权重：

```text
L_t = - w_y_t · log( p_hat_t, y_t )
```

这在机器人数据里很实用，避免模型只学会常见的"安全动作"。

<div class="fig">
  <svg viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="340" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a1a1a">交叉熵损失的三种形式</text>

    <!-- standard -->
    <g transform="translate(40,45)">
      <text x="80" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">1. 标准交叉熵</text>
      <g transform="translate(0,15)">
        <rect x="0" y="40" width="20" height="10" fill="#4285F4"/>
        <rect x="25" y="0" width="20" height="50" fill="#EA4335"/>
        <rect x="50" y="45" width="20" height="5" fill="#4285F4"/>
        <rect x="75" y="42" width="20" height="8" fill="#4285F4"/>
        <rect x="100" y="44" width="20" height="6" fill="#4285F4"/>
        <rect x="125" y="41" width="20" height="9" fill="#4285F4"/>
        <!-- axis -->
        <line x1="0" y1="50" x2="150" y2="50" stroke="#333" stroke-width="1"/>
        <line x1="0" y1="50" x2="0" y2="0" stroke="#333" stroke-width="1"/>
      </g>
      <text x="80" y="95" text-anchor="middle" font-size="10" fill="#555">真实类别 probability = 1</text>
    </g>

    <!-- label smoothing -->
    <g transform="translate(250,45)">
      <text x="80" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">2. 标签平滑</text>
      <g transform="translate(0,15)">
        <rect x="0" y="35" width="20" height="15" fill="#4285F4"/>
        <rect x="25" y="5" width="20" height="45" fill="#EA4335"/>
        <rect x="50" y="38" width="20" height="12" fill="#4285F4"/>
        <rect x="75" y="36" width="20" height="14" fill="#4285F4"/>
        <rect x="100" y="37" width="20" height="13" fill="#4285F4"/>
        <rect x="125" y="35" width="20" height="15" fill="#4285F4"/>
        <line x1="0" y1="50" x2="150" y2="50" stroke="#333" stroke-width="1"/>
        <line x1="0" y1="50" x2="0" y2="0" stroke="#333" stroke-width="1"/>
      </g>
      <text x="80" y="95" text-anchor="middle" font-size="10" fill="#555">真实类别 1-ε，其余均分 ε</text>
    </g>

    <!-- weighted -->
    <g transform="translate(460,45)">
      <text x="80" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">3. 加权交叉熵</text>
      <g transform="translate(0,15)">
        <rect x="0" y="42" width="20" height="8" fill="#4285F4"/>
        <rect x="25" y="2" width="20" height="48" fill="#EA4335"/>
        <rect x="50" y="10" width="20" height="40" fill="#FBBC04"/>
        <rect x="75" y="8" width="20" height="42" fill="#FBBC04"/>
        <rect x="100" y="12" width="20" height="38" fill="#FBBC04"/>
        <rect x="125" y="9" width="20" height="41" fill="#FBBC04"/>
        <line x1="0" y1="50" x2="150" y2="50" stroke="#333" stroke-width="1"/>
        <line x1="0" y1="50" x2="0" y2="0" stroke="#333" stroke-width="1"/>
      </g>
      <text x="80" y="95" text-anchor="middle" font-size="10" fill="#555">不同类别乘不同权重 w</text>
    </g>

    <!-- formula -->
    <rect x="30" y="155" width="620" height="45" rx="6" fill="#f8f9fa" stroke="#dadce0"/>
    <text x="340" y="173" text-anchor="middle" font-size="13" fill="#1a1a1a">标准：L = -log p̂ₜ,ᵧₜ</text>
    <text x="340" y="190" text-anchor="middle" font-size="13" fill="#1a1a1a">加权：L = -wᵧₜ · log p̂ₜ,ᵧₜ ｜ 平滑：真实标签 = 1-ε，其余 = ε/(V-1)</text>
  </svg>
  <p class="fig-cap">图：三种交叉熵变体。标准版最硬；标签平滑防过拟合；加权版处理类别不均衡。</p>
</div>

---

## 6. 承上启下：RT-1 → RT-2 → 原生 VLA

RT-1 的意义不是"最高性能"，而是**证明了 VLA 可以工程化**：

- 把图像变成 token；
- 把动作也变成 token；
- 用 Decoder-only Transformer + 因果掩码 + 交叉熵统一训练。

这条路线后来被 RT-2 发扬光大：它把视觉语言大模型（VLM）直接接在动作输出上，让机器人能利用互联网规模的视觉-语言知识。再到 π0、OpenVLA、小米，则走向了**原生 VLA + 连续生成**，把精度和平滑性又往上推了一级。

> 所以 RT-1 不是终点，而是"把控制问题重写成语言问题"的起点。理解它的 TokenLearner、因果掩码和交叉熵，就等于理解了后来所有 VLA 的公共基因。

> 投资视角：具身智能的模型栈正在快速收敛。离散路线（RT-1/RT-2/OpenVLA）和连续路线（π0/小米）不是二选一，而是会在不同延迟/精度/成本场景里并存。关注能把两条路线都跑通、且有真实数据闭环的团队。
