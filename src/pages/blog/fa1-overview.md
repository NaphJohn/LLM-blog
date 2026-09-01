---
title: 前沿架构解码手记（一）：2026 前沿大模型总览——长上下文 × 效率 × 规模化 的不可能三角
description: 本系列顺着"推测解码"与"多模态"两条线，走到 2026 年最前沿的模型架构本身。开篇先立一个三角形：长上下文（1M）、推理/训练效率、参数规模化（万亿 MoE）三者互相拉扯。Kimi K3、MiniMax M3、DeepSeek V4 三家各自拽住一个顶点，用不同方式"改注意力"来破局。给出全系列循序渐进路线图。
pubDate: 2026-08-06
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa1-overview
layout: ../../layouts/BlogPost.astro
---

## 0. 本系列的定位

"推测解码手记"讲的是**怎么把现有大模型跑得更快**；"多模态解码手记"讲的是**模型怎么感知并作用于世界**。这两条线最后都汇到一个问题：**模型架构本身，在 2026 年走到了哪一步？**

2026 年集中亮相的三款前沿模型——**Kimi K3、MiniMax M3、DeepSeek V4**——恰好构成了一组"对照实验"：它们面对的是同一个工程难题，却给出了三种不同的解法。本系列就用"循序渐进"的方式，把这三家拆开讲清楚，最后再合起来看趋势。

> 路线图：**（一）总览与路线图** → （二）Kimi K3：注意力与 MoE 的双重进化 → （三）MiniMax M2.7→M3：稀疏注意力的跨越 → （四）DeepSeek V4：极致压缩与高效训练。

## 1. 一个三角形：长上下文 × 效率 × 规模化

理解这三家，先立一个"不可能三角"：

- **顶点 A · 长上下文（1M token）**：让模型"读得下整本书 / 整个代码库 / 一长串工具调用历史"。
- **顶点 B · 效率（低 KV、低每 token 算力）**：上下文变长、模型变大，注意力与 KV Cache 的成本会平方级爆炸。
- **顶点 C · 规模化（万亿级 MoE）**：想把能力做上去，参数必须堆到 1T+，训练和推理都很贵。

朴素做法（标准 softmax 注意力 + 稠密模型）在三角形中心，哪个顶点都够不着——上下文一长就 OOM，模型一大就烧钱。三家的破局思路，本质都是**改造注意力机制 + 上 MoE**，只是抓手不同：

```
            长上下文 (1M)
               /      \
            重设计     压缩
          注意力       注意力
             |          |
   Kimi K3   |    DeepSeek V4
   (KDA+MLA) |    (CSA + HCA)
             \          /
          稀疏注意力
             MiniMax M3
              (MSA)
               \      /
                规模化 (MoE)
```

<div class="keybox">
<strong>一句话：</strong>2026 前沿架构的竞争，不是"谁参数多"，而是"谁能把注意力改到既撑得起 1M 上下文、又不让 KV 与算力爆炸"。三家分别选了<strong>重设计 / 稀疏化 / 压缩</strong>三条路。
</div>

## 2. 注意力改造谱系（速查）：SWA 混合模型是什么

把 2026 年各家对注意力的"改造"摆成一条谱系，从"完全不改"到"彻底换范式"，方便后面每篇对号入座：

| 改造类型 | 代表 | 核心做法 | 注意力复杂度 |
|---|---|---|---|
| 全注意力（Full Attention） | 标准 Transformer、MiniMax M2.7 | 每个 token attend 全部历史 | O(n²) |
| 滑动窗口注意力（SWA） | Longformer / GPT-3 滑窗、早期 Gemma | 每个 token 只看窗口 w 内 | O(n·w) |
| **SWA 混合模型** | **Step-3.5-Flash（3:1）、Mistral / Gemma 类** | **同时堆叠 Full Attention 层与 Sliding Window Attention 层** | O(n²) 局部 + O(n·w) 主体 |
| 稀疏注意力（MSA） | MiniMax M3 | 学习/固定稀疏模式，只算重要对 | ≪ O(n²) |
| 压缩注意力（CSA + HCA） | DeepSeek V4 | 把 KV 压成块 / 极致压缩 | O(n·c) |
| 非注意力（Mamba / SSM） | Kimi K3（部分）、Jamba | 用状态空间替代注意力 | O(n) |

> **SWA 混合模型（Sliding-Window-Attention Hybrid）**：指**同时包含 Full Attention 层和 Sliding Window Attention 层的混合注意力架构模型**。典型做法是把大部分层设为滑动窗口（把主体 KV 开销从 O(n²) 压到 O(n·w)），再每隔若干层插入少量 Full Attention 层，负责"长程信息跨窗口传播"——否则窗口外的信息永远传不进来、模型会"健忘"。Step-3.5-Flash 用的就是 **3:1**（每 3 层滑窗 + 1 层全注意），详见 [Fw6：模型支持与选型](../fw6-model-support-selection)。

<div class="keybox">
<strong>为什么要"混合"？</strong>纯 SWA 便宜但"健忘"（窗口外信息丢失，长程依赖断掉）；纯 Full Attention 信息全但贵。SWA 混合 = 用少量 Full Attention 层"打通"窗口之间的长程依赖，主体仍用 SWA 省钱——这是"性价比"与"记忆力"之间的折中，也是 2026 年很多中等规模模型的首选注意力配方。
</div>

<div class="fig">
<svg viewBox="0 0 680 170" width="100%" role="img" font-family="-apple-system,PingFang SC,Microsoft YaHei,sans-serif">
  <title>注意力改造谱系</title>
  <desc>从全注意力到非注意力（Mamba）的复杂度谱系，SWA 混合模型位于滑窗与稀疏之间。</desc>
  <text x="20" y="22" font-size="14" font-weight="500" fill="#412402">注意力改造谱系（复杂度由高到低）</text>
  <line x1="40" y1="70" x2="640" y2="70" stroke="#C9C3B6" stroke-width="2"/>
  <g font-size="11" text-anchor="middle">
    <rect x="20" y="52" width="96" height="36" rx="6" fill="#FBE3E3" stroke="#C0492F" stroke-width="0.6"/>
    <text x="68" y="70" fill="#7A2415">Full Attn</text>
    <text x="68" y="84" fill="#7A2415" font-size="10">O(n²)</text>
    <rect x="126" y="52" width="96" height="36" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.6"/>
    <text x="174" y="70" fill="#0C447C">SWA</text>
    <text x="174" y="84" fill="#0C447C" font-size="10">O(n·w)</text>
    <rect x="232" y="48" width="100" height="44" rx="6" fill="#FFF3D6" stroke="#BA7517" stroke-width="1"/>
    <text x="282" y="68" fill="#633806" font-weight="600">SWA 混合</text>
    <text x="282" y="84" fill="#633806" font-size="10">Full+SWA 层</text>
    <rect x="342" y="52" width="96" height="36" rx="6" fill="#EAF4EA" stroke="#2F7A35" stroke-width="0.6"/>
    <text x="390" y="70" fill="#1C4A20">稀疏 MSA</text>
    <text x="390" y="84" fill="#1C4A20" font-size="10">≪O(n²)</text>
    <rect x="448" y="52" width="100" height="36" rx="6" fill="#EFEAF8" stroke="#534AB7" stroke-width="0.6"/>
    <text x="498" y="70" fill="#26215C">压缩 CSA+HCA</text>
    <text x="498" y="84" fill="#26215C" font-size="10">O(n·c)</text>
    <rect x="558" y="52" width="104" height="36" rx="6" fill="#E8F0F0" stroke="#2C7A7B" stroke-width="0.6"/>
    <text x="610" y="70" fill="#164B4B">Mamba/SSM</text>
    <text x="610" y="84" fill="#164B4B" font-size="10">O(n)</text>
  </g>
  <text x="20" y="120" font-size="11" fill="#444441">箭头方向：注意力 KV / 算力成本递减，记忆与长程能力递减；SWA 混合在"省钱"与"不忘"之间取折中。</text>
  <text x="20" y="150" font-size="11" fill="#444441">注：此谱系按"改造强度/成本"排列，非严格单调；具体模型常混合多种（如 V4 同时有 CSA/HCA + 滑窗兜底）。</text>
</svg>
</div>

## 3. 三家路线速写

| 模型 | 发布 | 核心抓手 | 上下文 | 参数量级 | 开放 |
|---|---|---|---|---|---|
| **Kimi K3** | 2026 | 重设计注意力（KDA + Gated MLA 3:1）+ Stable LatentMoE + 原生视觉 | 1M | 2.8T MoE | 权重开源 |
| **MiniMax M3** | 2026-06 | 稀疏注意力（MSA）+ 原生多模态 + computer use | 1M（M2.7 仅 200K） | 未公开（MoE） | 权重开源 |
| **DeepSeek V4** | 2026-04 | 压缩注意力（CSA + HCA）+ Muon 优化器 + OPD 蒸馏 | 1M（默认） | Pro 1.6T / Flash 284B（MoE） | MIT 开源 |

- **Kimi K3**——"把注意力重新设计一遍"：用 Kimi Delta Attention（KDA）配合 Gated MLA（3:1 混合），加 Attention Residuals 稳住训练；MoE 用 Stable LatentMoE（896 个路由专家、激活 16 个），视觉端用从零训练的 MoonViT-V2 做原生多模态。官方口径：相对 K2.5 的**缩放效率约 2.5×**。
- **MiniMax M3**——"把注意力变稀疏"：在 M2.7（标准注意力、200K、纯文本、仅 API）基础上，换上 MiniMax Sparse Attention（MSA），1M 上下文下**每 token 算力只有 M2.7 的约 1/20**，prefill 快约 9×、decode 快约 15×；同时补齐原生多模态（文本+图像+视频）和 computer use，并开源权重。
- **DeepSeek V4**——"把 KV 压到极致"：CSA（压缩稀疏注意力，4-token KV 块 + top-k）+ HCA（重度压缩注意力，128×）混合，外加 128-token 不压缩滑窗分支兜底；1M 上下文下**每 token 算力约 V3.2 的 27%、KV Cache 约 10%**；训练侧换上 Muon 优化器替代 AdamW，并用 OPD（On-Policy Distillation）蒸馏领域专家。

## 4. 循序渐进路线图

本系列刻意按"注意力改造强度"由轻到重排：

1. **（二）Kimi K3**：在 MLA 基础上**加一个差分项（KDA）**、把 MLA 与门控按 3:1 结合——属于"温和改注意力 + 重做 MoE/视觉"，最好懂，适合作为入门。
2. **（三）MiniMax M2.7→M3**：先看 M2.7 的**标准注意力** baseline，再看 M3 如何把它**稀疏化**——这是"注意力从稠密到稀疏"的现场演进，最直观地展示稀疏化的收益。
3. **（四）DeepSeek V4**：直接**压缩 KV 本身**（CSA/HCA），再叠加 Muon、OPD、FP8/FP4——改造最激进、工程最重，放在最后压轴。

这样读完，你会自然建立起一条主线：**稠密注意力 → 稀疏注意力 → 压缩注意力**，最终三家都收敛到"MoE + 1M 上下文"这一共同终点。

## 5. 贯穿全系列的主线 & 投资视角

三家的共同点远比差异重要：

- **都走向 MoE + 超长上下文**：稠密大模型的时代基本结束，1M 上下文成为新基线。
- **都改造注意力而非堆算力**：破局点在"注意力机制"这一层，不在单纯加卡。
- **都开源权重**（MIT / Apache）：前沿能力正在"商品化"，差异化从"模型能力"转向"工程、数据、生态"。

> 投资视角：注意力的"摩尔定律"（同样上下文下算力/KV 逐年大幅下降）直接利好**推理侧降本**——每降一档，端侧 / 机载实时 VLA、Agent 长程任务就多一分可行，这正是具身智能与 AI 应用的成本拐点。下一章从 Kimi K3 的"重设计注意力"开始，逐家拆解。
