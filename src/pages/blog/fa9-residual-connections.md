---
title: 前沿架构解码手记（九）：残差连接十年演进——ResNet → Hyper-Connections → mHC → Attention Residuals
description: 残差连接从 2015 年定形后十年未变，大模型把它变成了瓶颈：等权相加稀释浅层信息、撑爆深层隐状态。三条改进路线中，Hyper-Connections 拓宽度但训练不稳，mHC 用双随机矩阵约束把稳定性找回来且已进 DeepSeek-V4，Attention Residuals 把深度当序列、潜力最大但工程代价最高。附十年演进路线图、Post/Pre-norm 对比图、三条路线结构对比图、mHC 双随机矩阵图解与 Kimi Linear 48B-A3B 实验表。
pubDate: 2026-09-20
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa9-residual-connections
layout: ../../layouts/BlogPost.astro
---

> 整理自谯楠《大模型结构综述：残差连接的三种改进》，2026-09-20，附图解。

## 0. 结论先行

残差连接从 2015 年定形后十年未变，但大模型把它变成了瓶颈：**等权相加会稀释浅层信息、撑爆深层隐状态**。三条改进路线：

| 路线 | 做法 | 稳定性 | 落地程度 |
|------|------|--------|----------|
| **Hyper-Connections**（字节，2024） | 残差流 C → nC，可学习深度连接 + 宽度连接 | 训练不稳（无约束混合矩阵连乘易爆） | ICLR 2025 Borderline Accept |
| **mHC**（DeepSeek，2026） | 同样拓宽，但混合矩阵约束为**双随机** | 稳定：均值守恒、谱范数 ≤ 1、连乘不炸 | **已用于 DeepSeek-V4** |
| **Attention Residuals**（Kimi，2026） | 把深度当序列，每层对前面所有层做注意力 | 实验收益最高（Full 版） | Block 版 N = 8 即可落地 |

一句话记忆：**HC / mHC 是「把残差流撑大」的同一条路线，mHC 的双随机约束是让它真正可用的那一步；AttnRes 把深度维度当序列维度，是更激进的跨越，但工程适配期大概率像 Transformer 一样长。**

## 1. 残差连接为什么成了瓶颈

残差连接的信息聚合方式非常粗糙——**把所有前层的输出无差别等权相加**，由此产生三个问题：

- **信息稀释**：浅层特征向深层传递时，相对贡献随层数线性衰减；
- **隐状态爆炸**：深层 hidden state 聚合的信息越来越多，单条 C 维残差流容纳不下；
- **表示塌缩（Representation Collapse）**：层数增加时层与层之间的表示越来越相似，新增层的增量越来越小，模型的**有效层数小于实际层数**。注意这不等于训练失败——模型能堆很深，只是更深不带来等比例收益。

<div class="fig">
<svg viewBox="0 0 660 210" role="img" aria-label="残差连接十年演进路线">
  <title>图 1：残差连接十年演进路线（2015 → 2026）</title>
  <text x="20" y="26" font-size="13" fill="#6b7280">残差连接十年演进（2015 → 2026）：同一个瓶颈，三条改进路线</text>
  <rect x="20" y="48" width="140" height="34" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="90" y="69" font-size="12" fill="#185FA5" text-anchor="middle">2015 · ResNet</text>
  <rect x="20" y="82" width="140" height="78" rx="6" fill="#f9fafb" stroke="#d1d5db"/>
  <text x="90" y="102" font-size="11" fill="#374151" text-anchor="middle">恒等映射解决梯度消</text>
  <text x="90" y="120" font-size="11" fill="#374151" text-anchor="middle">失，152 层成为可能；</text>
  <text x="90" y="138" font-size="11" fill="#374151" text-anchor="middle">但聚合方式是所有</text>
  <text x="90" y="156" font-size="11" fill="#374151" text-anchor="middle">前层等权相加</text>
  <rect x="180" y="48" width="140" height="34" rx="6" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="250" y="69" font-size="12" fill="#534AB7" text-anchor="middle">2024 · Hyper-Conns</text>
  <rect x="180" y="82" width="140" height="78" rx="6" fill="#f9fafb" stroke="#d1d5db"/>
  <text x="250" y="102" font-size="11" fill="#374151" text-anchor="middle">残差流宽度 C → nC</text>
  <text x="250" y="120" font-size="11" fill="#374151" text-anchor="middle">拆成可学习的深度连</text>
  <text x="250" y="138" font-size="11" fill="#374151" text-anchor="middle">接 + 宽度连接（流间</text>
  <text x="250" y="156" font-size="11" fill="#374151" text-anchor="middle">信息交换）</text>
  <rect x="340" y="48" width="140" height="34" rx="6" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="410" y="69" font-size="12" fill="#534AB7" text-anchor="middle">2026 · mHC</text>
  <rect x="340" y="82" width="140" height="78" rx="6" fill="#f9fafb" stroke="#d1d5db"/>
  <text x="410" y="102" font-size="11" fill="#374151" text-anchor="middle">双随机矩阵约束混合</text>
  <text x="410" y="120" font-size="11" fill="#374151" text-anchor="middle">矩阵，多流残差重新</text>
  <text x="410" y="138" font-size="11" fill="#374151" text-anchor="middle">稳定；已用于</text>
  <text x="410" y="156" font-size="11" fill="#374151" text-anchor="middle">DeepSeek-V4</text>
  <rect x="500" y="48" width="140" height="34" rx="6" fill="#faeeda" stroke="#BA7517"/>
  <text x="570" y="69" font-size="12" fill="#854F0B" text-anchor="middle">2026 · AttnRes</text>
  <rect x="500" y="82" width="140" height="78" rx="6" fill="#f9fafb" stroke="#d1d5db"/>
  <text x="570" y="102" font-size="11" fill="#374151" text-anchor="middle">把深度维度当序列维</text>
  <text x="570" y="120" font-size="11" fill="#374151" text-anchor="middle">度，每层对前面所有</text>
  <text x="570" y="138" font-size="11" fill="#374151" text-anchor="middle">层做注意力；Block</text>
  <text x="570" y="156" font-size="11" fill="#374151" text-anchor="middle">版 N = 8 即可</text>
  <g stroke="#9ca3af" stroke-width="1.5" fill="none">
    <path d="M163 65 L173 65"/><path d="M168 61 L174 65 L168 69"/>
    <path d="M323 65 L333 65"/><path d="M328 61 L334 65 L328 69"/>
    <path d="M483 65 L493 65"/><path d="M488 61 L494 65 L488 69"/>
  </g>
  <text x="20" y="192" font-size="11" fill="#6b7280">蓝 = 恒等残差（ResNet）｜ 紫 = 拓宽残差流（HC / mHC）｜ 橙 = 跨层注意力（AttnRes）</text>
</svg>
<figcaption>图 1：残差连接十年演进路线。2015 年 ResNet 定形后聚合方式十年未变；2024 年字节 Hyper-Connections 第一次把残差流撑宽；2026 年 DeepSeek mHC 用双随机矩阵把稳定性找回来并进入真实模型，Kimi Attention Residuals 则把深度维度当序列维度。</figcaption>
</div>

## 2. Post-norm 与 Pre-norm 的取舍

要理解三条改进路线，先得看清现代 LLM 的默认设置是怎么来的：

- **Post-norm（原始 Transformer）**：`x_{l+1} = LayerNorm(x_l + F(x_l))`。优点是每层强制重写表示，不易塌缩；缺点是梯度要穿过每层 LayerNorm 的**雅可比连乘**，训练初期必须精心设计 warmup。
- **Pre-norm（现代 LLM 主流）**：`x_{l+1} = x_l + F(LayerNorm(x_l))`。优点是恒等路径直连，梯度有兜底；缺点是**大主干 + 小增量**——主干逐层累加变大、单层修正项保持常数，越深越像在巨大主干上做微调，于是深层易塌缩。

<div class="fig">
<svg viewBox="0 0 660 290" role="img" aria-label="Post-norm 与 Pre-norm 的信号与梯度路径对比">
  <title>图 2：Post-norm 与 Pre-norm 的信号与梯度路径</title>
  <text x="175" y="26" font-size="13" fill="#6b7280" text-anchor="middle">Post-norm · 原始 Transformer</text>
  <text x="475" y="26" font-size="13" fill="#6b7280" text-anchor="middle">Pre-norm · 现代 LLM 主流</text>
  <rect x="90" y="46" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="175" y="65" font-size="12" fill="#185FA5" text-anchor="middle">x_l</text>
  <rect x="90" y="98" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="175" y="117" font-size="12" fill="#185FA5" text-anchor="middle">F(x_l)</text>
  <rect x="90" y="150" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="175" y="169" font-size="12" fill="#185FA5" text-anchor="middle">x_l + F(x_l)</text>
  <rect x="90" y="202" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="175" y="221" font-size="12" fill="#185FA5" text-anchor="middle">LayerNorm(·) = x_l+1</text>
  <rect x="390" y="46" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="475" y="65" font-size="12" fill="#185FA5" text-anchor="middle">x_l</text>
  <rect x="390" y="98" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="475" y="117" font-size="12" fill="#185FA5" text-anchor="middle">LayerNorm(x_l)</text>
  <rect x="390" y="150" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="475" y="169" font-size="12" fill="#185FA5" text-anchor="middle">F(LayerNorm(x_l))</text>
  <rect x="390" y="202" width="170" height="30" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="475" y="221" font-size="12" fill="#185FA5" text-anchor="middle">x_l + F(·) = x_l+1</text>
  <g stroke="#378ADD" stroke-width="1.5" fill="none">
    <path d="M175 76 L175 94"/><path d="M171 90 L175 97 L179 90"/>
    <path d="M175 128 L175 146"/><path d="M171 142 L175 149 L179 142"/>
    <path d="M175 180 L175 198"/><path d="M171 194 L175 201 L179 194"/>
    <path d="M475 76 L475 94"/><path d="M471 90 L475 97 L479 90"/>
    <path d="M475 128 L475 146"/><path d="M471 142 L475 149 L479 142"/>
    <path d="M475 180 L475 198"/><path d="M471 194 L475 201 L479 194"/>
  </g>
  <g stroke="#7f77dd" stroke-width="1.5" fill="none" stroke-dasharray="4 3">
    <path d="M90 61 C 56 61, 56 165, 84 165"/><path d="M80 161 L87 165 L80 169"/>
    <path d="M560 61 C 594 61, 594 217, 566 217"/><path d="M572 213 L565 217 L572 221"/>
  </g>
  <text x="84" y="118" font-size="10" fill="#6b7280" text-anchor="end">恒等路径</text>
  <text x="566" y="140" font-size="10" fill="#6b7280" text-anchor="start">恒等路径</text>
  <text x="175" y="252" font-size="11" fill="#6b7280" text-anchor="middle">优点：每层强制重写表示，不易塌缩</text>
  <text x="175" y="270" font-size="11" fill="#6b7280" text-anchor="middle">缺点：梯度穿 LN 雅可比连乘，需 warmup</text>
  <text x="475" y="252" font-size="11" fill="#6b7280" text-anchor="middle">优点：恒等路径直连，梯度有兜底</text>
  <text x="475" y="270" font-size="11" fill="#6b7280" text-anchor="middle">缺点：大主干 + 小增量，深层易塌缩</text>
</svg>
<figcaption>图 2：Post-norm 把 LayerNorm 放在残差相加之后，每层强制重写表示但梯度要穿过 LN 雅可比连乘；Pre-norm 把恒等路径直连到下一层，梯度有兜底，代价是主干越来越大、单层修正越来越小。</figcaption>
</div>

两者的分工很清楚：**Post-norm 保深度表达但梯度不稳；Pre-norm 梯度稳但容易塌缩**。现代 LLM 普遍选 Pre-norm，理由很务实：工程上先解决「能稳定训练」比「更充分利用深度」更要紧。

## 3. 三条改进路线

<div class="fig">
<svg viewBox="0 0 660 280" role="img" aria-label="三条改进路线的残差结构对比">
  <title>图 3：三条改进路线的残差结构对比</title>
  <text x="115" y="26" font-size="12" fill="#6b7280" text-anchor="middle">标准残差 · 1 条流 · C 维</text>
  <text x="325" y="26" font-size="12" fill="#6b7280" text-anchor="middle">mHC 多流残差 · n 条流 · nC 维</text>
  <text x="535" y="26" font-size="12" fill="#6b7280" text-anchor="middle">AttnRes 跨层注意力 · 每层直连所有前层</text>
  <path d="M220 44 L220 272" stroke="#d1d5db" stroke-dasharray="3 3"/>
  <path d="M430 44 L430 272" stroke="#d1d5db" stroke-dasharray="3 3"/>
  <rect x="40" y="64" width="62" height="22" rx="4" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="71" y="79" font-size="11" fill="#185FA5" text-anchor="middle">F1</text>
  <rect x="40" y="106" width="62" height="22" rx="4" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="71" y="121" font-size="11" fill="#185FA5" text-anchor="middle">F2</text>
  <rect x="40" y="148" width="62" height="22" rx="4" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="71" y="163" font-size="11" fill="#185FA5" text-anchor="middle">F3</text>
  <rect x="152" y="56" width="10" height="126" rx="3" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="170" y="122" font-size="10" fill="#6b7280" text-anchor="start">C 维残差流</text>
  <g stroke="#378ADD" stroke-width="1.2" fill="none">
    <path d="M71 86 L71 104"/><path d="M67 100 L71 106 L75 100"/>
    <path d="M71 128 L71 146"/><path d="M67 142 L71 148 L75 142"/>
    <path d="M102 75 L146 75"/><path d="M142 71 L149 75 L142 79"/>
    <path d="M102 117 L146 117"/><path d="M142 113 L149 117 L142 121"/>
    <path d="M102 159 L146 159"/><path d="M142 155 L149 159 L142 163"/>
  </g>
  <text x="124" y="104" font-size="10" fill="#6b7280" text-anchor="middle">等权 +1</text>
  <text x="115" y="246" font-size="11" fill="#6b7280" text-anchor="middle">权重固定为 1，等权相加</text>
  <text x="115" y="264" font-size="11" fill="#6b7280" text-anchor="middle">信息稀释 + 隐状态爆炸</text>
  <rect x="262" y="60" width="26" height="140" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <rect x="306" y="60" width="26" height="140" rx="4" fill="#e9e4fb" stroke="#7f77dd"/>
  <rect x="350" y="60" width="26" height="140" rx="4" fill="#e2dbf6" stroke="#7f77dd"/>
  <text x="325" y="48" font-size="10" fill="#6b7280" text-anchor="middle">n 条流 · nC 维</text>
  <g stroke="#7f77dd" stroke-width="1.2" fill="none">
    <path d="M290 112 L302 112"/><path d="M296 108 L289 112 L296 116"/><path d="M298 108 L305 112 L298 116"/>
    <path d="M334 112 L346 112"/><path d="M340 108 L333 112 L340 116"/><path d="M342 108 L349 112 L342 116"/>
    <path d="M290 152 L302 152"/><path d="M296 148 L289 152 L296 156"/><path d="M298 148 L305 152 L298 156"/>
    <path d="M334 152 L346 152"/><path d="M340 148 L333 152 L340 156"/><path d="M342 148 L349 152 L342 156"/>
  </g>
  <text x="325" y="98" font-size="10" fill="#6b7280" text-anchor="middle">混合矩阵 H_res（双随机）</text>
  <text x="325" y="246" font-size="11" fill="#6b7280" text-anchor="middle">拓宽残差流并做可学习混合</text>
  <text x="325" y="264" font-size="11" fill="#6b7280" text-anchor="middle">层间信息交换且数值稳定</text>
  <rect x="462" y="64" width="58" height="22" rx="4" fill="#faeeda" stroke="#BA7517"/>
  <text x="491" y="79" font-size="11" fill="#854F0B" text-anchor="middle">F4</text>
  <rect x="462" y="110" width="58" height="22" rx="4" fill="#faeeda" stroke="#BA7517"/>
  <text x="491" y="125" font-size="11" fill="#854F0B" text-anchor="middle">F3</text>
  <rect x="462" y="156" width="58" height="22" rx="4" fill="#faeeda" stroke="#BA7517"/>
  <text x="491" y="171" font-size="11" fill="#854F0B" text-anchor="middle">F2</text>
  <rect x="462" y="202" width="58" height="22" rx="4" fill="#faeeda" stroke="#BA7517"/>
  <text x="491" y="217" font-size="11" fill="#854F0B" text-anchor="middle">F1</text>
  <g stroke="#BA7517" stroke-width="1.2" fill="none">
    <path d="M520 121 C 556 121, 556 72, 542 72"/><path d="M548 68 L541 72 L548 76"/>
    <path d="M520 167 C 578 167, 578 75, 542 75"/><path d="M548 71 L541 75 L548 79"/>
    <path d="M520 213 C 598 213, 598 78, 542 78"/><path d="M548 74 L541 78 L548 82"/>
  </g>
  <text x="560" y="102" font-size="10" fill="#6b7280" text-anchor="start">α3</text>
  <text x="578" y="146" font-size="10" fill="#6b7280" text-anchor="start">α2</text>
  <text x="594" y="196" font-size="10" fill="#6b7280" text-anchor="start">α1</text>
  <text x="535" y="246" font-size="11" fill="#6b7280" text-anchor="middle">加权求和替代等权相加</text>
  <text x="535" y="264" font-size="11" fill="#6b7280" text-anchor="middle">Block 化后 N = 8 即拿到大部分收益</text>
</svg>
<figcaption>图 3：三条路线对比。标准残差只有一条 C 维流、所有前层等权相加；mHC 把残差流拓宽成 n 条 nC 维流并做可学习混合（混合矩阵约束为双随机）；AttnRes 让每层直接关注前面所有层的输出，用注意力权重替代固定的 1。</figcaption>
</div>

### 3.1 Hyper-Connections（字节，2024）——拓宽残差流

核心是把残差流从 C 维扩到 **nC 维**（n 为扩展率），并引入可学习的**深度连接**（层间加权）与**宽度连接**（流间信息交换）。一层 HC 可解耦为这两部分，输出写作 `Ĥ = Bᵀ·T(HᵀA_m)ᵀ + A_rᵀ·H`。

短板很直接：**显存墙**。残差流扩大 n 倍（常用 n = 4）意味着显存占用与 I/O 成倍增加，大规模预训练成本极高。ICLR 2025 评审给了 Borderline Accept——认可性能，但质疑理论创新性与工程代价。

### 3.2 mHC（DeepSeek，2026）——用双随机矩阵换稳定

HC 的单层形式是 `x_{l+1} = H_res·x_l + H_postᵀ·F(H_pre·x_l)`。**表达能力来自 H_res，不稳定性也来自它**——原始 HC 里 H_res 是无约束矩阵，多层连乘 ∏H_i 后放大就爆炸、衰减就消失。

mHC 的做法不是退化成单位矩阵（那样就失去了流间交换能力），而是把 H_res 约束到**双随机矩阵集合（Birkhoff polytope）**：行和 = 1、列和 = 1、元素全非负；同时 H_pre 与 H_post 施加非负约束（正负混合会造成信息抵消）。

<div class="fig">
<svg viewBox="0 0 660 268" role="img" aria-label="mHC 的双随机矩阵约束">
  <title>图 4：mHC 的双随机矩阵约束</title>
  <text x="20" y="26" font-size="12" fill="#6b7280">H_res 约束到双随机矩阵集合：行和 = 列和 = 1，元素全非负</text>
  <text x="170" y="46" font-size="11" fill="#534AB7">H_res（3 条流示例）</text>
  <rect x="170" y="52" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="197" y="76" font-size="12" fill="#534AB7" text-anchor="middle">0.3</text>
  <rect x="228" y="52" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="255" y="76" font-size="12" fill="#534AB7" text-anchor="middle">0.5</text>
  <rect x="286" y="52" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="313" y="76" font-size="12" fill="#534AB7" text-anchor="middle">0.2</text>
  <rect x="170" y="94" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="197" y="118" font-size="12" fill="#534AB7" text-anchor="middle">0.1</text>
  <rect x="228" y="94" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="255" y="118" font-size="12" fill="#534AB7" text-anchor="middle">0.3</text>
  <rect x="286" y="94" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="313" y="118" font-size="12" fill="#534AB7" text-anchor="middle">0.6</text>
  <rect x="170" y="136" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="197" y="160" font-size="12" fill="#534AB7" text-anchor="middle">0.6</text>
  <rect x="228" y="136" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="255" y="160" font-size="12" fill="#534AB7" text-anchor="middle">0.2</text>
  <rect x="286" y="136" width="54" height="38" rx="4" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="313" y="160" font-size="12" fill="#534AB7" text-anchor="middle">0.2</text>
  <text x="346" y="76" font-size="12" fill="#374151" text-anchor="start">= 1</text>
  <text x="346" y="118" font-size="12" fill="#374151" text-anchor="start">= 1</text>
  <text x="346" y="160" font-size="12" fill="#374151" text-anchor="start">= 1</text>
  <text x="197" y="192" font-size="12" fill="#374151" text-anchor="middle">= 1</text>
  <text x="255" y="192" font-size="12" fill="#374151" text-anchor="middle">= 1</text>
  <text x="313" y="192" font-size="12" fill="#374151" text-anchor="middle">= 1</text>
  <text x="170" y="212" font-size="10" fill="#6b7280" text-anchor="start">行和 = 1（每行）｜ 列和 = 1（每列）｜ 元素 ≥ 0</text>
  <path d="M396 44 L396 226" stroke="#d1d5db" stroke-dasharray="3 3"/>
  <text x="412" y="62" font-size="12" fill="#185FA5" text-anchor="start">均值守恒</text>
  <text x="412" y="80" font-size="11" fill="#6b7280" text-anchor="start">1ᵀH = 1ᵀ ⇒ 多流平均信号逐层不变</text>
  <text x="412" y="118" font-size="12" fill="#185FA5" text-anchor="start">不放大信号</text>
  <text x="412" y="136" font-size="11" fill="#6b7280" text-anchor="start">‖H‖₂ ≤ 1 ⇒ ‖Hx‖₂ ≤ ‖x‖₂，谱范数被压住</text>
  <text x="412" y="174" font-size="12" fill="#185FA5" text-anchor="start">连乘仍稳定</text>
  <text x="412" y="192" font-size="11" fill="#6b7280" text-anchor="start">对乘法封闭 ⇒ ∏H_i 仍在同一集合内</text>
  <text x="20" y="240" font-size="11" fill="#6b7280">HC（无约束）：10·x₁ − 8·x₂ + 5·x₃ → 可能失控</text>
  <text x="20" y="258" font-size="11" fill="#534AB7">mHC（双随机）：0.3·x₁ + 0.5·x₂ + 0.2·x₃ → 稳定加权平均</text>
</svg>
<figcaption>图 4：mHC 把混合矩阵 H_res 约束到双随机矩阵集合。行和与列和都为 1、元素非负，带来三个性质：多流平均信号逐层不变、谱范数不超过 1、多层连乘后仍在集合内——无约束 HC 的 10·x₁ − 8·x₂ + 5·x₃ 式组合被换成稳定的加权平均。</figcaption>
</div>

三个性质一句话概括：**均值守恒、不放大信号、连乘仍稳定**。

**生成方式**：先把 n×C 的多流状态展平做 RMSNorm，再生成三个无约束 logits，最后分别投影——`H_pre = σ(·)`、`H_post = 2σ(·)`、`H_res = Sinkhorn-Knopp(·)`（交替行列归一化，迭代约 20 次）。

工程侧配套优化（这是 mHC 真正能落地的关键）：

| 优化 | 做法 | 省什么 |
|------|------|--------|
| **Kernel Fusion** | 三个映射的投影合并成一次矩阵乘（nC → n²+2n）；RMSNorm 先乘后除；Sinkhorn-Knopp 整体放进单个 custom kernel | 一次 nC 维中间读写 + 多次 kernel launch |
| **选择性重计算** | 连续 L_r 层打包为一个 block，前向只存 block 起始输入，反向再重算轻量部分；最优 block size `L_r* ≈ √(nL/(n+2))` | 激活显存 |
| **DualPipe 通信 overlap** | mHC 在 PP 之间要传 nC 维而非 C 维，用三类 stream（常规计算 / 通信 / 高优先级小计算）重叠通信与计算 | pipeline bubble |

DeepSeek 已经把 mHC 用进 **DeepSeek-V4**，是三个方案中唯一进入真实模型的。

### 3.3 Attention Residuals（Kimi，2026）——把注意力旋转 90 度

思路完全跳出了「拓宽残差流」的框架：Kimi 团队认为**网络的深度维度和序列的时间维度本质同构**。既然序列上每个 token 能关注前面所有 token，为什么不让每一层去关注前面所有层的输出？

于是把等权求和换成加权求和：`h_l = Σ α_{i→l} · v_i`，其中 `α_{i→l} = softmax(q_lᵀ·RMSNorm(k_i))`，`q_l = w_l`（可学习），k_i = v_i 即第 i 层的输出。

Full AttnRes 理论完美，但每层都要访问前面所有层输出：计算 O(L²d)、内存与通信 O(Ld)。工程上取 **Block AttnRes**：把 L 层分成 N 个 block，block 内部沿用传统残差、block 之间用注意力残差。经验值 **N = 8** 就能拿到 Full 版的绝大部分收益，复杂度降到 Memory O(N)、Computation O(N²)。

推理侧还有一套两阶段策略：因为 query 是静态的（w_l），可以把一个 block 内 S 层的 query 打包成一个 MatMul——**Phase 1** 一次性算 inter-block attention 并返回 online-softmax 状态，**Phase 2** 在 block 内部串行算局部和并更新注意力权重，把访存从 S 次降到 1 次。

## 4. 实验结果（Kimi Linear 48B-A3B 结构）

| 激活参数 | 层数 | batch | Baseline | Block AttnRes | Full AttnRes | mHC(-lite) |
|---------|------|-------|----------|---------------|--------------|------------|
| 194M | 24 | 192 | 1.931 | 1.909 | **1.899** | 1.906 |
| 241M | 26 | 256 | 1.895 | 1.875 | **1.874** | 1.869 |
| 296M | 28 | 320 | 1.829 | 1.809 | **1.804** | 1.807 |
| 436M | 32 | 384 | 1.766 | 1.746 | **1.737** | 1.747 |
| 528M | 34 | 432 | 1.719 | 1.693 | **1.692** | 1.694 |

整体排序：**Full AttnRes > Block AttnRes ≈ mHC > Baseline**。值得注意的是 Block AttnRes（N = 8）已经非常接近 Full 版，而 mHC 在同一量级——两条路线的收益相当，区别在工程代价。

## 5. 我的判断

- **HC 与 mHC 是同一条路线**：都在 hidden_size 不变、不增加 Attention/FFN 计算量的前提下，把层间残差流维度撑大以容纳更多信息。mHC 的双随机约束是让这条路线真正可用的那一步。
- **mHC 已成为事实标准的最有力候选**：唯一进了真实模型（DeepSeek-V4），且配了 Kernel Fusion / 重计算 / DualPipe 一整套工程优化——性能没停在理论阶段。
- **AttnRes 的想象力更大，但落地更远**。作者打了个很好的比方：标准残差到 HC/mHC 类似 RNN 到 LSTM；而注意力残差更像是 RNN 到 Transformer 的跨越。注意力机制已被证明能解决长距离依赖，代价是巨大算力——Transformer 从 2017 年提出到 2022 年真正一统天下，走了五年。AttnRes 大概率也要经历同样长的工程适配期。
- **工程侧启示**：如果做推理适配（比如国产卡上的大模型推理优化），mHC 带来的残差流从 C 扩到 nC 意味着 KV Cache 之外的额外显存与跨 stage 通信，kernel fusion 与重计算是必须自己复现的；AttnRes 的 block 内 query 打包 + 两阶段 online-softmax 则是一套很值得借鉴的访存优化范式。

## 6. 与系列文章的衔接

- **fa2（Kimi K3）**：AttnRes 出自 Kimi 团队——把「用注意力改造架构」的思路从序列内推到层间，与 fa2 的注意力重设计一脉相承。
- **fa4（DeepSeek V4）**：mHC 已进入 DeepSeek-V4，是 fa4 极致压缩与高效训练路线的底层组件之一。
- **fa6（Gated DeltaNet）**：线性注意力改的是「序列内怎么算」，本文改的是「层间怎么传」——两个维度互不替代。
- **sys9 / sys12**：mHC 的 nC 残差流意味着 KV Cache 之外的额外显存与跨 stage 通信，可用那两篇的公式与工具估算。
- **op1 / op2**：RMSNorm、Sinkhorn-Knopp 归一化与 kernel fusion 的算子级实现，可对照算子系列的手记。
