---
title: DSpark 深度解析：半自回归起草 + 马尔可夫头 + 置信度调度
description: 拆解 DSpark（DeepSeek + 北大开源）的半自回归草稿、马尔可夫头顺序建模与置信度调度器，以及它在 vLLM / SGLang 上的落地。
pubDate: 2026-07-31
series: 推测解码手记
lang: zh
altLang: en
altHref: /en/blog/ep4-dspark
layout: ../../layouts/BlogPost.astro
---

## 1. 背景与来源

**DSpark** 由 **DeepSeek + 北京大学**于 2026-06-27 开源（MIT 协议，仓库 `deepseek-ai/DeepSpec`）。它与 DFlash 走的是**不同路线**：不靠扩散去噪，而是用**半自回归 + 马尔可夫头**来建顺序、用**置信度调度**来动态控验证长度。

> 先纠正一个常见混淆：**DSpark 不是 MTP，也不是简单地"一个 Draft 模型自回归生成多个 token"**。DSpark 的 Draft 核心是「DFlash 的 block-parallel backbone + 轻量 Markov Head」，再加一个 Confidence Head——即"并行生成 + 轻量序列依赖 + 置信度调度"。官方论文与 vLLM Speculators 文档都是这么描述的。MTP 是模型原生、训练期嵌入的多令牌头；DSpark 是独立的轻量 drafter（共享 embedding/LM head、目标冻结）。完整链路见 [Ep7：MTP Head 与置信度头](../ep7-mtp-dspark)。

## 2. 核心一：Semi-Autoregressive Drafter + Markov Head

- **半自回归（semi-autoregressive）**：以"块"为单位并行出块，块间可以并行，块内仍有顺序；
- **马尔可夫头（Markov head）**：在块内注入 token 间的顺序依赖，保证块内生成顺序一致、不至于乱序。

这与 DFlash 的扩散范式形成对照：DSpark 在"自回归 ↔ 纯并行"之间取折中——比纯自回归并行、比纯扩散更可控。

## 3. 核心二：Confidence Scheduler（置信度调度器）

这是 DSpark 最巧妙的一笔：

- 不固定 block size，而是**根据草稿置信度动态决定本次验证多少 token**；
- **高置信** → 多验证（一次多换几个 token）；**低置信** → 少验证（避免把整块都喂进去却大量被拒的浪费）。

相对 DFlash 固定 block size=16，DSpark 的验证长度随"当前该不该信草稿"自适应，进一步逼近 α 的理论上限。

## 3.5 Confidence Head 的精确机制（c_k 与 prefix survival probability）

上一节只说了"高置信多验证"，这里补上精确数学。Confidence Head 是一个轻量 head（`hidden state → Linear → sigmoid → c_k`），输出每个草稿位置 k 的标量置信度：

> **c_k = P(token k 被接受 | 前面的 token 都被接受)**

即"在前缀都正确的条件下，第 k 个 token 继续被 Target 接受的条件概率"，监督信号来自草稿分布与目标分布的 TV 距离。真正关心的是"前面这一整段能不能全部活下来"——把条件概率连乘得到 **prefix survival probability**：

> **a_{r,j} = ∏_{i=1}^{j} c_{r,i}**

它随 j 单调递减（第 5 个 token 值不值得验证，取决于前面 1~4 个是否先通过）。Hardware-Aware Scheduler 按 a_j 降序贪心加入候选、预期接受 token 饱和即早停，从而动态决定本轮验证长度。

<div class="fig">
<svg viewBox="0 0 680 400" width="100%" role="img" font-family="-apple-system,PingFang SC,Microsoft YaHei,sans-serif">
  <title>Confidence Head 预测 prefix survival probability</title>
  <desc>草稿 backbone 一次前向出 γ 个隐藏态，每个位置经 Confidence Head 输出条件接受概率 c_k，前缀存活概率 a_j 为 c_1..c_j 的连乘，单调递减，调度器按 a_j 降序贪心截断尾部。</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="20" y="22" font-size="15" font-weight="500" fill="#412402">Confidence Head → Prefix Survival Probability</text>

  <rect x="24" y="52" width="84" height="42" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
  <text x="66" y="77" font-size="13" font-weight="500" fill="#0C447C" text-anchor="middle">Anchor D</text>

  <line x1="108" y1="73" x2="128" y2="73" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arrow)"/>
  <rect x="128" y="52" width="172" height="42" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
  <text x="214" y="73" font-size="13" font-weight="500" fill="#26215C" text-anchor="middle">Draft Backbone</text>
  <text x="214" y="90" font-size="11" fill="#3C3489" text-anchor="middle">1 次前向出 γ 位</text>

  <line x1="300" y1="94" x2="300" y2="108" stroke="#888780" stroke-width="1.2"/>
  <line x1="88" y1="108" x2="412" y2="108" stroke="#888780" stroke-width="1"/>
  <line x1="88" y1="108" x2="88" y2="118" stroke="#888780" stroke-width="1"/>
  <line x1="168" y1="108" x2="168" y2="118" stroke="#888780" stroke-width="1"/>
  <line x1="248" y1="108" x2="248" y2="118" stroke="#888780" stroke-width="1"/>
  <line x1="328" y1="108" x2="328" y2="118" stroke="#888780" stroke-width="1"/>
  <line x1="412" y1="108" x2="412" y2="118" stroke="#888780" stroke-width="1"/>

  <g>
    <rect x="62" y="118" width="52" height="38" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="88" y="137" font-size="12" fill="#0C447C" text-anchor="middle">h_1</text>
    <rect x="142" y="118" width="52" height="38" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="168" y="137" font-size="12" fill="#0C447C" text-anchor="middle">h_2</text>
    <rect x="222" y="118" width="52" height="38" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="248" y="137" font-size="12" fill="#0C447C" text-anchor="middle">h_3</text>
    <rect x="302" y="118" width="52" height="38" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="328" y="137" font-size="12" fill="#0C447C" text-anchor="middle">h_4</text>
    <rect x="386" y="118" width="52" height="38" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="412" y="137" font-size="12" fill="#0C447C" text-anchor="middle">h_5</text>
  </g>

  <g>
    <line x1="88" y1="156" x2="88" y2="176" stroke="#BA7517" stroke-width="1.2" marker-end="url(#arrow)"/>
    <line x1="168" y1="156" x2="168" y2="176" stroke="#BA7517" stroke-width="1.2" marker-end="url(#arrow)"/>
    <line x1="248" y1="156" x2="248" y2="176" stroke="#BA7517" stroke-width="1.2" marker-end="url(#arrow)"/>
    <line x1="328" y1="156" x2="328" y2="176" stroke="#BA7517" stroke-width="1.2" marker-end="url(#arrow)"/>
    <line x1="412" y1="156" x2="412" y2="176" stroke="#BA7517" stroke-width="1.2" marker-end="url(#arrow)"/>
  </g>

  <g>
    <rect x="62" y="176" width="52" height="42" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="88" y="196" font-size="12" fill="#633806" text-anchor="middle">Conf</text>
    <text x="88" y="211" font-size="11" fill="#854F0B" text-anchor="middle">Head</text>
    <rect x="142" y="176" width="52" height="42" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="168" y="196" font-size="12" fill="#633806" text-anchor="middle">Conf</text>
    <text x="168" y="211" font-size="11" fill="#854F0B" text-anchor="middle">Head</text>
    <rect x="222" y="176" width="52" height="42" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="248" y="196" font-size="12" fill="#633806" text-anchor="middle">Conf</text>
    <text x="248" y="211" font-size="11" fill="#854F0B" text-anchor="middle">Head</text>
    <rect x="302" y="176" width="52" height="42" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="328" y="196" font-size="12" fill="#633806" text-anchor="middle">Conf</text>
    <text x="328" y="211" font-size="11" fill="#854F0B" text-anchor="middle">Head</text>
    <rect x="386" y="176" width="52" height="42" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="412" y="196" font-size="12" fill="#633806" text-anchor="middle">Conf</text>
    <text x="412" y="211" font-size="11" fill="#854F0B" text-anchor="middle">Head</text>
  </g>

  <g>
    <text x="88" y="246" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">c_1</text>
    <text x="168" y="246" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">c_2</text>
    <text x="248" y="246" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">c_3</text>
    <text x="328" y="246" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">c_4</text>
    <text x="412" y="246" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">c_5</text>
  </g>

  <text x="20" y="276" font-size="12" fill="#444441">c_k = P(接受第 k 个 | 前 k-1 个都被接受) —— 条件接受概率，监督信号来自草稿/目标分布 TV 距离</text>

  <g>
    <rect x="70" y="270" width="40" height="60" fill="#FAC775" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="150" y="282" width="40" height="48" fill="#FAC775" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="230" y="294" width="40" height="36" fill="#FAC775" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="310" y="306" width="40" height="24" fill="#FAC775" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="390" y="316" width="40" height="14" fill="#FAC775" stroke="#BA7517" stroke-width="0.5"/>
  </g>
  <text x="90" y="348" font-size="11" fill="#633806" text-anchor="middle">a_1</text>
  <text x="170" y="348" font-size="11" fill="#633806" text-anchor="middle">a_2</text>
  <text x="250" y="348" font-size="11" fill="#633806" text-anchor="middle">a_3</text>
  <text x="330" y="348" font-size="11" fill="#633806" text-anchor="middle">a_4</text>
  <text x="410" y="348" font-size="11" fill="#633806" text-anchor="middle">a_5</text>

  <text x="20" y="372" font-size="12" fill="#444441">prefix survival prob: a_j = c_1·c_2·…·c_j（前 j 个全部被接受的概率，随 j 单调递减）</text>
  <text x="20" y="392" font-size="12" fill="#444441">调度器：按 a_j 降序贪心加入候选，预期接受 token 饱和即早停 → 截断尾部低置信 token</text>
</svg>
</div>

两点提醒：**"高置信度"= 高 c_k = 高接受/存活概率**，不是草稿模型对自己生成 token 的 softmax 概率；神经网络天然过度自信，原始 c_k 的 ECE 达 3~8%，DSpark 用 Sequential Temperature Scaling (STS) 逐位校准，把 ECE 压到约 1%，这是调度器敢直接拿 c_k 排序的前提。

## 4. 核心三：大模型并行验证，零质量损失

与 DFlash 一样，DSpark 也是**无损推测解码**：草稿块交给大模型一次并行前向验证，拒绝采样保证输出分布严格等于目标模型。

## 5. 效果与生态

| 指标 | 数值 / 状态 |
|---|---|
| V4 加速 | **57–85%**（据 DeepSeek / 北大发布资料） |
| 吞吐 | **+400%** |
| vLLM 落地 | PR **#46995** 在合入（复用稀疏 MLA 非因果索引、`DSparkSpeculator` 继承 `DFlash`、Triton 非因果 SWA 内核） |
| 多后端 | 同时支持 SGLang / OpenInfer |

## 6. 与 DFlash 的关系（预告）

DSpark 改的是"草稿顺序建模 + 验证长度调度"，DFlash 改的是"草稿并行范式 + 执行引擎"——二者**不在同一层，正交可叠加**。这正是 OpenInfer 能在 Qwen3-4B 上**同时**挂两套路径的原因。完整对比见 [DFlash vs DSpark](../ep5-dflash-vs-dspark)。

> 本篇属于「推测解码手记」系列 Ep4。前情：[DFlash 深度解析](../ep3-dflash)；对比篇：[DFlash vs DSpark](../ep5-dflash-vs-dspark)。
