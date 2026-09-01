---
title: MTP Head 与置信度头：把推测解码三条自草稿路线串成一条链
description: 从 MTP 多令牌预测头讲起，串起 EAGLE-3 / DFlash / DSpark 三条自草稿路线，拆解 DSpark 的并行 backbone + Markov Head + Confidence Head 与前缀存活概率调度。
pubDate: 2026-08-19
series: 推测解码手记
lang: zh
altLang: en
altHref: /en/blog/ep7-mtp-dspark
layout: ../../layouts/BlogPost.astro
---

> 一句话主线：**MTP 让模型在训练期学会"一次 forward 多预测几个 token"；EAGLE-3 用专门训练的 Draft Head 自回归地猜；DFlash 一口气并行猜一整块；DSpark = DFlash 并行 backbone + 轻量 Markov Head + Confidence Head + 负载感知调度。** 三者都是"自草稿、不养独立小 LM"，只是草稿拓扑不同。

先纠正一个最容易混淆的点：**DSpark 不是 MTP，也不是简单地"一个 Draft 模型自回归生成多个 token"**。DSpark 的 Draft 核心是「DFlash 的 block-parallel backbone + 轻量 Markov Head」，再加一个 Confidence Head——即"并行生成 + 轻量序列依赖 + 置信度调度"。官方论文与 vLLM Speculators 文档都是这么描述的。下面把这整条链路拆开讲。

## 一、MTP Head 到底是什么

MTP = **Multi-Token Prediction**。普通 LM Head 一次只出下一个 token：

```text
A B C D → Transformer → LM Head → P(E | ABCD) → E
```

而 MTP 在原模型上额外挂多个 prediction head / MTP module，一个 forward 同时得到一串未来 token：

```text
A B C D → Transformer
              ├─ LM Head      → E
              ├─ MTP-1 Head   → F
              ├─ MTP-2 Head   → G
              └─ MTP-3 Head   → H
```

于是一次 forward 得到 `E F G H`。核心思想就是：**在训练阶段显式训练模型预测未来多个 token**（vLLM Speculators 对 MTP 的定义也是"finetune 模型原生的 multi-token prediction head"）。

<div class="fig">
<svg viewBox="0 0 680 330" width="100%" role="img" font-family="-apple-system,PingFang SC,Microsoft YaHei,sans-serif">
  <title>MTP Head 结构（DeepSeek-V3 多令牌预测模块）</title>
  <desc>主 Transformer 隐藏态 + 前移一步 token 嵌入，经 MTP Transformer Block、RMSNorm、共享 LM Head，额外预测一个未来 token；D 个模块可堆叠实现多令牌监督。</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="20" y="22" font-size="15" font-weight="500" fill="#173404">MTP Head 结构（DeepSeek-V3 多令牌预测模块）</text>

  <rect x="24" y="54" width="150" height="58" rx="8" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="99" y="80" font-size="14" font-weight="500" fill="#27500A" text-anchor="middle">主 Transformer</text>
  <text x="99" y="98" font-size="12" fill="#3B6D11" text-anchor="middle">共享词表 / LM Head</text>

  <line x1="174" y1="83" x2="256" y2="107" stroke="#185FA5" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="190" y="78" font-size="12" fill="#0C447C">h_t</text>
  <text x="190" y="94" font-size="11" fill="#5F5E5A">pos t 隐藏态</text>

  <rect x="96" y="150" width="92" height="44" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="142" y="172" font-size="13" font-weight="500" fill="#633806" text-anchor="middle">Emb(t+1)</text>
  <text x="142" y="188" font-size="11" fill="#854F0B" text-anchor="middle">前移一步 token</text>
  <line x1="188" y1="172" x2="256" y2="172" stroke="#BA7517" stroke-width="1.5" marker-end="url(#arrow)"/>

  <rect x="236" y="50" width="250" height="210" rx="10" fill="#F4FAEF" stroke="#639922" stroke-width="1"/>
  <text x="361" y="72" font-size="14" font-weight="500" fill="#27500A" text-anchor="middle">MTP 模块 (d=1)</text>

  <rect x="256" y="86" width="210" height="44" rx="6" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="361" y="113" font-size="13" font-weight="500" fill="#27500A" text-anchor="middle">MTP Transformer Block</text>

  <line x1="361" y1="130" x2="361" y2="146" stroke="#888780" stroke-width="1.2" marker-end="url(#arrow)"/>
  <rect x="256" y="146" width="210" height="34" rx="6" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="361" y="167" font-size="13" fill="#27500A" text-anchor="middle">RMSNorm</text>

  <line x1="361" y1="180" x2="361" y2="196" stroke="#888780" stroke-width="1.2" marker-end="url(#arrow)"/>
  <rect x="256" y="196" width="210" height="34" rx="6" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="361" y="217" font-size="13" fill="#27500A" text-anchor="middle">LM Head (shared)</text>

  <line x1="486" y1="213" x2="548" y2="213" stroke="#639922" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="552" y="209" font-size="12" fill="#3B6D11">预测</text>
  <text x="552" y="228" font-size="15" font-weight="500" fill="#173404">t+2</text>

  <text x="236" y="284" font-size="12" fill="#444441">D 个 MTP 模块可堆叠：MTP1→t+2, MTP2→t+3, …, MTP-D→t+D+1（训练期多令牌监督）</text>
  <text x="236" y="304" font-size="12" fill="#444441">推理期：MTP1 复用主模型隐藏态，额外预测 1 个 token 作投机草稿头（近乎零成本）</text>
</svg>
</div>

> 注意：MTP Head 并不是"4 个完全独立的 Linear"。现代 MTP 会让后面的预测位置利用前面的 hidden/token 信息，不同模型的具体 MTP module 结构不同。

## 二、为什么 MTP 能用于 Speculative Decoding

MTP 预测出 `E F G H` 后，Target / verifier 可以**并行验证**整条：

```text
A B C D → E ✓
             F ✓
             G ✓
             H ✗
```

最终接受 `E F G`、丢掉 `H`，并让 Target 在 `G` 之后纠正。所以 **MTP 本身就是现成的 speculative decoding Drafter**——DeepSeek 系列里常见的「Target Model 挂 MTP modules → 并行 verify」就是这么来的。

## 三、EAGLE-3 和 MTP 有什么区别

这是最关键的一层。EAGLE-3 属于**专门训练出来的 Draft Head / Speculator**，不是给原模型外挂几个 MTP Head：

```text
Target Model → hidden states → Eagle3 Draft Head → 自回归生成 t1→t2→t3→t4 → Target Verify
```

EAGLE-3 的 draft token 之间有**明显的序列依赖**（vLLM 文档把它描述成 "autoregressively predict draft tokens using Llama-style draft layers"）。而 MTP 是模型原生、训练期就嵌入的多令牌头。二者都能当 Drafter，但来源和草稿拓扑不同。

## 四、DFlash 又是什么

DFlash 走完全不同的方向——**一次 forward 直接并行预测整个 block**：

```text
Eagle3（串行）：   DFlash（并行）：
t1                Input ┬→ t1
 ↓                  ├→ t2
t2                 ├→ t3
 ↓                  ├→ t4
t3                 ├→ t5
 ↓                  └→ t6
t4
```

DFlash 优势是**非常快**；代价是 token 与 token 之间依赖不足，越往后接受率越低——这就是 **acceptance decay / suffix decay**（DSpark 论文明确指出，纯 parallel drafter 的主要短板就是块内 token dependency 不足）。

## 五、DSpark 到底怎么解决

DSpark 的主体仍是 DFlash 的并行生成，但在上面加一个**轻量 Markov Head**，让第 k 个 token 感知前一个 token：

```text
Anchor → DFlash Parallel Backbone → t1 t2 t3 ... tN
                                      │
                                 Markov Head
                                      │
                          加入 token-to-token 局部依赖
```

即"**并行 backbone + 很轻的局部序列依赖**"，这也是论文标题 **Semi-Autoregressive Generation** 的由来。

## 六、Markov Head 为什么有用

纯并行时，每个 `t_k` 只依赖 context；加 Markov Head 后，`t_k` 还依赖前一个 token `t_{k-1}`。官方实现里 Markov Head 是一个**低秩 logit bias**：

```text
B = W1 @ W2      （默认 rank = 256）
```

根据前一个 token 对当前 draft logits 加 bias，开销几乎为零，却补上了并行草稿尾部 incoherent / 易被拒的短板。

## 七、然后才轮到 Confidence Head

Confidence Head 是一个轻量 head：

```text
hidden state → Linear → sigmoid → c_k
```

其中：

```text
c_k = P(token k 被接受 | 前面的 token 都被接受)
```

这非常重要——它不是"这个 token 自己有多大概率"，而是"**如果前面全部正确，那么这个 token 继续正确的条件概率是多少**"。

## 八、"高置信度"是不是就是"高概率"

**是，但要非常精确地说**：这里的"高置信度" = Confidence Head 预测的 **acceptance probability 高**。例如 `c1=0.95, c2=0.90, c3=0.85, c4=0.40`——它对应的是

```text
P(t1 被接受 | 前面正确)
P(t2 被接受 | t1 正确)
P(t3 被接受 | t1,t2 正确)
P(t4 被接受 | t1,t2,t3 正确)
```

而不是 `P(t1), P(t2), P(t3), P(t4)` 这些 token 自身的生成概率。所以严格叫 **conditional acceptance probability**。

## 九、为什么叫 Prefix Survival Probability

真正关心的是"前面这一整段能不能全部活下来"。把条件概率连乘：

```text
Prefix 1: 0.95
Prefix 2: 0.95 × 0.90 = 0.855
Prefix 3: 0.95 × 0.90 × 0.85 = 0.727
Prefix 4: 0.95 × 0.90 × 0.85 × 0.40 = 0.291
```

DSpark 论文给出的定义就是 `a_{r,j} = ∏_{i=1}^{j} c_{r,i}`——prefix survival probability 是各位置 conditional probability 的连乘。

<div class="fig">
<svg viewBox="0 0 680 400" width="100%" role="img" font-family="-apple-system,PingFang SC,Microsoft YaHei,sans-serif">
  <title>Confidence Head 预测 prefix survival probability</title>
  <desc>草稿 backbone 一次前向出 γ 个隐藏态，每个位置经 Confidence Head 输出条件接受概率 c_k，前缀存活概率 a_j 为 c_1..c_j 的连乘，单调递减，调度器按 a_j 降序贪心截断尾部。</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="20" y="22" font-size="15" font-weight="500" fill="#412402">Confidence Head → Prefix Survival Probability（DSpark 验证调度核心）</text>
  <text x="380" y="22" font-size="11" fill="#3B6D11">即「Confidence Head ↓ 预测 prefix survival probability」</text>

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

## 十、DSpark 的完整结构

把上面串起来，端到端流程是：Target 先生成 anchor → DSpark 并行出 `E F G H` + confidence → scheduler 截断成 prefix → Target 并行验证。

<div class="fig">
<svg viewBox="0 0 680 380" width="100%" role="img" font-family="-apple-system,PingFang SC,Microsoft YaHei,sans-serif">
  <title>DSpark 端到端流水线</title>
  <desc>Target 生成 Anchor，DFlash 并行 backbone 一次出块，Markov Head 加局部依赖，Confidence Head 出 c_k，调度器按前缀存活概率决定验证长度，回到 Target 并行验证。</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="20" y="22" font-size="15" font-weight="500" fill="#26215C">DSpark 端到端流水线</text>

  <rect x="250" y="30" width="180" height="40" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
  <text x="340" y="54" font-size="13" font-weight="500" fill="#0C447C" text-anchor="middle">Target Model（生成 Anchor）</text>

  <line x1="340" y1="70" x2="340" y2="80" stroke="#888780" stroke-width="1.2" marker-end="url(#arrow)"/>
  <rect x="300" y="80" width="80" height="34" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
  <text x="340" y="101" font-size="13" font-weight="500" fill="#0C447C" text-anchor="middle">Anchor D</text>

  <line x1="340" y1="114" x2="340" y2="126" stroke="#888780" stroke-width="1.2" marker-end="url(#arrow)"/>
  <rect x="225" y="126" width="230" height="44" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
  <text x="340" y="148" font-size="13" font-weight="500" fill="#26215C" text-anchor="middle">DFlash Parallel Backbone</text>
  <text x="340" y="166" font-size="11" fill="#3C3489" text-anchor="middle">1 次前向，块并行</text>

  <line x1="340" y1="170" x2="340" y2="184" stroke="#888780" stroke-width="1"/>
  <line x1="138" y1="184" x2="458" y2="184" stroke="#888780" stroke-width="1"/>
  <line x1="138" y1="184" x2="138" y2="190" stroke="#888780" stroke-width="1"/>
  <line x1="218" y1="184" x2="218" y2="190" stroke="#888780" stroke-width="1"/>
  <line x1="298" y1="184" x2="298" y2="190" stroke="#888780" stroke-width="1"/>
  <line x1="378" y1="184" x2="378" y2="190" stroke="#888780" stroke-width="1"/>
  <line x1="458" y1="184" x2="458" y2="190" stroke="#888780" stroke-width="1"/>
  <g>
    <rect x="110" y="190" width="56" height="34" rx="5" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
    <text x="138" y="211" font-size="12" fill="#26215C" text-anchor="middle">t_1</text>
    <rect x="190" y="190" width="56" height="34" rx="5" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
    <text x="218" y="211" font-size="12" fill="#26215C" text-anchor="middle">t_2</text>
    <rect x="270" y="190" width="56" height="34" rx="5" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
    <text x="298" y="211" font-size="12" fill="#26215C" text-anchor="middle">t_3</text>
    <rect x="350" y="190" width="56" height="34" rx="5" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
    <text x="378" y="211" font-size="12" fill="#26215C" text-anchor="middle">t_4</text>
    <rect x="430" y="190" width="56" height="34" rx="5" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
    <text x="458" y="211" font-size="12" fill="#26215C" text-anchor="middle">t_N</text>
  </g>

  <line x1="138" y1="224" x2="138" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="218" y1="224" x2="218" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="298" y1="224" x2="298" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="378" y1="224" x2="378" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="458" y1="224" x2="458" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="138" y1="236" x2="458" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="340" y1="236" x2="340" y2="248" stroke="#888780" stroke-width="1"/>

  <rect x="225" y="248" width="230" height="38" rx="8" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="340" y="268" font-size="13" font-weight="500" fill="#27500A" text-anchor="middle">Markov Head</text>
  <text x="340" y="284" font-size="11" fill="#3B6D11" text-anchor="middle">prev token → logit bias (B=W1W2, r=256)</text>

  <line x1="340" y1="286" x2="340" y2="298" stroke="#888780" stroke-width="1.2" marker-end="url(#arrow)"/>
  <rect x="225" y="298" width="230" height="34" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="340" y="319" font-size="13" font-weight="500" fill="#633806" text-anchor="middle">Confidence Head → c_k (sigmoid)</text>

  <line x1="340" y1="332" x2="340" y2="344" stroke="#888780" stroke-width="1.2" marker-end="url(#arrow)"/>
  <rect x="225" y="344" width="230" height="32" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="340" y="364" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">Hardware-aware Scheduler（按 a_j 截断）</text>

  <path d="M455 360 C 580 360 580 50 430 50" fill="none" stroke="#BA7517" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#arrow)"/>
  <text x="470" y="210" font-size="11" fill="#854F0B">回到 Target 并行验证</text>
  <text x="470" y="226" font-size="11" fill="#854F0B">accept / reject</text>
</svg>
</div>

## 十一、DSpark 与 EAGLE-3 / DFlash / MTP 的关系

把四个放到一张图：**EAGLE-3 = 浅层自回归 drafter；DFlash = 并行 drafter；MTP = 原生多令牌头；DSpark = DFlash + Markov Head + Confidence Head + Hardware-aware Scheduler**。所以 DSpark 与 EAGLE-3 的关系**不是"DSpark = 改进版 EAGLE-3"**，而是"在并行草稿（DFlash）上叠顺序头与置信度调度"。官方 Speculators 文档明确把 DSpark 定义成 "extends DFlash with Markov head + confidence head"。

<div class="fig">
<svg viewBox="0 0 680 410" width="100%" role="img" font-family="-apple-system,PingFang SC,Microsoft YaHei,sans-serif">
  <title>DSpark、EAGLE-3 树、MTP Head 三种自草稿范式对照</title>
  <desc>左：DSpark 并行块加顺序头产出扁平块；中：EAGLE-3 特征层自回归产出候选树；右：MTP 主模型挂多头预测偏移 token。三者均自草稿、不养独立小 LM。</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="20" y="22" font-size="15" font-weight="500" fill="#26215C">DSpark / EAGLE-3 tree / MTP Head：三种自草稿范式对照</text>

  <rect x="30" y="40" width="190" height="30" rx="6" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="125" y="60" font-size="14" font-weight="500" fill="#27500A" text-anchor="middle">DSpark</text>
  <rect x="250" y="40" width="190" height="30" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
  <text x="345" y="60" font-size="14" font-weight="500" fill="#0C447C" text-anchor="middle">EAGLE-3 (tree)</text>
  <rect x="470" y="40" width="190" height="30" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="565" y="60" font-size="14" font-weight="500" fill="#633806" text-anchor="middle">MTP Head</text>

  <g>
    <rect x="40" y="92" width="38" height="34" rx="5" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
    <text x="59" y="113" font-size="12" fill="#27500A" text-anchor="middle">E</text>
    <rect x="86" y="92" width="38" height="34" rx="5" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
    <text x="105" y="113" font-size="12" fill="#27500A" text-anchor="middle">F</text>
    <rect x="132" y="92" width="38" height="34" rx="5" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
    <text x="151" y="113" font-size="12" fill="#27500A" text-anchor="middle">G</text>
    <rect x="178" y="92" width="38" height="34" rx="5" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
    <text x="197" y="113" font-size="12" fill="#27500A" text-anchor="middle">H</text>
    <line x1="78" y1="109" x2="86" y2="109" stroke="#639922" stroke-width="1.4" marker-end="url(#arrow)"/>
    <line x1="124" y1="109" x2="132" y2="109" stroke="#639922" stroke-width="1.4" marker-end="url(#arrow)"/>
    <line x1="170" y1="109" x2="178" y2="109" stroke="#639922" stroke-width="1.4" marker-end="url(#arrow)"/>
    <text x="125" y="152" font-size="12" fill="#3B6D11" text-anchor="middle">γ 个扁平 token（块）</text>
  </g>

  <g>
    <rect x="335" y="88" width="40" height="30" rx="5" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="355" y="107" font-size="12" fill="#0C447C" text-anchor="middle">root</text>
    <rect x="295" y="134" width="36" height="28" rx="5" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="313" y="151" font-size="11" fill="#0C447C" text-anchor="middle">a</text>
    <rect x="375" y="134" width="36" height="28" rx="5" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="393" y="151" font-size="11" fill="#0C447C" text-anchor="middle">b</text>
    <line x1="355" y1="118" x2="313" y2="134" stroke="#185FA5" stroke-width="1.4" marker-end="url(#arrow)"/>
    <line x1="355" y1="118" x2="393" y2="134" stroke="#185FA5" stroke-width="1.4" marker-end="url(#arrow)"/>
    <text x="345" y="186" font-size="12" fill="#0C447C" text-anchor="middle">一棵候选树（树注意力验证）</text>
  </g>

  <g>
    <rect x="495" y="88" width="46" height="26" rx="5" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="518" y="105" font-size="11" fill="#633806" text-anchor="middle">head1</text>
    <rect x="495" y="120" width="46" height="26" rx="5" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="518" y="137" font-size="11" fill="#633806" text-anchor="middle">head2</text>
    <rect x="495" y="152" width="46" height="26" rx="5" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="518" y="169" font-size="11" fill="#633806" text-anchor="middle">headD</text>
    <line x1="541" y1="127" x2="580" y2="127" stroke="#BA7517" stroke-width="1.4" marker-end="url(#arrow)"/>
    <rect x="582" y="100" width="30" height="16" rx="3" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="582" y="124" width="30" height="16" rx="3" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="582" y="148" width="30" height="16" rx="3" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="565" y="190" font-size="12" fill="#854F0B" text-anchor="middle">D 个偏移头 → 扁平输出</text>
  </g>

  <g font-size="12" fill="#444441">
    <text x="30" y="222">· 一次前向出 γ 个位置</text>
    <text x="30" y="244">· 扁平块（非树）</text>
    <text x="30" y="266">· 独立训练的轻量 drafter</text>
    <text x="30" y="288">· 共享 embedding/LM head</text>
    <text x="30" y="310">· 含 Confidence Head 调度</text>

    <text x="250" y="222">· 自回归逐位置特征预测</text>
    <text x="250" y="244">· 动态草稿树 + 树注意力</text>
    <text x="250" y="266">· 多层融合 + Training-Time Test</text>
    <text x="250" y="288">· 加速最高 6.5×</text>
    <text x="250" y="310">· 自草稿流派的开山祖</text>

    <text x="470" y="222">· 主模型挂 D 个头（预训练）</text>
    <text x="470" y="244">· 每头预测 offset d+1</text>
    <text x="470" y="266">· 推理作草稿头（零成本）</text>
    <text x="470" y="288">· 复用主模型权重</text>
    <text x="470" y="310">· V4 MTP-1 基线被其替换</text>
  </g>

  <line x1="30" y1="338" x2="650" y2="338" stroke="#888780" stroke-width="0.5"/>
  <text x="20" y="362" font-size="12" fill="#444441">三者都「自草稿、不养独立小 LM」。DSpark 在并行草稿(DFlash)上 + 顺序头 + 置信度调度，接受长度比 EAGLE-3 +26~31%、</text>
  <text x="20" y="382" font-size="12" fill="#444441">比 DFlash +16~18%，并取代 V4 的 MTP-1 基线。EAGLE-3 是树、MTP 是偏移多头、DSpark 是扁平块——草稿拓扑不同。</text>
</svg>
</div>

## 十二、DSpark 的 Draft 是不是"一次生成多个 token"

**是。** 但要拆成两层看：

- **DFlash Backbone**：一次 forward，并行产出整个 block 的候选 token（`t1..t6`）。
- **Markov Head**：利用 `previous token → logit bias` 修正每个位置，增加 token 间依赖。

所以 DSpark 是"**并行 backbone + 轻量级 sequential dependency**"，既不是纯并行、也不是纯自回归——这正是 Semi-Autoregressive Generation。

## 十三、最容易记的对比表

| 方法 | Draft 怎么产生 | Token 间依赖 | 一次 forward | 主要问题 / 优势 |
|---|---|---|---|---|
| 普通 Decode | Target | 强 | 1 token | 慢 |
| EAGLE-3 | 自回归 Draft Head | **强** | 多步 | Draft 串行 |
| DFlash | Block Parallel | **弱** | 多 token | 后缀 acceptance decay |
| **DSpark** | DFlash + Markov | **中等** | 多 token | 兼顾速度和 acceptance |
| MTP | 原生 MTP Head | 取决于具体 MTP 结构 | 多 token | 需模型原生支持 / 训练 |

一句话收尾：**EAGLE-3 是一个一个猜但猜得准；DFlash 一口气猜一堆但后面容易猜歪；DSpark 是一口气猜一堆，同时用 Markov Head 让后面的猜测参考前一个 token，再用 Confidence Head 判断这一串到底值得验证到哪里。** 而"高置信度 = 高概率"的精确含义是：该位置在**前缀已正确的条件下被 Target 接受的概率**，把这些条件概率连乘得到 prefix survival probability，scheduler 再据此决定验证长度。论文在 DeepSeek-V4 线上流量中报告，相比生产 MTP-1 基线，per-user generation speed 提升约 **60%–85%**。

> 本篇属于「推测解码手记」系列 Ep7。前情：[DFlash 深度解析](../ep3-dflash)、[DSpark 深度解析](../ep4-dspark)、[DFlash vs DSpark](../ep5-dflash-vs-dspark)、[EAGLE-3 深度解析](../ep6-eagle3)；回到系列首页见[推测解码手记](../)。
