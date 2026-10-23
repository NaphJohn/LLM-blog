---
title: 前沿架构解码手记（十）：MiniMax M3.1-Flash——60 层全部稀疏化，一次「不动结构」的效率升级
description: 顺着 fa3 继续拆 MiniMax。把 M3 与 M3.1-Flash 从 Embedding → 60 层骨干 → LM Head 的完整数据流画成一张图，定位两模型唯一的结构分歧点：M3 在 Layer 0–2 保留 Full Attention，M3.1-Flash 换成 MSA，60 层在长上下文下全部变成「索引选块 + 块稀疏 GQA」；MoE 层数不变，路由从 top-4+1 shared 调优为 129 experts / top-5。另有 QAT、DSpark 与 Day-0 部署生态。
pubDate: 2026-10-23
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa10-minimax-m31-flash
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么看 M3.1-Flash

fa3 拆过 M2.7 → M3 的「稀疏注意力跨越」。2026-10-19，MiniMax M3.1-Flash 先上 API；10-23，权重在 Hugging Face / ModelScope 开源（minimax-community license）——**428B 总参、每 token 激活 23B、1M 上下文**，参数量与 M3 持平，但 BF16 体积从 854GB 降到 252GB。

官方口径的相对 M3 三处改动：

1. **前三层 full-attention 全部替换为 sparse attention**（本文主角）；
2. **routed MoE experts 与 attention paths 量化感知训练（QAT）**——dense 层 MXFP8、experts NVFP4，官方权重即低精度版；
3. **DSpark 投机解码**替代 M3 的 EAGLE-like MTP，draft head 随权重一起发布。

本文只聚焦第 1 条的结构细节，回答一个问题：**两个模型的差异，到底发生在数据流的哪一格。**

## 1. 完整数据流：Embedding → 60 层 → LM Head

先给结论：**差异不在入口、不在出口、也不在 MoE 的 57 层数量上，全部集中在 Layer 0–2 的注意力路径。**

<div class="fig">
<svg viewBox="0 0 680 754" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MiniMax-M3 与 MiniMax-M3.1-Flash 从 Embedding 到 60 层骨干再到 LM Head 的完整结构对比">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Embedding → 60 层骨干 → LM Head：分歧只发生在中间</text>

  <rect x="265" y="40" width="150" height="40" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="340" y="60" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#444441">Token IDs</text>
  <line x1="340" y1="80" x2="340" y2="98" stroke="#5F5E5A" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="185" y="104" width="310" height="52" rx="10" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="340" y="123" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#444441">Embedding</text>
  <text x="340" y="143" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">vocab 200,064 · hidden 6,144</text>

  <line x1="340" y1="156" x2="340" y2="174" stroke="#888780" stroke-width="1.5"/>
  <line x1="185" y1="174" x2="495" y2="174" stroke="#888780" stroke-width="1.5"/>
  <line x1="185" y1="174" x2="185" y2="190" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="495" y1="174" x2="495" y2="190" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="40" y="196" width="290" height="34" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.2"/>
  <text x="185" y="213" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#3C3489">MiniMax-M3 · 混合注意力 60 层</text>
  <rect x="350" y="196" width="290" height="34" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="495" y="213" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#085041">MiniMax-M3.1-Flash · 全层 MSA</text>
  <line x1="185" y1="230" x2="185" y2="246" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="495" y1="230" x2="495" y2="246" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="40" y="252" width="290" height="112" rx="12" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.8"/>
  <text x="185" y="270" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#3C3489">Layer 0 – 2 · Dense 层</text>
  <rect x="52" y="284" width="128" height="70" rx="8" fill="#CECBF6" stroke="#534AB7" stroke-width="1.2"/>
  <text x="116" y="300" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#3C3489">Full Attention</text>
  <text x="116" y="317" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">64 heads · 4 KV</text>
  <text x="116" y="332" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">GQA · QK-Norm</text>
  <text x="116" y="347" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">head_dim 128</text>
  <rect x="190" y="284" width="128" height="70" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="254" y="300" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">Dense MLP</text>
  <text x="254" y="317" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">intermediate</text>
  <text x="254" y="332" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">12,288 · SwiGLU</text>
  <text x="254" y="347" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">非 MoE 层</text>
  <line x1="181" y1="319" x2="189" y2="319" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="350" y="252" width="290" height="112" rx="12" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.8"/>
  <text x="495" y="270" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">Layer 0 – 2 · Dense MLP 保留</text>
  <rect x="362" y="284" width="128" height="70" rx="8" fill="#9FE1CB" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="426" y="300" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">MSA 稀疏注意力</text>
  <text x="426" y="317" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">Indexer 4 heads</text>
  <text x="426" y="332" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">top-16 块选择</text>
  <text x="426" y="347" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">block-sparse GQA</text>
  <rect x="500" y="284" width="128" height="70" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="564" y="300" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">Dense MLP</text>
  <text x="564" y="317" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">intermediate</text>
  <text x="564" y="332" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">12,288 · SwiGLU</text>
  <text x="564" y="347" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">未替换为 MoE</text>
  <line x1="491" y1="319" x2="499" y2="319" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>

  <line x1="185" y1="364" x2="185" y2="382" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="495" y1="364" x2="495" y2="382" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="40" y="386" width="290" height="126" rx="12" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.8"/>
  <text x="185" y="404" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#3C3489">Layer 3 – 59 · MoE 层 ×57</text>
  <rect x="52" y="418" width="128" height="84" rx="8" fill="#CECBF6" stroke="#534AB7" stroke-width="1.2"/>
  <text x="116" y="434" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#3C3489">MSA 稀疏注意力</text>
  <text x="116" y="450" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">① Indexer 选块</text>
  <text x="116" y="465" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">② block-sparse GQA</text>
  <text x="116" y="480" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">≈2,048 tok/query</text>
  <rect x="190" y="418" width="128" height="84" rx="8" fill="#AFA9EC" stroke="#534AB7" stroke-width="1.2"/>
  <text x="254" y="434" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#3C3489">MoE FFN</text>
  <text x="254" y="450" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">128 experts</text>
  <text x="254" y="465" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">top-4 + 1 shared</text>
  <text x="254" y="480" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">scaling 2.0</text>
  <line x1="181" y1="460" x2="189" y2="460" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="350" y="386" width="290" height="126" rx="12" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.8"/>
  <text x="495" y="404" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">Layer 3 – 59 · MoE 层 ×57</text>
  <rect x="362" y="418" width="128" height="84" rx="8" fill="#9FE1CB" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="426" y="434" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">MSA 稀疏注意力</text>
  <text x="426" y="450" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">① Indexer 选块</text>
  <text x="426" y="465" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">② block-sparse GQA</text>
  <text x="426" y="480" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">≈2,048 tok/query</text>
  <rect x="500" y="418" width="128" height="84" rx="8" fill="#5DCAA5" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="564" y="434" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">MoE FFN · 融合配置</text>
  <text x="564" y="450" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">129 experts</text>
  <text x="564" y="465" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">top-5 激活</text>
  <text x="564" y="480" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">层数不变 ×57</text>
  <line x1="491" y1="460" x2="499" y2="460" stroke="#888780" stroke-width="1.5" marker-end="url(#arr)"/>

  <line x1="185" y1="512" x2="185" y2="530" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="495" y1="512" x2="495" y2="530" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="85" y="534" width="200" height="32" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="185" y="550" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#444441">Final RMSNorm · Gemma-style</text>
  <rect x="395" y="534" width="200" height="32" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="495" y="550" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#444441">Final RMSNorm · Gemma-style</text>
  <line x1="185" y1="566" x2="185" y2="574" stroke="#5F5E5A" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="495" y1="566" x2="495" y2="574" stroke="#5F5E5A" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="85" y="580" width="200" height="40" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="185" y="593" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">LM Head</text>
  <text x="185" y="609" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">200,064 × 6,144</text>
  <rect x="395" y="580" width="200" height="40" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="495" y="593" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">LM Head</text>
  <text x="495" y="609" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">200,064 × 6,144</text>

  <rect x="40" y="638" width="600" height="76" rx="12" fill="#FAEEDA" stroke="#854F0B" stroke-width="1.2"/>
  <text x="340" y="658" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#633806">关键分歧：Layer 0 – 2 的注意力路径</text>
  <text x="340" y="677" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#854F0B">M3 = Full Attention + Dense MLP ｜ Flash = MSA + Dense MLP</text>
  <text x="340" y="696" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#854F0B">Embedding · Final RMSNorm · LM Head · 57 层 MoE —— 两模型完全一致</text>
</svg>
<p class="cap">图 1：MiniMax-M3（左，紫）与 MiniMax-M3.1-Flash（右，绿）从 Embedding 到 LM Head 的完整数据流。灰色为两模型共享且不变的组件，分歧集中在 Layer 0–2 的注意力路径。</p>
</div>

### ① 入口与出口：两模型完全相同

Embedding 与 LM Head 均从 M3 原样继承：词表 **200,064**、隐藏维度 **6,144**，LM Head 为 200,064 × 6,144，未做 weight tying 的明确改动。差异不发生在「入口」和「出口」。

### ② M3 的「混合」是 Layer 0–2 与 Layer 3–59 的分界

- **Layer 0–2**：Full Attention + Dense MLP。Full Attention 为 64 heads / 4 KV heads（GQA）、head_dim = 128、per-head QK-Norm；Dense MLP 的 intermediate = 12,288，SwiGLU-OAI；
- **Layer 3–59**：MSA + MoE FFN——128 个 routed experts、top-4 激活 + 1 个 shared expert、routed_scaling_factor = 2.0。

这个分界由 `moe_layer_freq` 与 `sparse_attention_freq` 两个配置里**前三位的 0** 严格决定。Full Attention 层没有 MSA 的 Indexer 分支，走完整因果注意力。

### ③ M3.1-Flash 的改动精确到「注意力路径」

Flash 把前 3 层的 Full Attention 替换为 MSA，使 60 层全部具备稀疏注意力；但 **Layer 0–2 仍使用 Dense MLP，没有提前进入 MoE**。于是 Layer 0–2 变成 MSA + Dense MLP，Layer 3–59 仍是 MSA + MoE。MoE 侧的变化属于路由策略调优——从 top-4 + 1 shared 调为融合后的 **129 experts / top-5**——不改变 57 层 MoE 的层数本身。

### ④ MSA 不是一个注意力，而是两段式

<div class="fig">
<svg viewBox="0 0 680 396" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MSA 两段式稀疏注意力内部结构：Indexer 选块加 block-sparse GQA">
  <defs>
    <marker id="arr2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">MSA 模块内部：两段式稀疏注意力</text>

  <rect x="40" y="120" width="110" height="56" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="95" y="140" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">前文 KV</text>
  <text x="95" y="160" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">因果序列</text>
  <line x1="154" y1="148" x2="182" y2="148" stroke="#5F5E5A" stroke-width="1.5" marker-end="url(#arr2)"/>

  <rect x="186" y="93" width="160" height="110" rx="10" fill="#B5D4F4" stroke="#185FA5" stroke-width="1.2"/>
  <text x="266" y="118" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#0C447C">① Indexer</text>
  <text x="266" y="140" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">4 index heads</text>
  <text x="266" y="158" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">128-dim 索引空间</text>
  <text x="266" y="176" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">128-token 块打分</text>
  <text x="266" y="194" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">选 top-16 块</text>
  <line x1="350" y1="148" x2="378" y2="148" stroke="#185FA5" stroke-width="1.5" marker-end="url(#arr2)"/>

  <rect x="382" y="93" width="160" height="110" rx="10" fill="#B5D4F4" stroke="#185FA5" stroke-width="1.2"/>
  <text x="462" y="118" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#0C447C">② Block-sparse GQA</text>
  <text x="462" y="140" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">仅对选中块计算</text>
  <text x="462" y="158" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">GQA 64 Q / 4 KV</text>
  <text x="462" y="176" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">每 query 预算</text>
  <text x="462" y="194" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">≈2,048 token</text>
  <line x1="546" y1="148" x2="574" y2="148" stroke="#185FA5" stroke-width="1.5" marker-end="url(#arr2)"/>

  <rect x="578" y="120" width="62" height="56" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="609" y="148" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">输出</text>

  <line x1="364" y1="207" x2="364" y2="248" stroke="#888780" stroke-width="0.5" stroke-dasharray="4 3"/>

  <rect x="40" y="252" width="290" height="104" rx="12" fill="#FBEAF0" stroke="#993556" stroke-width="1.2"/>
  <text x="185" y="278" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#72243E">Full Attention（仅 M3 Layer 0–2）</text>
  <text x="185" y="300" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#993556">无 Indexer 分支</text>
  <text x="185" y="320" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#993556">直接对全量 KV 做 GQA</text>
  <text x="185" y="340" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#993556">计算量随序列长度平方增长</text>

  <rect x="350" y="252" width="290" height="104" rx="12" fill="#E1F5EE" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="495" y="278" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#085041">MSA（Flash 全 60 层 · M3 后 57 层）</text>
  <text x="495" y="300" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0F6E56">索引选块 + 块稀疏 GQA</text>
  <text x="495" y="320" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0F6E56">每 query ≈2,048 token</text>
  <text x="495" y="340" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0F6E56">长上下文无平方复杂度遗留层</text>
</svg>
<p class="cap">图 2：MSA 的两段式结构——Indexer 先按 128-token 块打分选 top-16 块，再在选中块上做 block-sparse GQA；与 Full Attention 的模块级差异在于有无 Indexer 分支。</p>
</div>

每个 MSA 层内部：轻量 **Indexer**（4 个 index heads、128 维索引空间）先按 128-token 块给前文打分，选出 top-16 个最相关的块；再在这些块上做 **block-sparse GQA**，把每个 query 的实际注意力预算锁定在约 2,048 token。Full Attention 层没有 Indexer，直接对全量 KV 做 GQA——这是 M3 的 Full Attention 层与 MSA 层在模块级最本质的差异。

<div class="keybox">
<strong>一句话概括：</strong>两个模型的分歧点不在 embedding、不在 lm_head、也不在 MoE 的 57 层数量上，而集中在 Layer 0–2 的注意力路径：M3 在这里保留 Full Attention，M3.1-Flash 把它替换为 MSA，使 60 层在长上下文下的注意力计算全部变成「索引选块 + 块稀疏 GQA」，不再有按平方复杂度运行的遗留层。
</div>

## 2. 为什么值得为「前三层」较真

前三层只占 60 层的 5%，看起来微不足道。但 Full Attention 的计算量随序列长度**平方**增长——在 1M 上下文下，这三层恰是整个骨干里唯一还在按 O(n²) 运行的部分。M3.1-Flash 把最后一处平方也抹掉，收益与 fa3 里 MSA 同源：长上下文下算力不随长度爆炸。

工程侧还有一对组合拳：

- **NVFP4 KV cache**：indexer 与 attention 路径的 KV 均为 NVFP4，vLLM 在 Blackwell 上用 CUDA 13.4 的 QMUL4 指令在 kernel 内反量化，indexer QK decode 快至 1.7×、稀疏注意力 decode 快至 1.3×；
- **Indexer decode 上下文并行（DCP）**：512K 上下文、4× GB300 下 indexer 每稀疏层快至 3.3×——与 sys7 / sys8 的 CP / DCP 切分思路同源。

## 3. 注意力之外：QAT 与 DSpark

- **QAT（量化感知训练）**：routed experts NVFP4、attention paths 量化感知训练，dense 层 MXFP8——官方权重即低精度版，不是「训练完再压缩」的事后量化。checkpoint 从 854.2GB（BF16）降到 252.4GB（含约 2.3GB 的 DSpark draft head）。
- **DSpark**：投机解码从 M3 的 EAGLE-like MTP 换成 DSpark draft head，随权重发布、serving 框架直接调用——与 ep4 / ep5 / ep8 三篇手记里的 DSpark 是同一条线，vLLM 配方里 `num_speculative_tokens`  Blackwell 默认 7、AMD MI355X 默认 3。
- **Day-0 生态**：10-23 权重上线当天，vLLM / SGLang / Hugging Face / ModelScope 全部打通；昇腾、平头哥、昆仑芯、摩尔线程、沐曦等国产芯片与 Together AI、Fireworks、Nebius 等国际云同步适配。多芯片部署走 FlagOS。

```text
vLLM（Blackwell，TP4）关键 serving 参数：
  --block-size 128                 # MSA 的 KV 块粒度，与 Indexer 的 128-token 块对齐
  --kv-cache-dtype nvfp4           # 注意力 KV cache 低精度
  --attention-config '{"indexer_kv_dtype": "nvfp4"}'
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,...}'
  --disable-hybrid-kv-cache-manager
```

## 4. 基准速览（官方口径）

| 基准 | M3.1 Flash | 参照：GPT-5.6 Sol |
|---|---|---|
| Automation-Bench v1.0.6 | 62.8 | 50.3 |
| Terminal Bench 4.0 | 38.1 | 51.8 |
| DeepSWE v1.1 | 73.7 | 74.0 |
| OSWorld V2 | 61.7 | 70.6 |
| CyberGym | 82.2 | 83.6 |
| HLE（no tools） | 43.4 | 55.6 |
| LOCA-Bench 256K | 60.3 | 60.0 |

Agentic / 工具调用类基准（Automation-Bench、BankerToolBench 76.2）是它的强项，恰好对应 M3 家族的 computer use 与长上下文定位；硬编码与推理类仍落后第一梯队。**以上均为厂商记分卡口径，独立榜单待补。**

## 5. 与系列文章的衔接

- **fa3（MiniMax M2.7 → M3）**：MSA 的首次跨越是本文的前半页；本文补上 M3 → M3.1-Flash 的「最后一平方」。
- **sys3（MSA / CSA / HCA）**：MSA 是三种注意力改造路线里「不压缩 KV、只选重要位置」的一支，本文给出它在 M3 家族里的最新形态与配置参数。
- **ep4 / ep5 / ep8（DSpark）**：M3.1-Flash 把 DSpark 做成随权重发布的 draft head，spec decoding 从论文/框架特性变成模型自带能力。
- **fw2 / fw3（vLLM / SGLang）**：M3.1-Flash 的 Day-0 serving 配方（block-size 128、NVFP4 KV、indexer DCP）可直接对照框架手记阅读。

> 下一篇进入算子讲解手记：把 MLA 多头潜在注意力写成可照着实现的逐行伪代码（op1）。
