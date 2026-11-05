---
title: 前沿架构解码手记（十）：MiniMax M3.1-Flash——60 层全部稀疏化，一次「不动结构」的效率升级
description: 顺着 fa3 继续拆 MiniMax。把 M3 与 M3.1-Flash 的 config.json 逐字段跑一遍 diff，定位两模型唯一的结构分歧点：sparse_attention_freq 从「前三位为 0」变成「60 位全为 1」，M3 在 Layer 0–2 保留 Full Attention，M3.1-Flash 换成 MSA；MoE 侧 128 experts / top-4 + 1 shared / scaling 2.0 两版逐字节相同。另有 rope_theta 翻倍、QAT 四档精度、DSpark 三段式与 Day-0 部署生态。
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
  <text x="564" y="434" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">MoE FFN · 参数不变</text>
  <text x="564" y="450" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">128 experts</text>
  <text x="564" y="465" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">top-4 + 1 shared</text>
  <text x="564" y="480" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">仅精度变 · 层数不变 ×57</text>
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

Flash 把前 3 层的 Full Attention 替换为 MSA，使 60 层全部具备稀疏注意力；但 **Layer 0–2 仍使用 Dense MLP，没有提前进入 MoE**。于是 Layer 0–2 变成 MSA + Dense MLP，Layer 3–59 仍是 MSA + MoE。

**注意 MoE 拓扑其实没动**——路由仍是 `128 experts / top-4 + 1 shared / scaling 2.0`，两版逐字节相同。这一点流传的错误说法很多，下一节用配置原文逐字段验证。

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

## 2. 把两份 config.json 摊开：真正的 diff 只有 4 行

前面那张数据流图是「读出来的结论」。但结论要能站住，得回到配置原文。我把两个仓库的 `config.json` 抓下来跑了逐字段 diff，`text_config` 部分的结果很干净：

**31 个共有字段中，29 个完全一致，只有 2 个不同；另有 2 个字段在 M3.1 中被删除。**

<div class="fig">
<svg viewBox="0 0 680 470" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="M3 与 M3.1-Flash 的 config.json 逐字段 diff 结果">
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">text_config 逐字段 diff：31 个共有字段 → 29 同 / 2 变 / 2 删</text>

  <rect x="20" y="44" width="200" height="86" rx="10" fill="#E1F5EE" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="120" y="68" text-anchor="middle" dominant-baseline="central" font-size="20" font-weight="700" fill="#085041">29</text>
  <text x="120" y="92" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">字段完全相同</text>
  <text x="120" y="112" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0F6E56">含全部 MoE 路由参数</text>

  <rect x="240" y="44" width="200" height="86" rx="10" fill="#FAEEDA" stroke="#854F0B" stroke-width="1.2"/>
  <text x="340" y="68" text-anchor="middle" dominant-baseline="central" font-size="20" font-weight="700" fill="#633806">2</text>
  <text x="340" y="92" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#633806">字段取值改变</text>
  <text x="340" y="112" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#854F0B">rope_theta · 稀疏频率表</text>

  <rect x="460" y="44" width="200" height="86" rx="10" fill="#FBEAF0" stroke="#993556" stroke-width="1.2"/>
  <text x="560" y="68" text-anchor="middle" dominant-baseline="central" font-size="20" font-weight="700" fill="#72243E">2</text>
  <text x="560" y="92" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#72243E">字段被删除</text>
  <text x="560" y="112" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#993556">MTP 两个字段 → DSpark</text>

  <line x1="340" y1="136" x2="340" y2="160" stroke="#888780" stroke-width="1.5"/>

  <rect x="20" y="166" width="640" height="284" rx="12" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="40" y="192" font-size="13" font-weight="700" fill="#444441">四行真正的差异，逐条落位</text>

  <text x="40" y="222" font-size="12" font-weight="700" fill="#0C447C">① rope_theta</text>
  <text x="175" y="222" font-size="12" fill="#444441" font-family="ui-monospace, monospace">5,000,000 → 10,000,000</text>
  <text x="420" y="222" font-size="11" fill="#5F5E5A">RoPE 基频翻倍 → 长上下文位置分辨率更细</text>

  <text x="40" y="252" font-size="12" font-weight="700" fill="#0C447C">② sparse_attention_freq</text>
  <text x="175" y="252" font-size="12" fill="#444441" font-family="ui-monospace, monospace">[0,0,0,1×57] → [1×60]</text>
  <text x="420" y="252" font-size="11" fill="#5F5E5A">前 3 层 Full Attention → MSA，本文主角</text>

  <text x="40" y="282" font-size="12" font-weight="700" fill="#0C447C">③ sparse_disable_index_value</text>
  <text x="175" y="282" font-size="12" fill="#444441" font-family="ui-monospace, monospace">[0,0,0,1×57] → [1×60]</text>
  <text x="420" y="282" font-size="11" fill="#5F5E5A">与 ② 同步，索引器全层启用</text>

  <text x="40" y="312" font-size="12" font-weight="700" fill="#0C447C">④ num_mtp_modules / num_nextn_predict_layers</text>
  <text x="175" y="312" font-size="12" fill="#444441" font-family="ui-monospace, monospace">7 / 1 → 字段不存在</text>
  <text x="420" y="312" font-size="11" fill="#5F5E5A">MTP 头整体移除，换 DSpark draft 目录</text>

  <line x1="40" y1="330" x2="640" y2="330" stroke="#B4B2A9" stroke-width="1" stroke-dasharray="4 3"/>

  <text x="40" y="352" font-size="12" font-weight="700" fill="#0F6E56">外加两项跨文件变化</text>
  <text x="40" y="376" font-size="11" fill="#444441">· 顶层新增 <tspan font-family="ui-monospace, monospace">quantization_config</tspan>（M3 完全没有这个键，即非量化权重）</text>
  <text x="40" y="400" font-size="11" fill="#444441">· <tspan font-family="ui-monospace, monospace">vision_config.image_size</tspan> 从 2016 降到 672，视觉塔其余 18 个字段不变</text>
  <text x="40" y="428" font-size="12" font-weight="700" fill="#72243E">MoE 结构：0 处变化</text>
  <text x="175" y="428" font-size="11" fill="#993556">128 experts / top-4 / 1 shared / scaling 2.0 逐字节相同</text>
</svg>
<p class="cap">图 3：两份 config.json 逐字段 diff 的结果。结构层面的真实变更只有稀疏注意力频率表一处，MoE 拓扑零改动。</p>
</div>

### 变了的：4 行

| 字段 | M3 | M3.1-Flash | 性质 |
|---|---|---|---|
| `rope_theta` | 5,000,000 | **10,000,000** | 基频翻倍 |
| `sparse_attention_config.sparse_attention_freq` | `[0,0,0,1,1,…]`（60 项） | `[1]×60` | **结构变更** |
| `sparse_attention_config.sparse_disable_index_value` | `[0,0,0,1,1,…]`（60 项） | `[1]×60` | 与上一条同步 |
| `num_mtp_modules` / `num_nextn_predict_layers` | `7` / `1` | **字段已删除** | MTP → DSpark |

另有两处跨文件变化：顶层**新增 `quantization_config`**（M3 根本没有这个键，说明 M3 发布的就是非量化权重），以及 `vision_config.image_size` 从 **2016 降到 672**（视觉塔另外 18 个字段完全不变）。

### 没变的：29 行，而且包括全部 MoE 参数

这份「没变清单」比「变了清单」更有信息量：

| 字段 | M3 | M3.1-Flash |
|---|---|---|
| `num_hidden_layers` | 60 | 60 |
| `hidden_size` | 6,144 | 6,144 |
| `num_attention_heads` / `num_key_value_heads` | 64 / 4 | 64 / 4 |
| `head_dim` | 128 | 128 |
| `num_local_experts` | **128** | **128** |
| `num_experts_per_tok` | **4** | **4** |
| `n_shared_experts` | **1** | **1** |
| `routed_scaling_factor` | **2.0** | **2.0** |
| `scoring_func` | **sigmoid** | **sigmoid** |
| `use_routing_bias` | **true** | **true** |
| `moe_layer_freq` | `[0,0,0,1×57]` | 完全相同 |
| `hidden_act` | `swigluoai` | `swigluoai` |
| `swiglu_alpha` / `swiglu_limit` | 1.702 / 7.0 | 1.702 / 7.0 |
| `use_gemma_norm` | true | true |
| `use_qk_norm` / `qk_norm_type` | true / `per_head` | 相同 |
| `rotary_dim` / `partial_rotary_factor` | 64 / 0.5 | 相同 |
| `max_position_embeddings` | 1,048,576 | 1,048,576 |
| `vocab_size` | 200,064 | 200,064 |
| `dense_intermediate_size` | 12,288 | 12,288 |
| `architectures` | `MiniMaxM3SparseForConditionalGeneration` | 相同 |

`sparse_attention_config` 内部除那两个数组外，剩余 7 个标量参数也一字未改：`sparse_block_size=128`、`sparse_topk_blocks=16`、`sparse_index_dim=128`、`sparse_num_index_heads=4`、`sparse_score_type="max"`、`sparse_local_block=1`、`sparse_init_block=0`。

上面的对照只是骨架。完整的 31 个 config 字段原文、逐层张量形状、参数量逐项核算、量化豁免清单与视觉塔全参数，我整理成了一份独立附录：**<a href="/LLM-blog/m31-architecture.html">MiniMax M3.1 Flash 架构详解（12 节 / 6 图 / 18 表）</a>**，可与本文对照阅读。

<div class="warnbox">
<strong>顺便纠一个流传很广的错误。</strong>网上不少解读（包括本文早期草稿）写「M3.1 把路由从 top-4+1 shared 调成了 129 experts / top-5」。这是错的：配置原文里 `num_local_experts=128`、`num_experts_per_tok=4`、`n_shared_experts=1`，两版逐字节相同。官方 model card 列的变更只有三条——稀疏注意力全层化、QAT、DSpark 替换 MTP——没有一条涉及 MoE 结构。真正的 MoE 侧变化在<strong>精度</strong>，不在<strong>拓扑</strong>：routed experts 从 MXFP8 换成了 NVFP4 QAT。
</div>

## 3. 为什么值得为「前三层」较真

前三层只占 60 层的 5%，看起来微不足道。但 Full Attention 的计算量随序列长度**平方**增长——在 1M 上下文下，这三层恰是整个骨干里唯一还在按 O(n²) 运行的部分。M3.1-Flash 把最后一处平方也抹掉，收益与 fa3 里 MSA 同源：长上下文下算力不随长度爆炸。

工程侧还有一对组合拳：

- **NVFP4 KV cache**：indexer 与 attention 路径的 KV 均为 NVFP4，vLLM 在 Blackwell 上用 CUDA 13.4 的 QMUL4 指令在 kernel 内反量化，indexer QK decode 快至 1.7×、稀疏注意力 decode 快至 1.3×；
- **Indexer decode 上下文并行（DCP）**：512K 上下文、4× GB300 下 indexer 每稀疏层快至 3.3×——与 sys7 / sys8 的 CP / DCP 切分思路同源。

## 4. 量化：四档精度与 62 项豁免

`quantization_config` 是 M3.1 新引入的顶层键，也是它体积从 854.2 GB 掉到 252.4 GB 的直接原因。逐字段读：

| 字段 | 值 | 含义 |
|---|---|---|
| `quant_method` | `mxfp8` | 基准方案 MXFP8 |
| `weight_block_size` | `[1, 32]` | 每 32 个权重共用一个 scale |
| `activation_scheme` | `dynamic` | 激活动态量化 |
| `moe_quant_algo` | `NVFP4` | 专家层单独降到 FP4 |
| `moe_quant_format` | `nvfp4-pack-quantized` | **打包存储**（两值一字节） |
| `group_size` | `16` | NVFP4 每 16 元素一组 scale |

四档精度按敏感度分层：

| 组件 | 精度 | 为什么 |
|---|---|---|
| 路由专家 MLP | NVFP4 (W4A4) | 占 97% 参数，量化收益绝对最大 |
| 稠密层 / 注意力线性层 | MXFP8 | 数值敏感度中等 |
| 索引器 `q_proj` / `k_proj` | NVFP4 | 只影响块选择，误差可容忍 |
| MoE gate（57 个） | 高精度 | 路由一旦抖动，整个专家分布漂移 |
| `lm_head` | 高精度 | 直接决定 logits，误差线性传导 |
| `embed_tokens` | 高精度 | 稀疏查询，量化收益小 |
| `vision_tower` | 高精度 | 视觉特征动态范围大 |
| `multi_modal_projector` / `patch_merge_mlp` | 高精度 | 跨模态对齐层，误差会放大 |

`ignored_layers` **共 62 项**，构成可以对得上：57 个 MoE gate（层号恰好是 layers.3 – layers.59，与 `moe_layer_freq` 的 57 个 `1` 严格对齐）+ `lm_head` + `model.embed_tokens` + `multi_modal_projector` + `patch_merge_mlp` + `vision_tower`。

<div class="keybox">
<strong>一个容易被误读的细节：</strong>不量化清单里<strong>没有注意力输出投影</strong>。这说明它对精度的容忍度比 FFN 低、但比 gate 高，走的是 MXFP8 而非 NVFP4。如果你要自己微调或继续量化，这一层是最值得优先保护的。
</div>

体积账也能对上。KV cache 每 token 的元素数是 4 个 KV 头 × 128 维 × 60 层 × 2（K 和 V）= **61,440 个元素**；M3 下每元素 1 字节（FP8），M3.1 下 NVFP4 约 0.5625 字节（含每 16 元素一个 FP8 scale 的开销），得 **34,560 字节**。注意这不是正好一半——scale 的额外开销让它略高于 0.5。

<div class="warnbox">
<strong>为什么 Hugging Face 上参数量显示 220B 而不是 428B？</strong>因为 HF 按存储的 tensor 元素数统计，而 NVFP4 是打包存储——一个字节装两个值。专家权重打包后自动计数只认字节数，于是 428B 被报成 220B 左右。看 M3 仓库报 427B（全 MXFP8 不打包）就能对上。这是统计口径问题，不是模型缩水。
</div>

## 5. DSpark：三段式草稿头

M3.1 删掉 `num_mtp_modules=7` / `num_nextn_predict_layers=1`，换成独立目录 `dspark/` 下的 draft head。它的 `config.json` 是一个完整的模型配置，结构可拆成三段：

| 段 | 配置 | 作用 |
|---|---|---|
| 草稿骨干 | `num_hidden_layers=5`、`hidden_size=6144`、`dspark_ffn_hidden_size=12288`、`sliding_window=4096` | 5 层 SWA，只做局部上下文起草 |
| Markov 头 | `dspark_markov_rank=256`、`markov_head_type="vanilla"` | 低秩转移矩阵，约束相邻 token 的连贯性 |
| 置信度头 | `enable_confidence_head=true`、`confidence_head_with_markov=true` | 预测每步接受概率，供调度器决定截断位置 |

几个必须记准的字段：

- **`dspark_block_size = 7`** —— 一次起草 7 个 token。网上有些资料写 8，那是 SPEED-Bench 的评测设置，别和模型配置混用；
- **`dspark_target_layer_ids = [3, 17, 31, 45, 59]`** —— 从目标模型这 5 层抽取 hidden states 作为草稿条件的来源。注意取了 Layer 59，也就是最后一层；
- **`dspark_noise_token_id = 200058`** —— 掩码占位 token，落在词表 200,064 之内的尾部；
- **`dspark_use_dense_ffn = true`** —— 草稿层用稠密 FFN（`moe_layer_freq=[0,0,0,0,0]`），不做 MoE，控制草稿开销；
- **`dspark_hybrid_context_fusion = false`** —— 这版没启用混合上下文融合；
- **`dspark_aux_gemm_precision = "bf16"`** —— 辅助 GEMM 保持 bf16，不参与量化。

vLLM 配方里 `num_speculative_tokens` Blackwell 默认 **7**、AMD MI355X 默认 **3**——前者与 `dspark_block_size` 对齐，后者是显存/带宽权衡下的保守取值。

## 6. 视觉塔与多模态

两个模型的视觉塔几乎完全一致（18/19 个字段相同），唯一的差异是 `image_size` 从 2016 降到 672：

| 字段 | 值 | 说明 |
|---|---|---|
| `model_type` | `clip_vision_model` | CLIP 风格，非 SigLIP |
| `num_hidden_layers` | 32 | |
| `hidden_size` | 1,280 | |
| `num_attention_heads` | 16 | 标准 MHA，非 GQA |
| `intermediate_size` | 5,120 | |
| `patch_size` | 14 | |
| `position_embedding_type` / `rope_mode` | `rope` / `3d` | 3D RoPE，`rope_theta=10000` |
| `hidden_act` | `gelu` | 不是 SwiGLU |
| `layer_norm_eps` | 1e-05 | **LayerNorm**，非 RMSNorm |
| `img_token_compression_config` | `patch_merge`, `spatial_merge_size=2`, `temporal_patch_size=2` | 2×2 空间合并 + 2 帧时间合并 |
| `vision_segment_max_frames` | 4 | 单段最多 4 帧 |
| `projector_hidden_size` | 6,144 | 对齐文本 hidden |
| `image_seq_length` | 576 | 单图默认 token 数 |
| `process_image_mode` | `dynamic_res` | 动态分辨率 |
| `image_grid_pinpoints` | 36 组，336–2016 | 长宽比自适应切图 |

<div class="keybox">
<strong>视觉塔是整份配置里「最不像 MiniMax」的部分：</strong>LayerNorm + GELU + 标准 MHA + 3D RoPE——这是教科书式的 CLIP ViT，和文本侧那套 Gemma Norm + swigluoai + GQA 16:1 + QK-Norm 完全是两个世界的设计。多模态模型的视觉塔通常直接沿用开源 CLIP 权重，这份配置印证了这一点。
</div>

## 7. 部署要点：三个硬约束

```bash
vllm serve MiniMaxAI/MiniMax-M3.1-Flash \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --kv-cache-dtype nvfp4 \
  --trust-remote-code \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}'
```

三个容易踩的坑：

1. **`--block-size 128` 是硬约束，不能改**。MSA 的 Indexer 按 128-token 块打分，KV cache 的块粒度必须与之对齐，否则索引到物理块的映射直接错位；
2. **Hopper 上要回退**。NVFP4 的 QMUL4 反量化指令是 Blackwell（CUDA 13.4）才有，H100/H800 上得退回 Marlin MoE kernel，并把上下文长度限制到 262,144，同时关闭视频输入与 DSpark；
3. **视频与投机解码不能同时开**。视觉塔的 `vision_segment_max_frames=4` 与 DSpark 的 `sliding_window=4096` 在显存预算上互斥。

**Day-0 生态**：10-23 权重上线当天，vLLM / SGLang / Hugging Face / ModelScope 全部打通；昇腾、平头哥、昆仑芯、摩尔线程、沐曦等国产芯片与 Together AI、Fireworks、Nebius 等国际云同步适配，多芯片部署走 FlagOS。对一个 428B 的多模态 MoE 来说，这套生态铺开速度本身是它能被实际用起来的关键——毕竟愿意自己拉 252GB 权重的人不多。

## 8. 参数量核算：426B 是怎么来的

官方说「约 428B 总参、每 token 激活 23B」。自己算一遍能验证口径。

| 组件 | 计算式 | 参数量 |
|---|---|---|
| 路由专家（57 层） | 57 × 128 × 3 × 6144 × 3072 | 413.1 B |
| 共享专家（57 层） | 57 × 3 × 6144 × 3072 | 3.23 B |
| MoE gate（57 个） | 57 × 6144 × 128 | 0.045 B |
| 稠密 FFN（前 3 层） | 3 × 3 × 6144 × 12288 | 0.68 B |
| 注意力（60 层） | 60 × (6144×8192 + 2×6144×512 + 8192×6144) | 6.42 B |
| Embedding + LM Head | 2 × 200064 × 6144 | 2.46 B |
| **小计** | | **425.9 B** |
| 视觉塔 + 投影（不计入文本侧） | 32 层 ViT + 30720×6144 投影 | ≈ 0.8 B |
| **合计** | | **≈ 426.7 B** |

与官方 428B 差约 1.3B，缺口来自 RMSNorm 权重、视觉位置编码与各类 bias。

**激活 23B 的口径要看清**：`num_experts_per_tok=4` + `n_shared_experts=1`，即每 token 走 5 个专家 = 5 × 3 × 6144 × 3072 × 57 层的专家部分，加上 60 层注意力、3 层稠密 FFN 与两次投影。这个数**不含 embedding 与 lm_head**（各 1.23B，但它们只是查表，不算 per-token 计算量）；严格按 per-token 计算量算约 **25.5B**。

## 9. 基准速览（官方口径）

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

## 10. 数据存疑处

写这份拆解时，有几处口径对不齐。列出来，比含糊过去更有用：

| 疑点 | 现象 | 我的判断 |
|---|---|---|
| 参数量 220B vs 428B | HF 页面显示约 220B，官方说 428B | **统计口径差异**。NVFP4 打包存储导致 HF 按字节计数低估，见第 4 节 |
| `image_size` 2016 → 672 | 视觉塔预训练分辨率砍到 1/3 | **可能是默认值调整**，`image_grid_pinpoints` 仍支持到 2016，说明能力没丢，只是默认档位换了 |
| `sparse_init_block=0` | 没有强制保留首块 | 意味着**没有 attention sink**。这对超长上下文的开头 token 是否吃亏，官方未说明 |
| `dspark_block_size` 7 vs 8 | 配置是 7，部分资料写 8 | 7 是模型定义，8 是 SPEED-Bench 评测设置，**不要混用** |
| 激活 23B vs 25.5B | 官方 23B，逐项算出 25.5B | 官方口径**不含 embedding 与 lm_head**，两者都是 1.23B 的查表 |
| 基准分数 | 全部来自厂商记分卡 | **独立榜单待补**，尤其 Terminal Bench 与 HLE 落后明显 |

## 11. 与系列文章的衔接

- **fa3（MiniMax M2.7 → M3）**：MSA 的首次跨越是本文的前半页；本文补上 M3 → M3.1-Flash 的「最后一平方」。
- **sys3（MSA / CSA / HCA）**：MSA 是三种注意力改造路线里「不压缩 KV、只选重要位置」的一支，本文给出它在 M3 家族里的最新形态与配置参数。
- **ep4 / ep5 / ep8（DSpark）**：M3.1-Flash 把 DSpark 做成随权重发布的 draft head，spec decoding 从论文/框架特性变成模型自带能力。
- **fw2 / fw3（vLLM / SGLang）**：M3.1-Flash 的 Day-0 serving 配方（block-size 128、NVFP4 KV、indexer DCP）可直接对照框架手记阅读。

**附录**：本文正文只走「结构差异」一条主线，凡是需要摊开全量参数的地方都放进了 <a href="/LLM-blog/m31-architecture.html">MiniMax M3.1 Flash 架构详解</a>——那份文档有 12 节、6 张图、18 张表，覆盖全部 config 字段原文、单层数据流与张量形状、MoE 路由细节、参数量逐项核算、DSpark 三段式、四档量化与 62 项豁免、视觉塔 19 个字段、M3 → M3.1 的 12 项逐项 diff、部署坑与数据存疑处。

> 下一篇进入算子讲解手记：把 MLA 多头潜在注意力写成可照着实现的逐行伪代码（op1）。
