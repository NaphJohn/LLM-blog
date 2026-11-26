---
title: Transformer 精读：用「注意力」取代循环，把序列建模变成可并行计算
description: 拆解 Attention Is All You Need（Vaswani et al., 2017, Google）——自注意力如何替代 RNN/CNN、多头注意力在算什么、位置编码为何必不可少，以及它为何成为此后所有大模型的底座。
pubDate: 2026-08-12
series: 论文科普手记
lang: zh
altLang: en
altHref: /en/blog/pp1-transformer
layout: ../../layouts/BlogPost.astro
---

## 1. 一句话结论

**Transformer 的核心主张是：序列建模不需要「一个字一个字地读」（RNN），也不需要「用滑动窗口看」（CNN），而是让序列里的每个位置直接「抬头看一眼」所有其他位置，用一种叫「自注意力」的机制一次性算出谁和谁相关。** 这让它天然可并行、可堆深、可扩展到超长上下文——今天你用的每一个大模型，骨子里都是它。

## 2. 为什么要换掉 RNN？

在 2017 年之前，处理语言、语音、时间序列的主力是 **RNN / LSTM**：

- 它像读书一样**从左到右**逐个 token 处理，第 `t` 步必须等第 `t-1` 步算完。
- 这带来两个致命问题：
  1. **无法并行**：一句话 1000 个字，就得串行跑 1000 步，GPU 的千核算力用不上。
  2. **长程遗忘**：信息要从第 1 个字一路传递到第 1000 个字，越往后越稀释（梯度消失/爆炸）。虽然 LSTM 用门控缓解，但本质没变。

CNN（如 PixelCNN / 卷积文本模型）能并行，但**感受野受卷积核大小限制**，要看远距离依赖得堆很多层。

Transformer 的赌注：**直接建模任意两个位置之间的依赖，距离多远都只隔一步计算**。

## 3. 自注意力：每个词「抬头看全场」

自注意力（Self-Attention）的输入是一组向量 `X = [x₁, x₂, …, xₙ]`（每个 token 的嵌入）。它对每个位置算出三个向量：

- **Q（Query，我在找什么）**
- **K（Key，我能提供什么）**
- **V（Value，我实际携带的信息）**

通过一个线性层从 `X` 投影得到：`Q = XW_Q`，`K = XW_K`，`V = XW_V`。

然后每个位置对其它所有位置算「相关度」：

```
score(i, j) = Q_i · K_j / √d_k
attention(i) = softmax_j( score(i, j) ) · V_j
```

直觉：`Q_i` 和 `K_j` 越像（点积越大），第 `i` 个词就越「注意」第 `j` 个词，最终输出就是把所有位置的 `V` 按注意力权重**加权求和**。

> **√d_k 的作用**：当维度 `d_k` 很大时点积容易爆掉，除以它的平方根做缩放，让 softmax 梯度更稳定。这是论文里一个不起眼但很关键的小技巧。

关键点：**任意一个词看另一个词，都只经过一次矩阵乘法**，和第 `j` 个字距离多远无关。这就是「一步到位的全局依赖」。

## 4. 多头注意力：不只看一个角度

单一注意力只能学到一种「关系视角」。Transformer 用了 **Multi-Head Attention**：把 `Q/K/V` 切成 `h` 个头，每个头在各自子空间里独立算注意力，最后把结果拼起来再投影：

```
head_h = Attention(XW_Q^h, XW_K^h, XW_V^h)
MultiHead = Concat(head₁, …, head_h) · W_O
```

不同头会自然分工：有的盯语法依存、有的盯指代、有的盯语义相似。论文用 `h = 8` 个头、每个头维度 `64`（总维度 `512`）。

## 5. 位置编码：注意力本身「不认顺序」

自注意力对输入顺序是**排列不变**的——把句子打乱，每个词算出的加权求和结果只和「出现了哪些词」有关，和「谁在前」无关。但语言明显有顺序。

解决办法是给每个位置叠加一个**位置编码** `PE(pos)`：

- 论文用**正弦/余弦固定编码**：`PE(pos, 2i) = sin(pos / 10000^(2i/d))`，`PE(pos, 2i+1) = cos(...)`。
- 好处：它能让「位置 `pos+k` 的编码」线性表达为「位置 `pos` 编码」的函数，模型容易学会「相对位置」。

后来的模型（如 GPT）大多改用**可学习的位置嵌入**，思路一致：必须把顺序信息喂进去。

## 6. 编码器–解码器骨架

原始 Transformer 用于翻译，分两半：

- **Encoder（编码器）**：一堆自注意力 + 前馈层，把源句压成富含上下文的表示。它的自注意力是**双向**的（看前后文）。
- **Decoder（解码器）**：它自己的自注意力被换成了**掩码自注意力**（Masked Self-Attention，只能看已生成的部分，不能偷看未来），再加一层**交叉注意力**（Cross-Attention，Q 来自解码器、K/V 来自编码器），逐词生成译文。注意这两者都只是「注意力子层」，解码器每层总共两个注意力，不是三个。

<div class="fig">
<svg id="fig-qkv" style="width:100%;max-width:720px;height:auto;display:block;margin:0 auto" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 492" width="100%" role="img" aria-labelledby="fig-qkv-fig1Title fig-qkv-fig1Desc">
<title id="fig-qkv-fig1Title">三种注意力的 Q/K/V 来源对比</title>
<desc id="fig-qkv-fig1Desc">编码器自注意力、解码器掩码自注意力、交叉注意力三者的 Q、K、V 分别从哪个输入投影而来。前两者三路同源，第三者的 K/V 来自编码器。</desc>
<defs>
<marker id="fig-qkv-arG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="var(--gr-s)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
<marker id="fig-qkv-arB" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="var(--bl-s)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
<marker id="fig-qkv-arP" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="var(--pu-s)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
<marker id="fig-qkv-arT" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="var(--te-s)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<style>#fig-qkv{
--bg:#fcfcfd;
--t1:#2C2C2A; --t2:#5F5E5A;
--gr-p:#EAF3DE; --gr-c:#C0DD97; --gr-w:#FFFFFF; --gr-s:#3B6D11; --gr-t:#173404; --gr-x:#3B6D11;
--bl-p:#E6F1FB; --bl-c:#B5D4F4; --bl-w:#FFFFFF; --bl-s:#185FA5; --bl-t:#0C447C; --bl-x:#185FA5;
--pu-p:#EEEDFE; --pu-c:#CECBF6; --pu-w:#FFFFFF; --pu-s:#534AB7; --pu-t:#26215C; --pu-x:#534AB7;
--te-p:#E1F5EE; --te-c:#9FE1CB; --te-w:#FFFFFF; --te-s:#1D9E75; --te-t:#04342C; --te-x:#0F6E56;
--nu-p:#F1EFE8; --nu-s:#B4B2A9; --nu-t:#444441;
--rd-x:#A32D2D;
}:root[data-theme="dark"] #fig-qkv{
--bg:#161b22;
--t1:#F1EFE8; --t2:#D3D1C7;
--gr-p:#173404; --gr-c:#27500A; --gr-w:#0A2102; --gr-s:#97C459; --gr-t:#C0DD97; --gr-x:#97C459;
--bl-p:#042C53; --bl-c:#0C447C; --bl-w:#021A33; --bl-s:#85B7EB; --bl-t:#B5D4F4; --bl-x:#85B7EB;
--pu-p:#26215C; --pu-c:#3C3489; --pu-w:#1A1640; --pu-s:#AFA9EC; --pu-t:#CECBF6; --pu-x:#AFA9EC;
--te-p:#04342C; --te-c:#0F6E56; --te-w:#02211C; --te-s:#5DCAA5; --te-t:#9FE1CB; --te-x:#5DCAA5;
--nu-p:#2C2C2A; --nu-s:#888780; --nu-t:#D3D1C7;
--rd-x:#F09595;
}#fig-qkv text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}#fig-qkv .bg{fill:var(--bg)}#fig-qkv .h1{font-size:15px;font-weight:500;fill:var(--t1)}#fig-qkv .hd{font-size:12.5px;font-weight:500}#fig-qkv .tx{font-size:12px;fill:var(--t2)}#fig-qkv .sm{font-size:11px}#fig-qkv .gr-p{fill:var(--gr-p);stroke:var(--gr-s);stroke-width:.5}#fig-qkv .bl-p{fill:var(--bl-p);stroke:var(--bl-s);stroke-width:.5}#fig-qkv .pu-p{fill:var(--pu-p);stroke:var(--pu-s);stroke-width:.5}#fig-qkv .nu-p{fill:var(--nu-p);stroke:var(--nu-s);stroke-width:.5}#fig-qkv .gr-c{fill:var(--gr-c);stroke:var(--gr-s);stroke-width:.5}#fig-qkv .bl-c{fill:var(--bl-c);stroke:var(--bl-s);stroke-width:.5}#fig-qkv .pu-c{fill:var(--pu-c);stroke:var(--pu-s);stroke-width:.5}#fig-qkv .te-c{fill:var(--te-c);stroke:var(--te-s);stroke-width:.5}#fig-qkv .w{fill:var(--gr-w);stroke:var(--gr-s);stroke-width:.5}#fig-qkv .w-b{fill:var(--bl-w);stroke:var(--bl-s);stroke-width:.5}#fig-qkv .w-bd{fill:var(--bl-w);stroke:var(--bl-s);stroke-width:1;stroke-dasharray:3 2}#fig-qkv .w-p{fill:var(--pu-w);stroke:var(--pu-s);stroke-width:.5}#fig-qkv .w-t{fill:var(--te-w);stroke:var(--te-s);stroke-width:.5}#fig-qkv .t-gr{fill:var(--gr-t)}#fig-qkv .t-bl{fill:var(--bl-t)}#fig-qkv .t-pu{fill:var(--pu-t)}#fig-qkv .t-te{fill:var(--te-t)}#fig-qkv .x-gr{fill:var(--gr-x)}#fig-qkv .x-bl{fill:var(--bl-x)}#fig-qkv .x-pu{fill:var(--pu-x)}#fig-qkv .x-te{fill:var(--te-x)}#fig-qkv .x-rd{fill:var(--rd-x)}#fig-qkv .x-nu{fill:var(--nu-t)}#fig-qkv .ln-gr{stroke:var(--gr-s);stroke-width:1;fill:none}#fig-qkv .ln-bl{stroke:var(--bl-s);stroke-width:1;fill:none}#fig-qkv .ln-pu{stroke:var(--pu-s);stroke-width:1.5;fill:none}#fig-qkv .ln-te{stroke:var(--te-s);stroke-width:1.5;fill:none}#fig-qkv .mid{text-anchor:middle}</style>
<rect class="bg" x="0" y="0" width="680" height="492" rx="10"/>
<text class="h1" x="40" y="26">判据：Q、K、V 分别从哪儿投影出来</text>
<rect class="gr-p" x="40" y="40" width="600" height="116" rx="10"/>
<text class="hd t-gr" x="56" y="60">① 编码器自注意力 · 双向，看全场</text>
<rect class="gr-c" x="56" y="74" width="150" height="50" rx="6"/>
<text class="sm t-gr mid" x="131" y="91" dominant-baseline="central">编码器输入 X</text>
<text class="sm x-gr mid" x="131" y="108" dominant-baseline="central">源句全部 token</text>
<rect class="w" x="300" y="72" width="86" height="22" rx="4"/>
<text class="sm t-gr mid" x="343" y="83" dominant-baseline="central">Q = XW_Q</text>
<rect class="w" x="300" y="100" width="86" height="22" rx="4"/>
<text class="sm t-gr mid" x="343" y="111" dominant-baseline="central">K = XW_K</text>
<rect class="w" x="300" y="128" width="86" height="22" rx="4"/>
<text class="sm t-gr mid" x="343" y="139" dominant-baseline="central">V = XW_V</text>
<path class="ln-gr" d="M206 92L296 84" marker-end="url(#fig-qkv-arG)"/>
<path class="ln-gr" d="M206 99L296 112" marker-end="url(#fig-qkv-arG)"/>
<path class="ln-gr" d="M206 110L296 139" marker-end="url(#fig-qkv-arG)"/>
<text class="sm x-gr" x="404" y="86">三路同源，都来自同一个 X</text>
<text class="sm x-gr" x="404" y="108">每个词都能看前后所有词</text>
<text class="sm x-gr" x="404" y="130">没有掩码，矩阵全开</text>
<rect class="bl-p" x="40" y="164" width="600" height="116" rx="10"/>
<text class="hd t-bl" x="56" y="184">② 解码器 Masked Self-Attention · 因果，不许偷看</text>
<rect class="bl-c" x="56" y="198" width="150" height="50" rx="6"/>
<text class="sm t-bl mid" x="131" y="215" dominant-baseline="central">解码器输入 X</text>
<text class="sm x-bl mid" x="131" y="232" dominant-baseline="central">已生成的那半句</text>
<rect class="w-b" x="300" y="198" width="86" height="22" rx="4"/>
<text class="sm t-bl mid" x="343" y="209" dominant-baseline="central">Q = XW_Q</text>
<rect class="w-bd" x="300" y="226" width="86" height="22" rx="4"/>
<text class="sm x-bl mid" x="343" y="237" dominant-baseline="central">K · 被掩码截断</text>
<rect class="w-bd" x="300" y="254" width="86" height="22" rx="4"/>
<text class="sm x-bl mid" x="343" y="265" dominant-baseline="central">V · 被掩码截断</text>
<path class="ln-bl" d="M206 216L296 209" marker-end="url(#fig-qkv-arB)"/>
<path class="ln-bl" d="M206 223L296 237" marker-end="url(#fig-qkv-arB)"/>
<path class="ln-bl" d="M206 234L296 265" marker-end="url(#fig-qkv-arB)"/>
<text class="sm x-bl" x="404" y="212">还是三路同源，结构没变</text>
<text class="sm x-rd" x="404" y="234">差别只在：分数矩阵上三角置 -∞</text>
<text class="sm x-rd" x="404" y="256">softmax 后未来位置权重归零</text>
<rect class="pu-p" x="40" y="288" width="600" height="130" rx="10"/>
<text class="hd t-pu" x="56" y="308">③ Cross-Attention · 唯一来源不同的一个</text>
<rect class="pu-c" x="56" y="320" width="150" height="40" rx="6"/>
<text class="sm t-pu mid" x="131" y="340" dominant-baseline="central">解码器当前状态</text>
<rect class="te-c" x="56" y="368" width="150" height="40" rx="6"/>
<text class="sm t-te mid" x="131" y="388" dominant-baseline="central">编码器输出</text>
<rect class="w-p" x="300" y="320" width="86" height="22" rx="4"/>
<text class="sm t-pu mid" x="343" y="331" dominant-baseline="central">Q</text>
<rect class="w-t" x="300" y="366" width="86" height="22" rx="4"/>
<text class="sm t-te mid" x="343" y="377" dominant-baseline="central">K</text>
<rect class="w-t" x="300" y="394" width="86" height="22" rx="4"/>
<text class="sm t-te mid" x="343" y="405" dominant-baseline="central">V</text>
<path class="ln-pu" d="M206 338L296 331" marker-end="url(#fig-qkv-arP)"/>
<path class="ln-te" d="M206 382L296 377" marker-end="url(#fig-qkv-arT)"/>
<path class="ln-te" d="M206 390L296 405" marker-end="url(#fig-qkv-arT)"/>
<text class="sm x-pu" x="404" y="334">「我要找什么」由解码器决定</text>
<text class="sm x-te" x="404" y="382">「资料在哪」来自源句</text>
<text class="sm x-te" x="404" y="404">相当于拿目标去检索原文</text>
<rect class="nu-p" x="40" y="432" width="600" height="34" rx="8"/>
<text class="sm x-nu" x="56" y="449" dominant-baseline="central">前两个是「自己跟自己比」，第三个是「我跟别人比」—— 这才是三者真正的分野</text>
</svg>
<p class="fig-cap">图 1：三种注意力的 Q/K/V 来源。编码器自注意力与解码器掩码自注意力都是三路同源（同源于输入 X），区别只在掩码；交叉注意力是唯一 Q 与 K/V 不同源的一种——Q 由解码器给出，K/V 来自编码器。</p>
</div>
<div class="fig">
<svg id="fig-mtx" style="width:100%;max-width:720px;height:auto;display:block;margin:0 auto" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 538" width="100%" role="img" aria-labelledby="fig-mtx-fig2Title fig-mtx-fig2Desc">
<title id="fig-mtx-fig2Title">两种注意力的分数矩阵形状对比</title>
<desc id="fig-mtx-fig2Desc">掩码自注意力是 n×n 方阵且上三角被屏蔽，交叉注意力是 n×m 矩形且不屏蔽，行数为目标长度、列数为源长度。</desc>
<style>#fig-mtx{
--bg:#fcfcfd;
--t1:#2C2C2A; --t2:#5F5E5A; --t3:#66655F;
--bl-x:#185FA5; --te-x:#0F6E56; --rd-x:#A32D2D; --nu-t:#444441;
--al:#9FE1CB; --ms:#F7C1C1; --sl:#B5D4F4;
--bl-p:#E6F1FB; --bl-s:#185FA5;
--te-p:#E1F5EE; --te-s:#1D9E75;
--nu-p:#F1EFE8; --nu-s:#B4B2A9;
--cellln:#FFFFFF;
}:root[data-theme="dark"] #fig-mtx{
--bg:#161b22;
--t1:#F1EFE8; --t2:#D3D1C7; --t3:#B4B2A9;
--bl-x:#85B7EB; --te-x:#5DCAA5; --rd-x:#F09595; --nu-t:#D3D1C7;
--al:#0F6E56; --ms:#501313; --sl:#0C447C;
--bl-p:#042C53; --bl-s:#85B7EB;
--te-p:#04342C; --te-s:#5DCAA5;
--nu-p:#2C2C2A; --nu-s:#888780;
--cellln:#191918;
}#fig-mtx text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}#fig-mtx .bg{fill:var(--bg)}#fig-mtx .h1{font-size:15px;font-weight:500;fill:var(--t1)}#fig-mtx .hd{font-size:12.5px;font-weight:500}#fig-mtx .sm{font-size:11px}#fig-mtx .xs{font-size:10.5px}#fig-mtx .lb{font-size:10.5px;fill:var(--t3)}#fig-mtx .tt{font-size:11px;fill:var(--t2)}#fig-mtx .bl-p{fill:var(--bl-p);stroke:var(--bl-s);stroke-width:.5}#fig-mtx .te-p{fill:var(--te-p);stroke:var(--te-s);stroke-width:.5}#fig-mtx .nu-p{fill:var(--nu-p);stroke:var(--nu-s);stroke-width:.5}#fig-mtx .al{fill:var(--al);stroke:var(--cellln);stroke-width:1}#fig-mtx .ms{fill:var(--ms);stroke:var(--cellln);stroke-width:1}#fig-mtx .sl{fill:var(--sl);stroke:var(--cellln);stroke-width:1}#fig-mtx .sw{stroke:var(--t3);stroke-width:.5}#fig-mtx .t-bl{fill:var(--bl-x)}#fig-mtx .t-te{fill:var(--te-x)}#fig-mtx .t-rd{fill:var(--rd-x)}#fig-mtx .t-nu{fill:var(--nu-t)}#fig-mtx .mid{text-anchor:middle}#fig-mtx .end{text-anchor:end}</style>
<rect class="bg" x="0" y="0" width="680" height="538" rx="10"/>
<text class="h1" x="40" y="26">分数矩阵的形状，决定了两者本质不同</text>
<rect class="bl-p" x="40" y="40" width="600" height="214" rx="10"/>
<text class="hd t-bl" x="56" y="62">A · Masked Self-Attention：方阵，上三角屏蔽</text>
<text class="xs lb mid" x="170" y="82">K 的位置 →　／　Q 的位置 ↓</text>
<text class="xs lb mid" x="112" y="96">我</text>
<text class="xs lb mid" x="148" y="96">爱</text>
<text class="xs lb mid" x="184" y="96">你</text>
<text class="xs lb mid" x="220" y="96">。</text>
<rect class="al" x="94" y="104" width="36" height="36"/><text class="sm t-te mid" x="112" y="122" dominant-baseline="central">1.0</text>
<rect class="ms" x="130" y="104" width="36" height="36"/>
<rect class="ms" x="166" y="104" width="36" height="36"/>
<rect class="ms" x="202" y="104" width="36" height="36"/>
<text class="xs lb end" x="84" y="122" dominant-baseline="central">我</text>
<rect class="al" x="94" y="140" width="36" height="36"/>
<rect class="al" x="130" y="140" width="36" height="36"/>
<rect class="ms" x="166" y="140" width="36" height="36"/>
<rect class="ms" x="202" y="140" width="36" height="36"/>
<text class="xs lb end" x="84" y="158" dominant-baseline="central">爱</text>
<rect class="al" x="94" y="176" width="36" height="36"/>
<rect class="al" x="130" y="176" width="36" height="36"/>
<rect class="al" x="166" y="176" width="36" height="36"/>
<rect class="ms" x="202" y="176" width="36" height="36"/>
<text class="xs lb end" x="84" y="194" dominant-baseline="central">你</text>
<rect class="al" x="94" y="212" width="36" height="36"/>
<rect class="al" x="130" y="212" width="36" height="36"/>
<rect class="al" x="166" y="212" width="36" height="36"/>
<rect class="al" x="202" y="212" width="36" height="36"/>
<text class="xs lb end" x="84" y="230" dominant-baseline="central">。</text>
<text class="xs lb" x="268" y="248">对角线含自己，所以第一个词也能看自己</text>
<rect class="al sw" x="268" y="108" width="14" height="14" rx="2"/>
<text class="sm tt" x="290" y="115" dominant-baseline="central">可以看（含自己）</text>
<rect class="ms sw" x="268" y="132" width="14" height="14" rx="2"/>
<text class="sm tt" x="290" y="139" dominant-baseline="central">置 -∞，softmax 后为 0</text>
<text class="sm t-bl" x="268" y="166">n × n 方阵，行列都是目标序列</text>
<text class="sm t-bl" x="268" y="188">训练时一次并行算完，靠掩码造因果性</text>
<text class="sm t-bl" x="268" y="210">推理时只算最后一行，前面缓存下来</text>
<text class="sm t-rd" x="268" y="228">「只算最后一行」正是 KV Cache 的由来</text>
<rect class="te-p" x="40" y="266" width="600" height="176" rx="10"/>
<text class="hd t-te" x="56" y="288">B · Cross-Attention：矩形，n × m，无掩码</text>
<text class="xs lb mid" x="170" y="306">K / V 来自编码器（源句）→</text>
<text class="xs lb mid" x="112" y="322">I</text>
<text class="xs lb mid" x="148" y="322">love</text>
<text class="xs lb mid" x="184" y="322">you</text>
<text class="xs lb mid" x="220" y="322">.</text>
<rect class="sl" x="94" y="326" width="36" height="32"/><text class="xs t-bl mid" x="112" y="342" dominant-baseline="central">.7</text>
<rect class="sl" x="130" y="326" width="36" height="32"/><text class="xs t-bl mid" x="148" y="342" dominant-baseline="central">.2</text>
<rect class="sl" x="166" y="326" width="36" height="32"/><text class="xs t-bl mid" x="184" y="342" dominant-baseline="central">.1</text>
<rect class="sl" x="202" y="326" width="36" height="32"/><text class="xs t-bl mid" x="220" y="342" dominant-baseline="central">0</text>
<text class="xs lb end" x="84" y="342" dominant-baseline="central">我</text>
<rect class="sl" x="94" y="360" width="36" height="32"/><text class="xs t-bl mid" x="112" y="376" dominant-baseline="central">.1</text>
<rect class="sl" x="130" y="360" width="36" height="32"/><text class="xs t-bl mid" x="148" y="376" dominant-baseline="central">.6</text>
<rect class="sl" x="166" y="360" width="36" height="32"/><text class="xs t-bl mid" x="184" y="376" dominant-baseline="central">.3</text>
<rect class="sl" x="202" y="360" width="36" height="32"/><text class="xs t-bl mid" x="220" y="376" dominant-baseline="central">0</text>
<text class="xs lb end" x="84" y="376" dominant-baseline="central">爱</text>
<rect class="sl" x="94" y="394" width="36" height="32"/><text class="xs t-bl mid" x="112" y="410" dominant-baseline="central">.2</text>
<rect class="sl" x="130" y="394" width="36" height="32"/><text class="xs t-bl mid" x="148" y="410" dominant-baseline="central">.1</text>
<rect class="sl" x="166" y="394" width="36" height="32"/><text class="xs t-bl mid" x="184" y="410" dominant-baseline="central">.5</text>
<rect class="sl" x="202" y="394" width="36" height="32"/><text class="xs t-bl mid" x="220" y="410" dominant-baseline="central">.2</text>
<text class="xs lb end" x="84" y="410" dominant-baseline="central">你</text>
<text class="sm t-te" x="268" y="344">n × m 矩形，n 是目标长度、m 是源长度</text>
<text class="sm t-te" x="268" y="368">不屏蔽：每个目标词都能看整个源句</text>
<text class="sm t-te" x="268" y="392">K/V 只算一次，所有 decode 步反复复用</text>
<text class="sm t-te" x="268" y="416">所以编码器的 K/V 天然就是「已缓存」的</text>
<rect class="nu-p" x="40" y="454" width="600" height="64" rx="8"/>
<text class="hd" x="56" y="474" style="fill:var(--t1)">一句话记住</text>
<text class="sm t-nu" x="56" y="492">前者管「不许看未来」：上三角 -∞ 强行造出时序，是自回归的纪律。</text>
<text class="sm t-nu" x="56" y="509">后者管「从哪儿取信息」：Q 与 K/V 不同源，是解码器向源句提问的唯一通道。</text>
</svg>
<p class="fig-cap">图 2：两者的分数矩阵形状不同。掩码自注意力是 n×n 方阵，上三角置 -∞ 造出因果性；交叉注意力是 n×m 矩形，不屏蔽，每个目标词都能看整个源句。这条「只算最后一行」正是 KV Cache 的由来。</p>
</div>


</div>

- 每个子层外用 **残差连接 + LayerNorm**：`output = LayerNorm(x + Sublayer(x))`，这是能让深层网络稳定训练的关键。

<div class="fig">
<svg id="fig-blk" style="width:100%;max-width:720px;height:auto;display:block;margin:0 auto" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 572" width="100%" role="img" aria-labelledby="fig-blk-fig3Title fig-blk-fig3Desc">
<title id="fig-blk-fig3Title">解码器单个 block 的结构</title>
<desc id="fig-blk-fig3Desc">解码器每个 block 只有两个子层：掩码自注意力和交叉注意力，各自被残差连接与 LayerNorm 包住。掩码自注意力的 Q/K/V 来自本层输入，交叉注意力的 K/V 来自编码器输出。</desc>
<defs>
<marker id="fig-blk-ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>
<style>#fig-blk{
--bg:#fcfcfd;
--t1:#2C2C2A; --t2:#5F5E5A;
--pn:#F8F7F2;
--nu-p:#F1EFE8; --nu-s:#B4B2A9; --nu-t:#444441;
--bl-c:#B5D4F4; --bl-s:#185FA5; --bl-t:#0C447C;
--pu-c:#CECBF6; --pu-s:#534AB7; --pu-t:#26215C;
--te-c:#9FE1CB; --te-s:#1D9E75; --te-t:#04342C;
--am-s:#8A5600;
--fl-s:#6B6A65;
--op:#FFFFFF;
}:root[data-theme="dark"] #fig-blk{
--bg:#161b22;
--t1:#F1EFE8; --t2:#D3D1C7;
--pn:#201F1E;
--nu-p:#2C2C2A; --nu-s:#888780; --nu-t:#D3D1C7;
--bl-c:#0C447C; --bl-s:#85B7EB; --bl-t:#B5D4F4;
--pu-c:#3C3489; --pu-s:#AFA9EC; --pu-t:#CECBF6;
--te-c:#0F6E56; --te-s:#5DCAA5; --te-t:#9FE1CB;
--am-s:#FAC775;
--fl-s:#B4B2A9;
--op:#2C2C2A;
}#fig-blk text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}#fig-blk .bg{fill:var(--bg)}#fig-blk .h1{font-size:15px;font-weight:500;fill:var(--t1)}#fig-blk .sm{font-size:11px}#fig-blk .xs{font-size:10.5px}#fig-blk .md{font-size:12px;fill:var(--t2)}#fig-blk .pn{fill:var(--pn)}#fig-blk .nu-p{fill:var(--nu-p);stroke:var(--nu-s);stroke-width:.5}#fig-blk .bl-c{fill:var(--bl-c);stroke:var(--bl-s);stroke-width:.5}#fig-blk .pu-c{fill:var(--pu-c);stroke:var(--pu-s);stroke-width:.5}#fig-blk .te-c{fill:var(--te-c);stroke:var(--te-s);stroke-width:.5}#fig-blk .op{fill:var(--op);stroke:var(--nu-s);stroke-width:1}#fig-blk .t-bl{fill:var(--bl-t)}#fig-blk .t-pu{fill:var(--pu-t)}#fig-blk .t-te{fill:var(--te-t)}#fig-blk .t-nu{fill:var(--nu-t)}#fig-blk .t-am{fill:var(--am-s)}#fig-blk .fl{stroke:var(--fl-s);stroke-width:1.2;fill:none}#fig-blk .rs{stroke:var(--am-s);stroke-width:1.5;fill:none}#fig-blk .brk{stroke:var(--nu-s);stroke-width:1;stroke-dasharray:4 3;fill:none}#fig-blk .smry{fill:var(--bg);stroke:var(--nu-s);stroke-width:.5}#fig-blk .mid{text-anchor:middle}</style>
<rect class="bg" x="0" y="0" width="680" height="572" rx="10"/>
<text class="h1" x="40" y="26">解码器一层里到底有几个注意力</text>
<rect class="pn" x="40" y="40" width="600" height="444" rx="10"/>
<path class="brk" d="M68 80L56 80L56 452L68 452"/>
<text class="xs t-nu mid" x="24" y="266" transform="rotate(-90 24 266)">× N 层重复</text>
<path class="fl" d="M250 172L250 159" marker-end="url(#fig-blk-ar)"/>
<path class="fl" d="M250 135L250 74" marker-end="url(#fig-blk-ar)"/>
<text class="sm t-nu mid" x="250" y="64">输出 → 下一层</text>
<circle class="op" cx="250" cy="146" r="11"/>
<text class="t-nu mid" x="250" y="146" dominant-baseline="central" style="font-size:14px">+</text>
<text class="xs t-am" x="266" y="150" dominant-baseline="central">残差相加</text>
<rect class="pu-c" x="150" y="172" width="200" height="52" rx="6"/>
<text class="sm t-pu mid" x="250" y="192" dominant-baseline="central">Cross-Attention</text>
<text class="xs t-pu mid" x="250" y="209" dominant-baseline="central">交叉注意力 · Q 由解码器给出</text>
<rect class="te-c" x="430" y="172" width="200" height="52" rx="6"/>
<text class="sm t-te mid" x="530" y="192" dominant-baseline="central">编码器最终输出</text>
<text class="xs t-te mid" x="530" y="209" dominant-baseline="central">源句的 K / V 早已算好</text>
<path class="fl" d="M430 198L356 198" marker-end="url(#fig-blk-ar)"/>
<text class="xs t-te mid" x="392" y="187">K, V</text>
<text class="xs t-te" x="430" y="244">每步 decode 复用同一份 K/V</text>
<path class="fl" d="M250 242L250 226" marker-end="url(#fig-blk-ar)"/>
<rect class="nu-p" x="170" y="242" width="160" height="30" rx="5"/>
<text class="xs t-nu mid" x="250" y="257" dominant-baseline="central">LayerNorm</text>
<text class="xs t-nu" x="360" y="268">Pre-LN：先归一化，再进子层</text>
<path class="fl" d="M250 285L250 274" marker-end="url(#fig-blk-ar)"/>
<circle class="op" cx="250" cy="296" r="11"/>
<text class="t-nu mid" x="250" y="296" dominant-baseline="central" style="font-size:14px">+</text>
<path class="fl" d="M250 322L250 309" marker-end="url(#fig-blk-ar)"/>
<rect class="bl-c" x="150" y="322" width="200" height="52" rx="6"/>
<text class="sm t-bl mid" x="250" y="342" dominant-baseline="central">Masked Self-Attention</text>
<text class="xs t-bl mid" x="250" y="359" dominant-baseline="central">掩码自注意力 · 看自己</text>
<text class="xs t-bl" x="360" y="338">Q、K、V 全部来自本层输入</text>
<text class="xs t-bl" x="360" y="360">因果掩码让未来位置权重为 0</text>
<path class="fl" d="M250 392L250 376" marker-end="url(#fig-blk-ar)"/>
<rect class="nu-p" x="170" y="392" width="160" height="30" rx="5"/>
<text class="xs t-nu mid" x="250" y="407" dominant-baseline="central">LayerNorm</text>
<text class="xs t-nu" x="360" y="407">原论文是 Post-LN，现代实现多用 Pre-LN</text>
<text class="xs t-nu" x="360" y="430">差别只在残差是否穿过那个 LayerNorm</text>
<path class="fl" d="M250 440L250 424" marker-end="url(#fig-blk-ar)"/>
<rect class="nu-p" x="150" y="440" width="200" height="32" rx="6"/>
<text class="xs t-nu mid" x="250" y="456" dominant-baseline="central">输入 x（下层输出 / 词嵌入）</text>
<path class="rs" d="M250 434L112 434L112 296L239 296" marker-end="url(#fig-blk-ar)"/>
<path class="rs" d="M250 280L84 280L84 146L239 146" marker-end="url(#fig-blk-ar)"/>
<rect class="smry" x="40" y="494" width="600" height="66" rx="8"/>
<text class="sm t-nu" x="56" y="512">注意力子层只有两个：一个在自己的序列上做掩码注意力，一个向编码器取信息。</text>
<text class="sm t-nu" x="56" y="529">两者各被「残差 + LayerNorm」包住；橙色折线＝残差捷径，把输入原样送到加法处。</text>
<text class="sm t-nu" x="56" y="546">此处只画注意力，省略了其后的前馈网络 FFN。</text>
</svg>
<p class="fig-cap">图 3：解码器单层的结构。两个注意力子层各自被「残差捷径 + LayerNorm」包住，橙色折线是残差捷径；编码器输出的 K/V 从右侧接入交叉注意力。此处只画注意力子层，省略了其后的前馈网络 FFN。</p>
</div>


## 7. 为什么它成了大模型底座？

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 并行度 | 串行，几乎不能并行 | 全序列一次矩阵运算，GPU 友好 |
| 长程依赖 | 路径长、易遗忘 | 任意两点一步直达 |
| 可扩展性 | 堆深困难 | 残差+LayerNorm 支持堆到上百层 |
| 硬件友好 | 不规则循环 | 全是标准矩阵乘（MatMul） |

正因为这些，**GPT（只用解码器）、BERT（只用编码器）、ViT（把图当序列）、whisper、Sora 乃至 Diffusion Policy 的动作头**，底层都是这套注意力机制。可以说 2017 年这一篇，定义了之后八年深度学习的「通用计算原语」。

## 8. 小结与延伸

- Transformer = 自注意力 + 多头 + 位置编码 + 残差/LayerNorm。
- 它用「全局一步注意力」换掉了「串行循环」，把序列建模变成可并行、可扩展的矩阵运算。
- 想继续往下读：GPT 系列（ decoder-only 如何做生成）、BERT（双向编码做理解）、以及本博客「vLLM 与 SGLang 框架解码手记」里 KV Cache 如何成为注意力的工程瓶颈。

> 下一篇（pp2）我们聊 **Diffusion Policy**——它把 Transformer 同款的「注意力范式」之外的另一条线（扩散去噪）搬到了机器人动作生成上，正好和本系列的「VLA 解码手记」衔接。
