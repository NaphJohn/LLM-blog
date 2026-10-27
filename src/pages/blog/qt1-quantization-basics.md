---
title: '量化手记（一）：量化的基本语言——粒度、对称性、校准，与四条路线怎么选'
description: '为什么 W4A16、W8A8、W4A4KV4 这些命名里第一个字母是权重而不是激活？per-group=128 这个数字从哪来？对称量化为什么天生比非对称少一档？校准集到底校准的是什么？这一篇把量化的公分母讲清楚：一个反量化公式、三个旋钮（粒度/对称性/校准），以及 RTN、GPTQ、AWQ/SmoothQuant、QuaRot 四条路线分别在改哪个旋钮。'
pubDate: 2026-10-27
series: 量化手记
lang: zh
altLang: en
altHref: /en/blog/qt1-quantization-basics
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

所有量化方法——不管是 RTN、GPTQ、AWQ、SmoothQuant 还是 QuaRot——最终都落在**同一个反量化公式**上，区别只在两件事：**scale 怎么算**，以及**在量化之前对张量做了什么等价的数学变换**。前者是量化器设计，后者是分布改造。把这个公分母讲清楚，后面四篇就不会再是一堆陌生名词。

<div class="keybox">
<strong>本篇的位置：</strong>这是「量化手记」的开篇，只讲语言和地图，不讲具体算法推导。qt2 讲 RTN 与 GPTQ，qt3 讲 AWQ 与 SmoothQuant，qt4 讲 QuaRot / SpinQuant / DuQuant，qt5 讲 KV Cache 量化与 FP4。原来的算子篇 op2 保留为系列导读与谱系总表。
</div>

## 1. 一个公式统一所有量化方法

不管什么方法，量化只有两步：**量化（存下来）**和**反量化（算的时候还原）**。

```text
# 量化：把浮点 x 压成 b 位整数 q
q = clamp( round(x / s) + z ,  0 ,  2^b - 1 )

# 反量化：把 q 还原成浮点 ŷ
ŷ = s * ( q - z )
```

其中只有两个参数：

- **scale `s`**：一个量化格代表多大的数值范围。`s` 越小，分辨率越细，但能表示的区间越窄，超出的直接饱和截断（clipping）。
- **zero-point `z`**：这个整数网格的"零点"落在哪。用来让浮点 0 能被精确表示。

<div class="fig">
<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="对称量化与非对称量化的网格对比">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">同一个张量，两种网格：差别在于「零」落在哪一格</text>

  <!-- 上半：对称 -->
  <text x="20" y="56" font-size="14" font-weight="700" fill="#1d4ed8">对称量化（symmetric，z = 0）—— 权重的主场</text>
  <line x1="40" y1="90" x2="640" y2="90" stroke="#9ca3af" stroke-width="1.5"/>
  <!-- 刻度 -->
  <g stroke="#9ca3af" stroke-width="1.5">
    <line x1="40" y1="84" x2="40" y2="96"/><line x1="115" y1="84" x2="115" y2="96"/>
    <line x1="190" y1="84" x2="190" y2="96"/><line x1="265" y1="84" x2="265" y2="96"/>
    <line x1="340" y1="84" x2="340" y2="96"/><line x1="415" y1="84" x2="415" y2="96"/>
    <line x1="490" y1="84" x2="490" y2="96"/><line x1="565" y1="84" x2="565" y2="96"/>
    <line x1="640" y1="84" x2="640" y2="96"/>
  </g>
  <text x="40" y="112" font-size="11" fill="#374151" text-anchor="middle">-8</text>
  <text x="115" y="112" font-size="11" fill="#374151" text-anchor="middle">-6</text>
  <text x="190" y="112" font-size="11" fill="#374151" text-anchor="middle">-4</text>
  <text x="265" y="112" font-size="11" fill="#374151" text-anchor="middle">-2</text>
  <text x="340" y="112" font-size="11" fill="#374151" text-anchor="middle">0</text>
  <text x="415" y="112" font-size="11" fill="#374151" text-anchor="middle">2</text>
  <text x="490" y="112" font-size="11" fill="#374151" text-anchor="middle">4</text>
  <text x="565" y="112" font-size="11" fill="#374151" text-anchor="middle">6</text>
  <text x="640" y="112" font-size="11" fill="#374151" text-anchor="middle">7</text>
  <text x="20" y="136" font-size="12" fill="#6b7280">int4 有符号 [−8, 7]：共 16 个格，但正负不对称，且 0 恰在格点 → 表示 0 无损</text>
  <text x="20" y="156" font-size="12" fill="#6b7280">优点：矩阵乘时无 zero-point 项，kernel 更简单、更贴近 Tensor Core 的原生通路</text>

  <!-- 分隔 -->
  <line x1="20" y1="172" x2="660" y2="172" stroke="#e5e7eb" stroke-width="1.5"/>

  <!-- 下半：非对称 -->
  <text x="20" y="196" font-size="14" font-weight="700" fill="#047857">非对称量化（asymmetric，z ≠ 0）—— 激活的主场</text>
  <line x1="40" y1="226" x2="640" y2="226" stroke="#9ca3af" stroke-width="1.5"/>
  <g stroke="#9ca3af" stroke-width="1.5">
    <line x1="40" y1="220" x2="40" y2="232"/><line x1="115" y1="220" x2="115" y2="232"/>
    <line x1="190" y1="220" x2="190" y2="232"/><line x1="265" y1="220" x2="265" y2="232"/>
    <line x1="340" y1="220" x2="340" y2="232"/><line x1="415" y1="220" x2="415" y2="232"/>
    <line x1="490" y1="220" x2="490" y2="232"/><line x1="565" y1="220" x2="565" y2="232"/>
    <line x1="640" y1="220" x2="640" y2="232"/>
  </g>
  <text x="40" y="248" font-size="11" fill="#374151" text-anchor="middle">0</text>
  <text x="115" y="248" font-size="11" fill="#374151" text-anchor="middle">2</text>
  <text x="190" y="248" font-size="11" fill="#374151" text-anchor="middle">4</text>
  <text x="265" y="248" font-size="11" fill="#374151" text-anchor="middle">6</text>
  <text x="340" y="248" font-size="11" fill="#374151" text-anchor="middle">8</text>
  <text x="415" y="248" font-size="11" fill="#374151" text-anchor="middle">10</text>
  <text x="490" y="248" font-size="11" fill="#374151" text-anchor="middle">12</text>
  <text x="565" y="248" font-size="11" fill="#374151" text-anchor="middle">14</text>
  <text x="640" y="248" font-size="11" fill="#374151" text-anchor="middle">15</text>
  <text x="300" y="196" font-size="12" fill="#047857">uint4 [0, 15]：16 个格全部用在真实区间上</text>
</svg>
<p class="cap">图 1：对称量化的 16 个格要被正负两侧平分（且 −8 在实际数据里通常冗余），非对称量化的 16 个格可以全部落在 [min, max] 区间内。激活经过 ReLU/SwiGLU 后分布偏斜（负侧压缩、正侧长尾），非对称因此更省精度；权重零均值对称分布，对称量化就够了，还省掉 zero-point 的运算。</p>
</div>

### 1.1 三个常见的混淆点

**(1) 对称量化不是"少一格"，而是"少一档分辨率"。** int4 有符号范围是 [−8, 7]，看起来 16 格都能用；但因为 `z=0` 固定，实际是把 `[−absmax, +absmax]` 映射到 `[−8, 7]`，**正负两侧共享同一个 scale**。如果真实分布的 |min| 远小于 max（激活的典型情况），负侧就有一半格被浪费。这就是为什么权重量化偏爱对称、激活量化偏爱非对称。

**(2) `W4A16` 这种命名里，第一个字母永远是权重。** 格式是 `W<权重位宽>A<激活位宽>`。所以：

| 命名 | 权重 | 激活 | 典型方法 | 场景 |
|---|---|---|---|---|
| W8A8 | INT8/FP8 | INT8/FP8 | SmoothQuant、FP8 per-tensor | 数据中心高吞吐 |
| W4A16 | INT4 | FP16/BF16 | **GPTQ、AWQ** | 单卡/端侧部署主流 |
| W4A8 | INT4 | INT8 | GPTQ + SmoothQuant | 折中 |
| W4A4 | INT4/FP4 | INT4/FP4 | **QuaRot、SpinQuant** | 极致压缩 |
| W4A4KV4 | 4 | 4 | QuaRot 系列 | 长上下文 + 低显存 |

**注意 A16 的"16"不是"没量化"，而是激活保持高精度不量化。** 这类方法只优化权重显存与带宽——而 decode 阶段恰恰是**带宽瓶颈**（batch 小、每步只读一遍权重），所以 W4A16 的端到端收益在 decode 上非常明显，在 prefill 上则几乎无感（prefill 是算力瓶颈）。这一条决定了后面所有"该选哪种量化"的判断。

**(3) 位宽不等于信息量。** 4 bit 的均匀量化在假设分布均匀时最优；但 LLM 权重近似高斯，激活近似拉普拉斯+outlier。所以同样是 4 bit，**改变数值分布的预处理（旋转、缩放）比换量化器更能挽回精度**——这是 qt3、qt4 的全部动机。

## 2. 旋钮一：粒度（granularity）

粒度问的是"**一个 scale 管多少个元素**"。这是量化里最影响精度、也最影响工程复杂度的选择。

| 粒度 | 一个 scale 覆盖 | scale 数量（d=4096 线性层） | 精度 | 硬件友好度 |
|---|---|---|---|---|
| per-tensor | 整个张量 | 1 | 最差 | ★★★ 最好（标量缩放） |
| per-channel（per-column） | 一个输出通道 | 4096 | 中 | ★★★ 好（向量缩放，可融合进 epilogue） |
| per-group（group=128） | 128 个连续权重 | 4096×4096/128 = 131072 | 好 | ★★ 需查表 / gather |
| per-token（动态） | 一个 token 的激活向量 | 随 batch 变化 | 好（对激活） | ★★ 需运行时归约 |

<div class="keybox">
<strong><code>group_size = 128</code> 是怎么来的？</strong> 不是理论最优，而是<strong>开销/精度的工业折中点</strong>。一个 group 的 scale 通常存 FP16（16 bit），摊到 128 个元素上等于每元素多 0.125 bit。W4 变成有效 4.125 bit，压缩率从 4× 略降——可接受。若用 group=64，额外开销翻倍到 0.25 bit；若用 group=32，就逼近 0.5 bit，边际收益已经不划算。同时 128 是 GPU 上一次 warp/线程块读写能对齐的自然宽度。GPTQ、AWQ 的默认配置、以及几乎所有推理引擎的 <code>group_size</code> 参数默认值都是 128，来源就在这里。
</div>

**权重的粒度选择**几乎总是 per-group(128) 或 per-channel；**激活**则几乎是 per-token 动态（因为激活的范围随输入变化，必须运行时算）。

<div class="warnbox">
<strong>一个容易踩的坑：</strong>per-group 量化让权重不再是"规整的矩阵"，反量化需要在矩阵乘之前做 gather + scale，这会破坏 Tensor Core 的连续访存。这就是为什么很多部署会**把量化粒度对齐到 K 维的 tile 大小**（比如 group=128 且 K 是 128 的倍数），或者干脆在 kernel 里做 online dequant（反量化融进 GEMM 的主循环，见 qt2 的落地部分）。
</div>

## 3. 旋钮二：校准（calibration）——PTQ 唯一的数据依赖

**权重不需要校准**：训练完就是定值，absmax 直接读出来即可。

**激活需要校准**：激活的范围取决于输入，而推理时你不可能预知用户会问什么。所以 PTQ 用一小批**校准数据**跑一遍前向，记录每一层激活的统计量，再据此定 scale。

三种常见的统计量：

```text
# ① absmax：取校准集上绝对值的最大值
s = max(|X|) / (2^(b-1) - 1)

# ② percentile：取 99.9% 分位数，牺牲极端值换分辨率
s = percentile(|X|, 99.9) / (2^(b-1) - 1)

# ③ MSE 搜索：在 absmax 的一个比例区间上网格搜索，最小化重建误差
for r in [0.80, 0.85, ..., 1.00]:
    s = r * max(|X|) / (2^(b-1) - 1)
    loss = || dequant(quant(X, s)) - X ||^2
选 loss 最小的 s
```

实测经验：**percentile(99.9%) 和 MSE 搜索通常比朴素 absmax 好 0.1~0.3 perplexity**，因为 absmax 会被单个 outlier 绑架，把绝大多数正常值压到零附近。而 LLM 的激活恰恰是 outlier 重灾区（见 qt3）——所以校准策略在 LLM 上比在 CV 模型上重要得多。

校准集本身也有讲究：**128 条左右的 wiki-text 或 c4 样本**是通行做法。太少（<32）统计不稳，太多收益饱和而耗时线性增长。更重要的是**领域要匹配**：拿纯英文 wiki 校准的模型去跑中文或代码任务，激活分布对不上，掉点会明显放大。

## 4. 四条路线：本系列的地图

现在可以回答"GPTQ、AWQ、QuaRot 到底什么关系"了。它们不是互相竞争的四个方法，而是**改不同旋钮的四条路线**：

<div class="fig">
<svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="量化四条路线地图">
  <defs>
    <marker id="qarr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#6b7280"/>
    </marker>
  </defs>
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">四条路线：改的不是同一个东西</text>

  <!-- 起点 -->
  <rect x="20" y="42" width="120" height="56" rx="8" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="80" y="66" font-size="12" font-weight="700" fill="#111827" text-anchor="middle">浮点模型</text>
  <text x="80" y="84" font-size="11" fill="#6b7280" text-anchor="middle">FP16 / BF16</text>

  <!-- 分支线 -->
  <line x1="140" y1="70" x2="176" y2="70" stroke="#6b7280" stroke-width="1.5" marker-end="url(#qarr)"/>
  <line x1="176" y1="70" x2="176" y2="60" stroke="#6b7280" stroke-width="1.5"/>
  <line x1="176" y1="70" x2="176" y2="130" stroke="#6b7280" stroke-width="1.5"/>
  <line x1="176" y1="70" x2="176" y2="200" stroke="#6b7280" stroke-width="1.5"/>
  <line x1="176" y1="70" x2="176" y2="270" stroke="#6b7280" stroke-width="1.5"/>
  <line x1="176" y1="70" x2="176" y2="340" stroke="#6b7280" stroke-width="1.5"/>

  <!-- 路线 A：只改舍入 -->
  <line x1="176" y1="60" x2="206" y2="60" stroke="#6b7280" stroke-width="1.5" marker-end="url(#qarr)"/>
  <rect x="210" y="34" width="452" height="56" rx="8" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="224" y="54" font-size="12" font-weight="700" fill="#1d4ed8">A · 只改舍入：不动分布，只决定「怎么 round」</text>
  <text x="224" y="74" font-size="11" fill="#374151">RTN（直接取整）→ GPTQ（Hessian 补偿误差）→ 目标 W4A16，权重带宽瓶颈</text>

  <!-- 路线 B：改分布 - 缩放 -->
  <line x1="176" y1="130" x2="206" y2="130" stroke="#6b7280" stroke-width="1.5" marker-end="url(#qarr)"/>
  <rect x="210" y="104" width="452" height="56" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="224" y="124" font-size="12" font-weight="700" fill="#047857">B · 改分布 · 缩放：把 outlier 从难量化的地方搬走</text>
  <text x="224" y="144" font-size="11" fill="#374151">SmoothQuant（激活→权重）· AWQ（放大显著通道）→ 目标 W8A8 / W4A16</text>

  <!-- 路线 C：改分布 - 旋转 -->
  <line x1="176" y1="200" x2="206" y2="200" stroke="#6b7280" stroke-width="1.5" marker-end="url(#qarr)"/>
  <rect x="210" y="174" width="452" height="56" rx="8" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="224" y="194" font-size="12" font-weight="700" fill="#534AB7">C · 改分布 · 旋转：换一组基，让 outlier 摊平</text>
  <text x="224" y="214" font-size="11" fill="#374151">QuaRot（Hadamard）· SpinQuant（可学习 R）· DuQuant（旋转+置换）→ W4A4KV4</text>

  <!-- 路线 D：改训练 -->
  <line x1="176" y1="270" x2="206" y2="270" stroke="#6b7280" stroke-width="1.5" marker-end="url(#qarr)"/>
  <rect x="210" y="244" width="452" height="56" rx="8" fill="#faeeda" stroke="#BA7517"/>
  <text x="224" y="264" font-size="12" font-weight="700" fill="#854F0B">D · 改训练：让模型自己学会在低精度下工作</text>
  <text x="224" y="284" font-size="11" fill="#374151">QAT（伪量化进反向传播）→ 目标 FP4 / INT4，动态范围测底时的唯一出路</text>

  <!-- 收口 -->
  <line x1="176" y1="340" x2="206" y2="340" stroke="#6b7280" stroke-width="1.5" marker-end="url(#qarr)"/>
  <rect x="210" y="314" width="452" height="56" rx="8" fill="#fef2f2" stroke="#ef4444"/>
  <text x="224" y="334" font-size="12" font-weight="700" fill="#b91c1c">落地：KV Cache 量化（qt5）—— 与以上四条正交，压的是「token 数 × 层数」</text>
  <text x="224" y="354" font-size="11" fill="#374151">K per-channel / V per-token、FP8 E4M3/E5M2、FP4 微缩放格式</text>

  <!-- 底部说明 -->
  <text x="20" y="392" font-size="12" fill="#6b7280">可叠加：C（预变换）→ B（再缩放）→ A（最后定舍入）→ D（训练补偿）。A 与 B、C 是正交的。</text>
</svg>
<p class="cap">图 2：四条路线改的是量化的不同环节。A 只动量化器本身；B、C 在量化之前插入一个「数学上等价、数值上更友好」的变换；D 把量化误差放进反向传播。A 可以和 B、C 任意叠加——AWQ 定 scale 后再用 GPTQ 定舍入，是生产里最常见的组合。</p>
</div>

**这里有一个值得停一下的洞察：B 和 C 都是"等价变换"。** 

线性的好处是：只要变换是正交或者可逆的，模型输出**在数学上完全不变**。也就是说：

```text
Y = X W                      # 原始计算
Y = (X R)(R⁻¹ W)             # 数学上恒等，只要 R 可逆
Y = (X S⁻¹)(S W)             # 数学上恒等，只要 S 是对角可逆
```

但 `XR` 和 `X` 的数值分布不一样了——**outlier 从"集中在少数坐标"变成"摊平在所有坐标"**，于是同一个 4-bit 量化器的误差大幅下降。这就是为什么 QuaRot 敢说"无需训练"：它没有改变模型学到的任何东西，只是换了一组基来存放同样的信息。

这个技巧也是理解后面三篇的钥匙。qt3 的 `S` 是对角矩阵（缩放），qt4 的 `R` 是正交矩阵（旋转），两者都在"不改模型、只改数值分布"的框架内。

## 5. 精度该怎么量：perplexity 不够

量化论文的表格几乎都用 perplexity（PPL）掉点来说话。实际部署时要注意：

- **PPL 对量化不敏感，对下游任务敏感。** 一个 W4 模型可能 PPL 只涨 0.1，但代码生成或数学推理掉 5~10 个点——因为长尾 token 的分布被压坏了。选型时至少要跑 MMLU / GSM8K / HumanEval 里对口的那一个。
- **长上下文是最先崩的。** KV Cache 量化（qt5）和激活量化都会随序列长度放大误差累积。测的时候要按目标上下文长度跑，不能只在 2K 上测。
- **看"最坏层"而不是"平均层"。** 量化误差高度不均匀：往往只有 1~2 层（通常是第一层和最后一层、或者 down_proj）贡献了大半损失。GPTQ/AWQ 的很多变种都在做"混合精度"，就是把这几个层留在 8 bit 或 16 bit。

## 6. 选型速查

<div class="keybox">
<strong>先定瓶颈，再定方法——顺序不能反。</strong><br>
① <strong>显存墙</strong>（装不下）：KV Cache 量化/驱逐（qt5）> 权重 4bit（qt2/qt3）<br>
② <strong>带宽墙</strong>（decode 慢、batch 小）：权重 W4A16（AWQ 或 GPTQ），收益最大<br>
③ <strong>算力墙</strong>（prefill 慢、batch 大）：激活也要量化 → W8A8（SmoothQuant）或 FP8<br>
④ <strong>既要又要</strong>（W4A4 且不能掉点）：旋转（qt4）+ 量化器（qt2）叠加，或直接上 QAT
</div>

具体到"我该用哪一个"：

| 你的情况 | 建议 | 去读 |
|---|---|---|
| 单卡 24G 跑 32B，要最快落地 | AWQ 或 GPTQ W4A16 + group=128 | qt2 / qt3 |
| 有 GPU 可以跑几十分钟离线压缩 | GPTQ（Hessian 补偿，精度更好） | qt2 |
| 权重和激活都要 8bit，prefill 是瓶颈 | SmoothQuant W8A8 | qt3 |
| 要做到 W4A4 且逼近无损 | QuaRot / SpinQuant | qt4 |
| 长上下文（128K+）显存爆了 | KV Cache FP8 + 分层缓存 | qt5 + sys6 |
| FP4 硬件（Blackwell 级） | NVFP4 + QAT | qt5 |

## 7. 系列导航

- **qt2 · RTN 与 GPTQ**：A 路线。从 `round()` 到 Hessian 误差补偿的完整推导，附可跑伪代码。
- **qt3 · AWQ 与 SmoothQuant**：B 路线。两个方法方向相反地解决同一个问题——激活 outlier。
- **qt4 · QuaRot / SpinQuant / DuQuant**：C 路线。旋转为什么能"无损地"消掉 outlier，以及它在计算图的哪四个位置插入。
- **qt5 · KV Cache 量化与 FP4**：正交的一条线，也是长上下文最直接的显存杠杆。

站内相关：**op2（模型量化部署谱系）** 是本系列的导读与总表；**op3（昇腾 EagleQBSA）** 是这里所有理论在真实算子里的落地（QK INT8 / PV FP8 的分布感知配方）；**sys9 / sys10 / sys6** 讲 KV Cache 在系统层的管理。
