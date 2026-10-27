---
title: '量化手记（七）：量化的三个边界——投机解码、MLA 结构与端侧硬件'
description: 'qt1–qt6 都在量化内部讨论「怎么把误差压到最小」，这一篇向外看：量化不是一个可以独立优化的变量。第一条边界是投机解码——4 bit 让单 token 前向变快，却让树验证变慢，最优草稿树大小反而要从 60 降到 16；第二条是 MLA/GQA 结构——KIVI 的「K 用 per-channel、V 用 per-token」在 MLA 下直接失效，因为 cache 里存的东西根本不是 K 和 V；第三条是端侧硬件——决定一个量化格式能不能落 NPU 的是格式而不是位宽。三条边界最后指向同一句话：硬件决定格式，格式决定算法，算法只在前两者给定之后才有优化空间。'
pubDate: 2026-10-27
series: 量化手记
lang: zh
altLang: en
altHref: /en/blog/qt7-quantization-boundaries
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

前六篇有一个隐含的框架：**给定一个模型、一种位宽，怎么让量化误差最小**。这是个干净的优化问题——有目标函数（PPL 或逐层重建误差 `‖WX − ŴX̂‖`），有约束（4 bit、group size 128），有方法（RTN/GPTQ/AWQ/旋转/QAT）。

但真实的部署不是这样的。你量化完一个模型之后，它会立刻撞上三个外部约束：

```text
① 它要和别的加速手段叠加     → 投机解码
② 它要跑在某种注意力结构上   → GQA / MLA
③ 它要落到某种硬件上         → NPU / GPU / 数据中心卡
```

这三个约束中的每一个，都**能让"量化内部最优"的选择变成错的**——不是精度数字变差一点，而是整套配置的结论翻转：最优草稿树大小要减半、per-token scale 要放弃、4 bit 能不能用取决于你选的是 `Q4_0` 还是 `Q4_K_M`。

<div class="keybox">
<strong>本篇位置：</strong>qt1–qt6 讨论「量化内部怎么优化」，本篇讨论「量化受什么约束」。<strong>核心判断提前给：</strong>量化从来不是一个可以独立调优的变量，它是<strong>三个外部条件的函数</strong>。看到一篇论文说"我们的 4 bit 方案无损"，正确的问题不是"掉了多少 PPL"，而是"在什么树大小下、什么注意力结构上、什么硬件通路里无损"。前六篇的所有结论都是<strong>有条件成立</strong>的，这一篇把条件写出来。
</div>

## 1. 第一条边界：和投机解码叠加后，最优配置会翻转

### 1.1 先把加速比公式写清楚

投机解码的加速比不是"接受率越高越快"，它由一个更完整的式子决定：

```text
加速比 ≈ (α·k + 1) / (k·c_target + c_draft)
```

其中 `α` 是接受率、`k` 是草稿 token 数（或树的宽度）、`c_target` 是目标模型单次前向成本、`c_draft` 是草稿模型单次前向成本。

分母里两项都在做加法，但它们的**缩放方式完全不同**：`k·c_target` 随草稿长度线性增长（验证 k 个 token 就要跑 k 次目标模型前向，虽然可以 batch），而 `c_draft` 是固定开销。所以这个式子里藏着两个可以独立调节的旋钮：

- **提高 α**：分子变大，直接收益
- **压低 c_draft**：分母变小，直接收益

量化 draft 模型的直觉来自第二个旋钮——draft 小、跑得多、量化它收益大。但这里有两个陷阱，一个已经被踩过很多次，另一个直到 2025 年才被量化清楚。

### 1.2 陷阱一：draft 量化对接受率几乎无损，前提是别用错精度族

先给好消息：**draft 模型量化到 4 bit，接受率几乎不掉。**

SGLang 的一个实测（PR #39254，Qwen 系列 27B 级模型 + DFlash 投机头）给出了很干净的数字：

| draft 精度 | 平均接受长度 | 接受率 |
|---|---|---|
| BF16 | 3.25 | 0.31 |
| W4A16 | 3.23 | 0.32 |

位宽从 16 降到 4，接受长度只掉 0.02，接受率甚至微升（在噪声范围内）。原因不难理解：**投机解码的接受判据是 `u < p_target(x)/p_draft(x)`，它是一个比值**。draft 和 target 用同一种量化误差时，分子分母的偏差部分抵消；而即使不抵消，4 bit 引入的 top-k 排序扰动远小于采样噪声。

然后是坏消息，而且这个坏消息比它看起来严重得多：

**精度族必须和目标模型匹配，否则接受率归零。**

同一个 draft 模型（`-DFlash-NVFP4` 版本），配不同精度的 target：

| target 精度 | 接受率 |
|---|---|
| FP8 | **0%** |
| NVFP4 | **~62%** |

不是掉一半，是**完全归零**。这看起来荒谬——NVFP4 的 draft 配 FP8 的 target，精度明明更高，为什么反而崩？

原因是那个判据 `u < p_target/p_draft` 的**隐含假设**：它假设 draft 和 target 对同一个 token 给出的概率，差异只来自"谁更懂语言"，而不来自"谁用了不同的舍入方式"。NVFP4 和 FP8 的差异不只是一个更精确——它们的**误差结构完全不同**：NVFP4 是 16 元素一块 + FP8 scale，FP8 是逐张量或逐通道缩放。两个模型对同一个 logits 向量的舍入方向甚至可能是反的，于是 `p_draft` 在某些 token 上系统性偏大，判据 `u < p_target/p_draft` 大量误拒。

<div class="warnbox">
<strong>这条结论的实用形式：</strong>draft 和目标模型<strong>要么都用同一个量化格式族，要么 draft 保持 FP16/BF16</strong>。不要"draft 用更激进的量化、target 保持高精度"——这个组合在理论上像是赚的（分母更小、分子不变），实测是零加速甚至负加速。<strong>而最容易骗过你的指标是 PPL</strong>：NVFP4 draft 的 PPL 完全正常，甚至可能比 FP8 版本更接近 BF16，因为它确实是个更精确的格式。PPL 正常、接受率为零，这两件事可以同时成立。
</div>

### 1.3 陷阱二：4 bit 下草稿树该变小，不是变大

这是本轮查证里最反直觉的一条，也是我觉得最值得写进博客的一条。

先给背景。草稿树（tree speculation）的直觉是"树越大，一次验证能捞回的 token 越多"。在 BF16 下这个直觉基本成立：树宽从 8 加到 60，接受长度稳定上升，因为验证的边际成本（目标模型一次 batch 前向）相对便宜。

**但把目标模型量化到 4 bit 之后，这个关系会翻转。**

HierSpec（arXiv 2505.22179）做了一个很漂亮的测量：它定义了一个指标叫 **V/D**（Verification / Decoding 时间比），即"验证一次草稿树的时间" ÷ "纯自回归解码一个 token 的时间"。理想的投机解码应该让 V/D ≈ 1.0——验证一棵树约等于解一个 token 的成本，这才是"免费的并行"。

实测（A100 + Llama-3-8B）：

| 模型精度 | 树大小 | V/D |
|---|---|---|
| BF16 | 60 | ≈ 1.0（可接受） |
| **W4A16** | **60** | **1.8** ← 已经亏了 |
| **W4A16** | **约 16** | ≈ 1.0 |

同一个树大小 60，BF16 下 V/D 合理，4 bit 下 V/D 涨到 1.8——**验证变成了瓶颈**，加速比反而下降。

原因值得拆开看，因为它暴露了"量化收益"的一个边界条件：

<div class="fig">
<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="4bit 量化后草稿树的最优大小向左移动">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">同样是 4 bit，单 token 前向变快 2 倍，树验证只快 1.15 倍</text>
  <text x="20" y="42" font-size="11" fill="#6b7280">纵轴 = V/D（验证一棵树 ÷ 解码一个 token 的时间），理想值 1.0；横轴 = 草稿树大小</text>

  <!-- 坐标系 -->
  <line x1="70" y1="66" x2="70" y2="250" stroke="#9ca3af" stroke-width="1.2"/>
  <line x1="70" y1="250" x2="640" y2="250" stroke="#9ca3af" stroke-width="1.2"/>

  <!-- y 轴刻度 -->
  <g font-size="10" fill="#6b7280" text-anchor="end">
    <text x="62" y="72">2.4</text>
    <text x="62" y="112">1.8</text>
    <text x="62" y="152">1.2</text>
    <text x="62" y="192">0.6</text>
  </g>
  <g stroke="#e5e7eb" stroke-width="1">
    <line x1="70" y1="68" x2="640" y2="68"/>
    <line x1="70" y1="108" x2="640" y2="108"/>
    <line x1="70" y1="148" x2="640" y2="148"/>
    <line x1="70" y1="188" x2="640" y2="188"/>
  </g>

  <!-- V/D = 1.0 理想线 -->
  <line x1="70" y1="128" x2="640" y2="128" stroke="#15803d" stroke-width="1.4" stroke-dasharray="6 4"/>
  <text x="560" y="122" font-size="10.5" fill="#15803d" font-weight="700">V/D = 1.0 理想线</text>

  <!-- x 轴刻度 -->
  <g font-size="10.5" fill="#374151" text-anchor="middle">
    <text x="110" y="268">8</text>
    <text x="200" y="268">16</text>
    <text x="290" y="268">24</text>
    <text x="380" y="268">32</text>
    <text x="470" y="268">44</text>
    <text x="560" y="268">60</text>
  </g>
  <text x="355" y="288" font-size="11" fill="#6b7280" text-anchor="middle">草稿树大小（可同时验证的 token 数）</text>

  <!-- BF16 曲线：平缓上升，在 60 处约 1.0 -->
  <polyline points="110,196 200,180 290,166 380,152 470,140 560,130"
            fill="none" stroke="#2563eb" stroke-width="2.6"/>
  <circle cx="110" cy="196" r="3.6" fill="#2563eb"/><circle cx="200" cy="180" r="3.6" fill="#2563eb"/>
  <circle cx="290" cy="166" r="3.6" fill="#2563eb"/><circle cx="380" cy="152" r="3.6" fill="#2563eb"/>
  <circle cx="470" cy="140" r="3.6" fill="#2563eb"/><circle cx="560" cy="130" r="4.4" fill="#2563eb"/>
  <text x="572" y="196" font-size="11" fill="#2563eb" font-weight="700">BF16</text>
  <text x="436" y="140" font-size="10" fill="#2563eb">树大 → 更划算</text>

  <!-- W4A16 曲线：陡峭，60 处冲到 1.8 -->
  <polyline points="110,214 200,168 290,142 380,122 470,110 560,98"
            fill="none" stroke="#dc2626" stroke-width="3"/>
  <circle cx="110" cy="214" r="4" fill="#dc2626"/><circle cx="200" cy="168" r="4" fill="#dc2626"/>
  <circle cx="290" cy="142" r="4" fill="#dc2626"/><circle cx="380" cy="122" r="4" fill="#dc2626"/>
  <circle cx="470" cy="110" r="4" fill="#dc2626"/><circle cx="560" cy="98" r="5" fill="#dc2626"/>
  <text x="572" y="102" font-size="11" fill="#dc2626" font-weight="700">W4A16</text>

  <!-- 树大小 60 处的告警标注 -->
  <line x1="560" y1="98" x2="560" y2="250" stroke="#dc2626" stroke-width="1" stroke-dasharray="3 3"/>
  <rect x="392" y="78" width="158" height="20" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <text x="400" y="92" font-size="10" fill="#b91c1c" font-weight="700">树=60 时 V/D=1.8，验证成瓶颈</text>

  <!-- 最优树大小左移 -->
  <rect x="150" y="184" width="112" height="20" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="158" y="198" font-size="10" fill="#15803d" font-weight="700">4 bit 的最优点左移到 ~16</text>

  <!-- 下方解释 -->
  <line x1="20" y1="302" x2="660" y2="302" stroke="#e5e7eb" stroke-width="1"/>
  <text x="20" y="322" font-size="11" fill="#374151">为什么：4 bit 只加速「连续访存」的 GEMM，而树验证是非连续 KV 访问 + 树形 mask + 变长 batch —— 吃不到带宽红利</text>
</svg>
<p class="cap">图 1：量化把"单 token 前向"加速了 2 倍，但把"树验证"只加速了 1.15 倍。两个速度不等比缩放，最优点必然移动——所以 4 bit 下的草稿树应该比 BF16 下更小，而不是更大。</p>
</div>

三个原因，都指向同一件事：

1. **树验证的 KV 访问不连续。** 一棵树里 n 个候选共享前缀，但注意力要按树形 mask 去 gather 不连续的 KV 页。量化带来的是**带宽红利**，而 gather 访问的瓶颈是延迟和索引，不是带宽。
2. **树形 mask + 变长 batch = kernel 效率下降。** 4 bit 的快速 kernel（如 Marlin、Machete）针对的是**固定形状的大矩阵乘**。树验证喂给它的形状是 `[num_nodes, hidden]`，其中 `num_nodes` 随树结构动态变化，还带自定义 attention mask，落到的是慢路径。
3. **去量化开销的摊薄比例不同。** 4 bit 权重要先反量化再算，这个开销是 per-element 固定的。单 token 解码时矩阵乘很大，摊薄没问题；树验证时每个节点的有效计算量小，去量化的相对占比上升。

HierSpec 的解法很漂亮，值得记一下它的思路而不只是数字：既然瓶颈是"用大模型验证一棵大树"不划算，那就在中间插一个 **8B W4A16 的中间模型**去吸收树验证，让 70B 的目标模型只做**顺序验证**（每步只验一个 token，形状固定、走快路径）。结果：**98.19 tok/s，相对自回归 2.78×，相对 EAGLE-2 1.31×**。

> **这条边界的普适表述**：任何"量化 + 另一种并行加速"的叠加，都要分别测量两条路径的加速比。如果两种加速作用在同一段计算上，收益会重叠（1.5× × 1.5× ≠ 2.25×）；如果作用在不同段上，瓶颈会转移。量化加速的是"固定的、大块的、连续的"计算，而树结构验证恰恰是"动态的、碎片的、gather 式"的计算——这是它们不叠加的结构性原因。

### 1.4 两个真实事故：接受率是唯一验收指标

工程上有两类事故，PPL 和 loss 全都正常，只有接受率崩了。

**事故一：vLLM 里 4-bit draft 被静默 bit-cast 成 BF16。** 即使你提供了 4-bit 的 draft 权重，vLLM 默认仍按未量化配置构建模型，然后把 4-bit 张量**按位强转**成 BF16——得到的不是一个数值近似的模型，而是一个**数值完全错误**的模型。表现：接受率 1.03（也就是只接受 bonus token，等于没有投机），零加速。修法显式指定 `"quantization": "modelopt_mixed"`。

**事故二：SGLang 里 `fc.weight_packed` 被 `load_weights()` 静默跳过。** 量化后的权重名是 `fc.weight_packed`（打包过的 int32），而权重加载器只认 `fc.weight`。不匹配的键**不报错、只跳过**，于是 `fc` 层保持随机初始化。表现：接受率崩到只剩 bonus token。

<div class="warnbox">
<strong>两个事故的共同教训：量化链路上的错误不体现在损失上，只体现在接受率上。</strong>一个随机初始化的 fc 层，如果它只在 draft 模型的一小部分路径上，整体 loss 几乎不受影响；一个 bit-cast 错的权重，如果它数值上仍在合理范围（int32 的位模式当 BF16 读，很多值会落在极小或极大的指数区间），PPL 可能只是"略差"。<strong>所以投机解码 + 量化的验收，必须把"平均接受长度"和"接受率"作为一级指标</strong>，和 PPL 并列，而不是事后抽查。<br><br>
具体做法：跑一批固定 prompt，记录 `accept_length` 的均值与分布。健康的 4 bit draft 应该在 BF16 的 95% 以上（如 3.23 vs 3.25）。如果掉到 1.x，先查权重是否真的加载了——不要先怀疑算法。
</div>

## 2. 第二条边界：MLA / GQA 结构下，KV 量化的规则要改

qt5 讲的 KV 量化规则（K 用 per-channel、V 用 per-token）有一个隐含前提：**cache 里存的是 K 和 V，而且它们是两个独立的张量**。这个前提在 GQA 下勉强成立、在 MLA 下直接不成立。

### 2.1 GQA：统计样本变少，误差影响面变大

GQA（Grouped-Query Attention）让多个 query head 共享一组 KV head。看 per-channel scale 的统计口径：

```text
MHA：per-channel scale 的统计样本 = n_h × d_h        （每头独立）
GQA：per-channel scale 的统计样本 = n_kv_heads × d_h  （共享）
```

以 Llama-3-8B 为例：`n_h = 32`，`n_kv_heads = 8`（GQA 4:1）。于是：

- **统计样本从 32 组降到 8 组**——每组 scale 要从更多数据里估出来，估计方差变大（这个还不算严重，因为还有 token 维度可以平均）
- **真正的问题是误差影响面扩大 4 倍**——一个 KV head 的量化误差，现在会影响 4 个 query head 的输出

但 GQA 有一个**反方向的补偿**，这条很少有人讲：K 的 outlier 是"多个 query head 需求的并集"。在 MHA 里，每个 head 有自己的一组 outlier 通道，per-channel 的 scale 要贴合那个 head 的尖锐结构；在 GQA 里，8 个 query head 的注意力模式被平均到一个 KV head 上，**outlier 反而没那么尖了**。

推论：**GQA 下 per-channel 更容易做好，但 per-token 反而更危险**——因为 per-token 是对"这一个 token 在这个 KV head 上的全部通道"取一个 scale，共享后它的粒度变得太粗——它无法区分"这 4 个 query head 里只有 1 个需要那个大通道值"这种情况。qt5 那条"K 用 per-channel"的结论在 GQA 下不但成立，而且比 MHA 下更成立。

### 2.2 MLA：cache 里存的东西根本不是 K 和 V

这是本节的核心。MLA（Multi-head Latent Attention，DeepSeek V2/V3 用的结构）把 KV 压缩成一个低秩 latent，cache 里只存：

| 内容 | 维度 | 说明 |
|---|---|---|
| `c_KV` | 512 | KV 的压缩 latent，K 和 V 都从它投影出来 |
| `k_R` | 64 | 解耦的 RoPE 部分（decoupled RoPE key） |
| **合计** | **576** | vs MHA-7B 的 `2 × 32 × 128 = 8192`，**14× 压缩** |

<div class="fig">
<svg viewBox="0 0 680 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MHA GQA MLA 三种结构的 KV cache 布局对比">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">三种结构下，cache 里到底存了什么</text>
  <text x="20" y="42" font-size="11" fill="#6b7280">以 7B 级模型为例；格子高度不代表精确比例，标注给出实际元素数</text>

  <!-- ── MHA ── -->
  <text x="20" y="76" font-size="12" font-weight="700" fill="#1a1a1a">MHA：K 和 V 是两个独立张量</text>
  <rect x="20" y="86" width="150" height="30" rx="3" fill="#dbeafe" stroke="#93c5fd"/>
  <text x="95" y="105" font-size="11" fill="#1e40af" text-anchor="middle" font-weight="700">K  [32, 128]</text>
  <rect x="180" y="86" width="150" height="30" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="255" y="105" font-size="11" fill="#15803d" text-anchor="middle" font-weight="700">V  [32, 128]</text>
  <text x="348" y="105" font-size="11" fill="#6b7280">= 8192 / token</text>
  <text x="20" y="134" font-size="10.5" fill="#374151">→ 量化策略：K 用 per-channel（结构性 outlier）、V 用 per-token（瞬时性 outlier）</text>
  <text x="20" y="150" font-size="10.5" fill="#15803d">→ 两者是独立张量，可以分别用不同策略，KIVI 的二分法成立</text>

  <line x1="20" y1="166" x2="660" y2="166" stroke="#e5e7eb" stroke-width="1"/>

  <!-- ── GQA ── -->
  <text x="20" y="190" font-size="12" font-weight="700" fill="#1a1a1a">GQA：KV head 数少于 query head 数（Llama-3-8B 为 4:1）</text>
  <rect x="20" y="200" width="62" height="30" rx="3" fill="#dbeafe" stroke="#93c5fd"/>
  <text x="51" y="219" font-size="10" fill="#1e40af" text-anchor="middle">K[8,128]</text>
  <rect x="90" y="200" width="62" height="30" rx="3" fill="#dcfce7" stroke="#86efac"/>
  <text x="121" y="219" font-size="10" fill="#15803d" text-anchor="middle">V[8,128]</text>
  <text x="162" y="219" font-size="11" fill="#6b7280">= 2048 / token（4× 压缩）</text>
  <text x="20" y="248" font-size="10.5" fill="#374151">→ 统计样本 32 组 → 8 组；一个 KV head 的误差影响 4 个 query head</text>
  <text x="20" y="264" font-size="10.5" fill="#15803d">→ 但 outlier 是 4 个 query head 的并集，没那么尖 → per-channel 更容易</text>

  <line x1="20" y1="280" x2="660" y2="280" stroke="#e5e7eb" stroke-width="1"/>

  <!-- ── MLA ── -->
  <text x="20" y="304" font-size="12" font-weight="700" fill="#1a1a1a">MLA：cache 里存的是 latent，K 和 V 都是投影出来的</text>
  <rect x="20" y="314" width="176" height="30" rx="3" fill="#fef3c7" stroke="#fcd34d"/>
  <text x="108" y="333" font-size="11" fill="#92400e" text-anchor="middle" font-weight="700">c_KV  [512]</text>
  <rect x="206" y="314" width="52" height="30" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <text x="232" y="333" font-size="10" fill="#b91c1c" text-anchor="middle" font-weight="700">k_R[64]</text>
  <text x="270" y="333" font-size="11" fill="#6b7280">= 576 / token（14× 压缩）</text>
  <text x="20" y="362" font-size="10.5" fill="#b91c1c">→ K 和 V 是同一个张量 c_KV 的两个投影：无法「K 用 per-channel、V 用 per-token」</text>
</svg>
<p class="cap">图 2：MHA 下 K/V 独立（KIVI 二分法成立）；GQA 下共享但仍是独立张量（结论更成立）；MLA 下只有一个 latent——所有"分别对待 K 和 V"的策略在这里失效。</p>
</div>

### 2.3 特例一：KIVI 的二分法直接失效

qt5 花了整节讲"K 用 per-channel、V 用 per-token"，理由是 K 的 outlier 沿通道维度结构性集中、V 的 outlier 沿 token 维度瞬时出现。

在 MLA 下，这个推理的**前提消失了**：`K = c_KV · W_UK`、`V = c_KV · W_UV`，两者从同一份 `c_KV` 投影而来，cache 里只有 `c_KV` 一个张量。你没法说"这个张量的前半适用 per-channel、后半适用 per-token"——它是一个整体。

正确的做法是：**统一用 per-channel**。理由是 `c_KV` 的结构——它由 SVD 式的低秩分解得到（`W_DKV` 下投影、`W_UK`/`W_UV` 上投影），这个结构会把方差**集中到少数前导 latent 维**上。也就是说 outlier 沿通道维度（512 个 latent 维）结构性集中，而不是沿 token 维度瞬时出现——这正好是"用 per-channel"的充分条件。

同理，qt5 讲的 V 侧 per-token 也失去依据：V 的"瞬时性"来自 softmax 输出的稀疏（某个 token 的注意力权重集中在少数位置），而在 MLA 里 `V` 是 `c_KV` 的线性变换，它的幅值结构由 latent 决定，不由 softmax 决定。

### 2.4 特例二：`k_R` 绝对不能量化

`k_R`（64 维，占 cache 的 11%）是解耦的 RoPE key。它有两个性质让量化变得极其危险：

**性质一：它参与 RoPE，量化误差会变成相位误差。** RoPE 是对 `k_R` 做旋转：`k_R ⊙ [cos(mθ), sin(mθ)]`。如果 `k_R` 有量化误差 `Δk_R`，这个误差会被旋转**搬运到正交分量上**，然后：

```text
scores 的相位误差 ≈ Δk_R 在 (cos, sin) 基上的投影
随位置距离 m 累积（因为 θ_m = m·θ₀ 线性增长）
```

结果是**长上下文下分数误差随距离线性/超线性累积**——短 prompt 测不出来，32k 以上直接崩。这类误差的隐蔽性和 qt5 讲的 softmax 指数放大是同一类。

**性质二：per-token scale 会破坏相对位置的线性可分解性。** RoPE 的核心性质是"相对位置编码可以写成内积形式"，这依赖 `k` 和 `q` 在同一组 `(cos, sin)` 基上旋转。如果每个 token 用自己的 scale，`s_t · (k ⊙ rot(mθ)) ≠ (s_t · k) ⊙ rot(mθ)`——量化和旋转不可交换（这正是 qt4 讲的"旋转免费性边界"在 KV 侧的翻版）。

**结论：`k_R` 保持 FP16。** 收益只有 11%，风险是长上下文崩掉。这是一个明确的不对称赌注，没有讨论空间。

### 2.5 特例三：量化误差会被 `W_UK` 重新混合

这条最技术，也最容易被通用工具忽略。

在 MHA 里，`ΔK` 对 scores 的影响是直接的：`Δscores = Q ΔKᵀ / √d`，通道之间独立。在 MLA 里，cache 里量化的是 `c_KV`，误差要先经过上投影：

```text
Δscores = Q · ΔKᵀ = Q · (W_UK · Δc_KV)ᵀ = Q · W_UK · Δc_KVᵀ
```

于是**误差在进入 scores 之前被 `W_UK` 重新混合了一遍**：`c_KV` 第 `j` 维的量化误差，会按 `W_UK` 第 `j` 列的范数和方向，分配到所有 `d_h` 个注意力维度上。

这带来一个通用工具（GPTQ/AWQ）发现不了的优化：**per-channel scale 不应该按 `c_KV` 各通道的 absmax 定，而应该按"`W_UK` 行范数 × 通道 absmax"加权定**——因为一个 absmax 很大但 `W_UK` 对应列范数很小的通道，它的量化误差根本传不到 scores 里，不值得给它精细的 scale；反过来，一个 absmax 中等但 `W_UK` 列范数很大的通道，误差会被放大，应该优先保精度。

```text
重要性权重（建议）：
  imp[j] = ‖W_UK[:, j]‖₂ × absmax(c_KV[:, j])

对比通用做法：
  imp[j] = absmax(c_KV[:, j])
```

### 2.6 特例四：R4 旋转在 MLA 下需要重推

qt4 讲了 QuaRot 系的 R1–R4 四个旋转插入位置。其中 **R4 是作用在 V 和 O 之间**（`V → R_V`、`O ← R_O`），用来处理 V 侧的 outlier——因为 `V·W_OV` 可以重写成 `(V·R)(Rᵀ·W_OV)`，只要 `R` 是正交矩阵，函数不变但 V 的分布被摊平了。

qt4 的推导有一个前提：**`V` 和 `O` 在同一空间、同维度**，旋转可以直接插入。

<div class="warnbox">
<strong>以下是我的推导，不是文献实测结论——请当作待验证的假设阅读。</strong><br><br>
在 MLA 下，`V` 由 `c_KV` 投影而来（`V = c_KV · W_UV`），而 RoPE 作用在 `k_R` 上、`c_KV` 上还要先叠加 `W_UK` 的混合。如果要在 MLA 下复用 R4 的思路，旋转不能作用在 `V` 上（`V` 是中间产物、不进 cache），而必须作用在 <strong><code>c_KV</code></strong> 上，且维度是<strong>吸收后的 <code>d_c + d_R = 576</code> 维</strong>，而不是 MHA 的 <code>d_h = 128</code> 维；并且必须在 <strong>RoPE 之前</strong>对 `c_KV` 施加（否则破坏 `k_R` 的旋转基）。<br><br>
我倾向认为这个变换在数学上成立（因为 `W_UK`、`W_UV` 是固定的线性映射，正交变换可以吸收进它们），但我<strong>没有找到公开的 MLA-W4A4 旋转量化实测量化结果</strong>来佐证。目前 MLA 模型（DeepSeek V2/V3 系列）的实际部署用的是 W4A16 或 W8，还不需要 R4 这一级的手段——所以这个问题在实践中还没被逼到必须解决。
</div>

## 3. 第三条边界：端侧 int4 的红利是真的，但不在你以为的地方

### 3.1 事实一：端侧主流是 W4A16，不是 W4A4

如果你看过数据中心量化的讨论，会以为"4 bit"指的是 W4A4（权重和激活都 4 bit）。**端侧不是。**

高通 QNN 的模型 bundle 里，绝大多数 LLM 模型的量化配置是 **`w4a16`**——权重 int4、激活 int16。原因很直接：**端侧解码是极端 memory-bound**。

```text
端侧 decode 阶段的算术强度 = 2 × batch / bytes_per_weight
                          ≈ 2 × 1 / 0.5625  ≈ 3.5 FLOP/byte   （4 bit, group 128）

而 Hexagon NPU 的算力/带宽比 ≈ 数十 FLOP/byte
```

也就是说，端侧解码时 NPU 的计算单元有 90% 以上在等数据。这种场景下，**权重位宽是唯一重要的变量**，激活位宽几乎不影响速度（激活的字节数比权重小两个数量级）。把激活也压到 4 bit 只会增加 kernel 复杂度、降低精度，换不来速度。

这是个很有用的校准：**数据中心的"W4A4 才是真 4 bit"叙事，在端侧不成立**。端侧的 W4A16 不是"妥协"，它是这个硬件约束下的正确选择。

### 3.2 事实二：决定能否落 NPU 的是格式，不是位宽

这条我觉得是整个端侧量化里最容易被忽略的规则。

llama.cpp 有一堆 4 bit 量化类型：`Q4_0`、`Q4_1`、`Q4_K_S`、`Q4_K_M`、`Q5_K_M`……直觉上它们"都是 4 bit，应该差不多"。但落到 Hexagon NPU 上：

| 量化类型 | 位宽 | 落点 |
|---|---|---|
| `Q4_0` | 4 bit | **Hexagon NPU** ✓ |
| `Q4_K_M` | ~4.5 bit | GPU / CPU ✗ |
| `Q5_K_M` | ~5.5 bit | GPU / CPU ✗ |

`Q4_K_M` 和 `Q4_0` 的平均位宽只差 0.5 bit，为什么一个能上 NPU、一个不能？

**因为 `Q4_K_M` 是"混合精度超块"（super-block）格式**：它在一个 256 元素的超块里，把 8 个子块（每块 32 元素）的 scale 用**不同精度**存——部分用 6 bit、部分用 4 bit，并额外存一组 super-block scale。这个格式在数学上更好（精度更高），但**NPU 的硬件通路不支持这种嵌套 scale 结构**。

于是：

```text
不是「4 bit 能不能上 NPU」，而是「这个 4 bit 的 scale 布局 NPU 认不认」。
```

<div class="warnbox">
<strong>这条规则的实际形式：</strong>选端侧量化格式时，<strong>先查硬件支持列表，再谈精度</strong>。<code>Q4_0</code> 的精度比 <code>Q4_K_M</code> 差，但如果只有 <code>Q4_0</code> 能跑 NPU，那 <code>Q4_K_M</code> 的精度优势毫无意义——它跑在 CPU 上，速度可能差 5 倍。<strong>这不是"精度换速度"的常规取舍，而是"格式决定通路"的离散跳变。</strong>类似的规则在数据中心也存在（NVFP4 需要 Blackwell、MXFP4 的 E8M0 scale 需要特定 MMA 支持），只是端侧更严格、更不透明。
</div>

### 3.3 事实三：红利在端侧是真的，在数据中心是假的

先给高通的官方口径（算子级微基准）：

| 指标 | INT4 vs INT8 |
|---|---|
| 性能 | **+90%** |
| 能效 | **+60%** |
| 张量吞吐 | **2×** |

这个数字是可信的，但要看清它的测量条件：**算子级微基准、memory-bound 场景、`w4a16` 配置**。在这个条件下，INT4 相对 INT8 几乎线性地减少了权重带宽，所以接近 2× 吞吐是符合预期的。

第三方可复现的端到端数字则温和得多：**Llama 2-7B 在骁龙 8 Gen 3 的 NPU 上 20 tok/s，GPU 上 13 tok/s**。注意这个对比是 **NPU vs GPU**，不是 int4 vs int8——它衡量的是架构差异（NPU 有专门的权重解压/矩阵乘通路），int4 的功劳只是其中一部分。

反过来看数据中心：H100/H200/B200 **都没有原生 INT4×INT4 MMA**（这一点 qt1 就讲了）。在数据中心做 W4A4，权重 4 bit 省下的带宽要花在"反量化到 FP16 再算"上，净收益远小于端侧。这就是为什么"4 bit 加速 2 倍"这个说法在端侧成立、在数据中心不成立——**不是量化方法不同，是硬件通路不同**。

### 3.4 端侧 int4 的收益结构，一句话总结

```text
端侧：memory-bound 极其严重 → 权重位宽是唯一变量 → W4A16 是正解
      格式决定能否落 NPU → 精度不是第一顺位，通路才是
      红利真实但来自「带宽 + 专用通路」，不来自「INT4 算得快」

数据中心：compute-bound 的 prefill + 带宽受限的 decode 混合
      没有原生 INT4 MMA → W4A4 的反量化开销吃掉大半收益
      → W4A16 + FP8 KV 才是当前甜点
```

## 4. 三条边界的共同教训

三条边界看起来毫不相关（一个是解码算法、一个是注意力结构、一个是硬件），但它们的教训是同一句话的三次重复：

<div class="keybox">
<strong>硬件决定格式 → 格式决定算法 → 算法只在前两者给定之后才有优化空间。</strong><br><br>
<strong>边界一（投机解码）：</strong>量化改变了"单 token 前向"和"树验证"两个成本的<strong>比值</strong>，于是最优配置（树大小）跟着变。这是算法层的约束传递。<br>
<strong>边界二（MLA）：</strong>注意力结构决定了 cache 里存什么张量，进而决定了哪种 scale 划分方式有意义。这是结构层的约束传递。<br>
<strong>边界三（端侧）：</strong>硬件支持什么 scale 布局，决定了哪些量化格式可用，进而决定了精度上限。这是硬件层的约束传递。<br><br>
三次传递的方向都一样：<strong>从下往上</strong>。所以"哪种量化方法最好"这个问题<strong>没有脱离上下文的答案</strong>——qt1–qt6 讲的所有方法，都是"给定格式与结构之后的优化"，而不是"能反向选择格式与结构"的方案。
</div>

一张表收束三条边界：

| 边界 | 会失效的默认假设 | 4 bit 下的正确做法 | 踩坑的隐蔽表现 |
|---|---|---|---|
| **投机解码** | 树越大越划算 | 树大小从 60 降到 ~16；或插中间模型吸收验证 | PPL 正常、接受率归零 |
| **投机解码** | draft 可以单独量化得更激进 | draft 与 target 同精度族，否则 draft 保 BF16 | PPL 正常、接受率为 0% |
| **GQA** | per-channel 和 per-token 可自由选 | K 用 per-channel（比 MHA 下更应如此） | 精度掉点在长上下文才显 |
| **MLA** | K 和 V 可以分别用不同策略 | 统一 per-channel（`c_KV` 的 outlier 在 latent 维） | 短 prompt 正常、32k 崩 |
| **MLA** | 所有 cache 组件都应量化 | `k_R` 保持 FP16（占 11%，风险极高） | 短上下文完全测不出来 |
| **MLA** | per-channel scale 按 absmax 定 | 按 `‖W_UK[:,j]‖ × absmax` 加权 | 掉点小但持续，难以归因 |
| **端侧** | 位宽相同就等价 | `Q4_0` 而非 `Q4_K_M`（NPU 只认简单 scale） | 模型能跑但慢 5 倍 |
| **端侧** | W4A4 比 W4A16 更先进 | W4A16（端侧激活位宽不影响速度） | W4A4 反而更慢 |

最后回到一个实践判断。前六篇教的是"怎么把量化做好"，这一篇要补的是：**做好之后要重新验证一遍**。因为前六篇的验收指标（PPL、逐层重建误差、下游任务分数）全都在"模型自身"这个封闭世界里测——它们测不出树验证的 V/D 比、测不出 32k 上下文的相位误差累积、测不出格式能不能落 NPU。

**量化的验收必须延伸到部署链路的终点**，而不是停在 checkpoint 上。这是这一篇唯一想说的事。

## 5. 系列导航

- 上一篇 **qt6 · QAT 与量化感知训练**：D 路线，STE / PACT / LSQ。本篇的边界三条对它同样适用——QAT 调出来的模型也会撞上投机解码与 MLA。
- **qt5 · KV Cache 量化与 FP4 前沿**：KIVI 的二分法、INT4 与 FP4 的精度分配差异。本篇 §2 是它的结构边界补充。
- **qt4 · QuaRot / SpinQuant / DuQuant**：C 路线旋转，R1–R4 四个位置与"旋转免费性边界"。本篇 §2.6 是 R4 在 MLA 下的重推。
- **qt3 · SmoothQuant 与 AWQ**：B 路线。本篇 §2.5 的 `W_UK` 行范数加权，本质是 AWQ「按重要性分配精度」思想在 MLA 上的重写。
- **qt2 · RTN 与 GPTQ**：A 路线，打包格式与 group size。本篇 §3.2 的 `Q4_K_M` 超块结构是打包格式的延伸。
- **qt1 · 量化的基本语言**：四条路线地图与命名规范。
- **op2 · 模型量化总览**：系列导读与谱系总表。

站内相关：**sd1–sd5（投机解码系列）** 讲了接受率的完整机制与树结构设计，本篇 §1 是它在量化视角的补丁；**fa4（DeepSeek V4）** 的 MLA→NSA 演进与本篇 §2 的 cache 结构直接相关；**fw5** 讲了推理引擎侧的 kernel 选择，本篇 §3.2 的"格式决定通路"是同一逻辑在硬件侧的表现。
