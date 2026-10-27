---
title: '量化手记（六）：QAT 与量化感知训练——STE 的不严谨之处、PACT 与 LSQ 两种修法'
description: 'qt1 的四路线图里，「D 改训练」只画了一格从未展开。但 PTQ 有天花板：权重固定、模型没有任何机会适应量化噪声，W4A4 上必然掉点。QAT 让模型在训练中看见误差，代价是 round 的导数几乎处处为零——于是有了 STE 这个「假装导数为 1」的有偏估计。本文讲清 STE 的三个已知问题，以及 PACT（学截断上界 α）与 LSQ（学 step size s，配 1/√(N·q_max) 梯度缩放）如何分别修补它们，附完整七步流程、可直接跑的 fake-quant 实现，最后回答一个实际问题：QAT 贵 100 倍，什么情况下才值得。'
pubDate: 2026-10-27
series: 量化手记
lang: zh
altLang: en
altHref: /en/blog/qt6-qat-ste-lsq-pact
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

qt1 那张四条路线地图上，A（改舍入）、B（改分布·缩放）、C（改分布·旋转）在 qt2–qt4 里各占一篇，只有 **D「改训练」** 至今只有一格方框。这一篇补上它。

前面五篇有一个共同前提，一直没被点破：**权重是固定的**。RTN 逐元素取整，GPTQ 用 Hessian 把误差摊到别的列，AWQ 搬移分布，QuaRot 把 outlier 摊平——它们都在**已经训练好的那个模型**上做误差最小化，模型自己没有任何机会说"我换个方式表达同一个函数，别在我最敏感的通道上动刀"。

QAT（Quantization-Aware Training）反过来：把量化算子塞进训练图，让梯度穿过它，模型在几万步里学会**在格子里生活**。代价是遇到了一个数学上很尴尬的障碍——

```text
round(x) 的导数，几乎处处为 0。
照实求导，梯度全零，网络一动不动。
```

于是整个 QAT 都建立在一个"谎言"上：**假装 round 的导数是 1**。这个技巧叫 STE（Straight-Through Estimator，直通估计器），1993 年就有了，至今仍是所有低比特训练的基础。它**有偏**、**不严谨**、**已知会引发三种具体故障**——而 PACT 和 LSQ 本质上都是在修补这三种故障中的某几个。

<div class="keybox">
<strong>本篇位置：</strong>qt1 四路线图里 D 路线的展开。qt2–qt5 全是 PTQ，本篇是唯一"权重会动"的一篇。<strong>核心判断提前给：</strong>W8 不需要 QAT（PTQ 已近乎无损）；W4A16 上 AWQ/GPTQ 够好，QAT 边际收益低；<strong>只有 W4A4 且不能接受 PTQ 落差时，QAT 是唯一选择</strong>——但它也是绝大多数端侧发布版 int4 模型（Gemma 3 QAT、Llama 3.2 1B/3B）的实际生产方式。
</div>

## 1. 先看清 PTQ 的天花板在哪

不谈玄学，先看一条曲线。同一个 LLaMA-2-7B，四条路线的困惑度（PPL）随位宽下探的表现：

<div class="fig">
<svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="四条路线在不同位宽下的困惑度增量">
  <text x="20" y="22" font-size="14" font-weight="700" fill="#1a1a1a">位宽越低，PTQ 与 QAT 的差距越大——4 bit 是分水岭，2 bit 是断崖</text>
  <text x="20" y="40" font-size="11" fill="#6b7280">纵轴 = 困惑度增量（ΔPPL，越低越好）；量级参考，非精确复现值</text>

  <!-- 坐标系 -->
  <line x1="76" y1="70" x2="76" y2="272" stroke="#9ca3af" stroke-width="1.2"/>
  <line x1="76" y1="272" x2="630" y2="272" stroke="#9ca3af" stroke-width="1.2"/>

  <!-- y 轴刻度 -->
  <g font-size="10" fill="#6b7280" text-anchor="end">
    <text x="68" y="76">+2.0</text>
    <text x="68" y="126">+1.5</text>
    <text x="68" y="176">+1.0</text>
    <text x="68" y="226">+0.5</text>
    <text x="68" y="275">0</text>
  </g>
  <g stroke="#e5e7eb" stroke-width="1">
    <line x1="76" y1="70" x2="630" y2="70"/>
    <line x1="76" y1="120" x2="630" y2="120"/>
    <line x1="76" y1="170" x2="630" y2="170"/>
    <line x1="76" y1="220" x2="630" y2="220"/>
  </g>

  <!-- x 轴刻度：8 / 6 / 4 / 2 bit -->
  <g font-size="11" fill="#374151" text-anchor="middle">
    <text x="140" y="292">W8</text>
    <text x="300" y="292">W6</text>
    <text x="460" y="292">W4</text>
    <text x="600" y="292">W2</text>
  </g>
  <text x="360" y="312" font-size="11" fill="#6b7280" text-anchor="middle">权重位宽（激活同降）</text>

  <!-- ① RTN：最差 -->
  <polyline points="140,262 300,232 460,120 600,32" fill="none" stroke="#dc2626" stroke-width="2.4"/>
  <circle cx="140" cy="262" r="3.6" fill="#dc2626"/><circle cx="300" cy="232" r="3.6" fill="#dc2626"/>
  <circle cx="460" cy="120" r="3.6" fill="#dc2626"/><circle cx="600" cy="32" r="3.6" fill="#dc2626"/>

  <!-- ② GPTQ / AWQ -->
  <polyline points="140,266 300,248 460,168 600,74" fill="none" stroke="#ea580c" stroke-width="2.4" stroke-dasharray="6 4"/>
  <circle cx="140" cy="266" r="3.6" fill="#ea580c"/><circle cx="300" cy="248" r="3.6" fill="#ea580c"/>
  <circle cx="460" cy="168" r="3.6" fill="#ea580c"/><circle cx="600" cy="74" r="3.6" fill="#ea580c"/>

  <!-- ③ 旋转 QuaRot -->
  <polyline points="140,267 300,256 460,196 600,110" fill="none" stroke="#2563eb" stroke-width="2.4" stroke-dasharray="2 3"/>
  <circle cx="140" cy="267" r="3.6" fill="#2563eb"/><circle cx="300" cy="256" r="3.6" fill="#2563eb"/>
  <circle cx="460" cy="196" r="3.6" fill="#2563eb"/><circle cx="600" cy="110" r="3.6" fill="#2563eb"/>

  <!-- ④ QAT：最优 -->
  <polyline points="140,269 300,264 460,244 600,190" fill="none" stroke="#15803d" stroke-width="3"/>
  <circle cx="140" cy="269" r="4.2" fill="#15803d"/><circle cx="300" cy="264" r="4.2" fill="#15803d"/>
  <circle cx="460" cy="244" r="4.2" fill="#15803d"/><circle cx="600" cy="190" r="4.2" fill="#15803d"/>

  <!-- 4bit 处的高亮竖线 -->
  <line x1="460" y1="70" x2="460" y2="272" stroke="#9ca3af" stroke-width="1" stroke-dasharray="3 3"/>
  <rect x="380" y="52" width="160" height="18" rx="3" fill="#fef3c7"/>
  <text x="390" y="65" font-size="10" fill="#92400e" font-weight="700">4 bit 处分歧最大：1.0 vs 0.3</text>

  <!-- 图例 -->
  <g font-size="11">
    <line x1="76" y1="322" x2="102" y2="322" stroke="#15803d" stroke-width="3"/>
    <text x="108" y="325" fill="#15803d" font-weight="700">QAT（模型会动）</text>
    <line x1="238" y1="322" x2="264" y2="322" stroke="#2563eb" stroke-width="2.4" stroke-dasharray="2 3"/>
    <text x="270" y="325" fill="#2563eb">QuaRot（旋转）</text>
    <line x1="392" y1="322" x2="418" y2="322" stroke="#ea580c" stroke-width="2.4" stroke-dasharray="6 4"/>
    <text x="424" y="325" fill="#ea580c">GPTQ / AWQ</text>
    <line x1="536" y1="322" x2="562" y2="322" stroke="#dc2626" stroke-width="2.4"/>
    <text x="568" y="325" fill="#dc2626">RTN</text>
  </g>
</svg>
<p class="cap">图 1 · 四条路线随位宽下探的表现。W8 上四条线基本重合（PTQ 已足够），分水岭在 W4，断崖在 W2。QAT 的曲线最平——因为它不是"更好地近似原模型"，而是"训练出一个适合格子的新模型"。</p>
</div>

读这张图要注意三件事：

**① W8 完全不值得 QAT。** PTQ 在 W8 上增量在 0.01–0.05 量级，落在评测噪声里。花 100 倍算力换一个测不出来的提升，是纯粹的浪费。

**② W4A16 上 QAT 的边际收益低。** AWQ / GPTQ 已经把增量压到 0.1–0.3，QAT 大概能再降一半，但代价是从"几分钟校准"变成"几小时到几天训练"。只有在**精度预算极紧**（比如端侧模型、要跟 FP16 版做 AB 对比、用户对质量零容忍）时才回本。

**③ W4A4 是 QAT 的主战场。** qt4 讲过，W4A4 上 QuaRot 把 LLaMA-2-7B 从 5.47 拉到 6.10（增量 0.63），SpinQuant 到 5.86、DuQuant 到 5.98。而 QAT 在 W4A4 上能压到 0.1–0.2 量级——**这是 PTQ 全线（A/B/C 三条路线加起来）都做不到的区间**。

为什么？因为 PTQ 三条路线本质上都在做同一件事：**在不动模型函数的前提下，找一组更好的 (s, z) 和更好的舍入决策**。而 4 bit 只有 15 个格子（对称 int4），表达能力就那么点，你怎么分配分辨率都会撞到墙。QAT 拆的是另一堵墙——它允许模型**改变自己的权重数值分布**，让"重要的事"用更少的信息表达出来。这是自由度上的差别，不是调参上的差别。

## 2. STE：整个 QAT 的基石，也是最不严谨的一步

### 2.1 问题：round 的导数不存在

一个最朴素的量化函数，前向是这样：

```python
def quantize_fwd(x, s, qmax):
    """对称量化，s 是 step size，qmax = 2**(b-1) - 1。"""
    q = torch.clamp(torch.round(x / s), -qmax - 1, qmax)   # ① 缩放 → ② 取整 → ③ 截断
    return q * s                                            # ④ 反量化
```

`round` 是这样的阶梯函数：

```text
round(x)
  3 ┤        ┌───────
  2 ┤    ┌───┘
  1 ┤┌───┘
  0 ┼┘
    └────┬────┬────┬───→ x
       0.5  1.5  2.5
```

除了在 0.5、1.5、2.5 这些点上导数不存在（或者说是一个冲激），**其余每一点的导数都是 0**。PyTorch 的实现里 `round` 的 backward 直接返回 0。

这意味着什么？意味着反向传播到了 `round` 这里，梯度全部被清零——**前面的所有参数（包括 s、包括上游的权重）都收不到任何信号**。QAT 的第一步就卡死了。

### 2.2 STE 的做法：假装导数等于 1

既然真实的导数是 0，那我们就不用它。**把 round 在反向传播时当成恒等函数**：

```python
class STERound(torch.autograd.Function):
    """前向 = round，反向 = 直通（梯度原样传回）。"""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 真实梯度几乎处处为 0；这里直接返回 grad_output，等价于 ∂round/∂x := 1
        return grad_output
```

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="STE 的前向与反向">
  <text x="20" y="22" font-size="14" font-weight="700" fill="#1a1a1a">STE：前向走真实的阶梯，反向走一条笔直的假路</text>

  <!-- 左：前向 -->
  <text x="40" y="52" font-size="12" font-weight="700" fill="#1d4ed8">前向（forward）：真实计算，round 被如实执行</text>
  <g>
    <rect x="40" y="66" width="96" height="34" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="88" y="88" font-size="12" fill="#1e40af" text-anchor="middle">x / s</text>
    <line x1="136" y1="83" x2="170" y2="83" stroke="#6b7280" stroke-width="1.6" marker-end="url(#ar1)"/>

    <rect x="172" y="66" width="96" height="34" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="220" y="88" font-size="12" fill="#1e40af" text-anchor="middle">round(·)</text>
    <line x1="268" y1="83" x2="302" y2="83" stroke="#6b7280" stroke-width="1.6" marker-end="url(#ar1)"/>

    <rect x="304" y="66" width="96" height="34" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="352" y="88" font-size="12" fill="#1e40af" text-anchor="middle">clip(·)</text>
    <line x1="400" y1="83" x2="434" y2="83" stroke="#6b7280" stroke-width="1.6" marker-end="url(#ar1)"/>

    <rect x="436" y="66" width="96" height="34" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="484" y="88" font-size="12" fill="#1e40af" text-anchor="middle">× s</text>
    <line x1="532" y1="83" x2="566" y2="83" stroke="#6b7280" stroke-width="1.6" marker-end="url(#ar1)"/>
    <text x="600" y="88" font-size="12" fill="#1e40af" font-weight="700">ŷ</text>
  </g>

  <!-- 分隔 -->
  <line x1="40" y1="122" x2="640" y2="122" stroke="#e5e7eb" stroke-width="1.5"/>

  <!-- 右：反向 -->
  <text x="40" y="148" font-size="12" font-weight="700" fill="#b91c1c">反向（backward）：绕过 round，梯度直通</text>
  <g>
    <text x="40" y="188" font-size="12" fill="#b91c1c" font-weight="700">∂L/∂ŷ</text>
    <line x1="104" y1="184" x2="176" y2="184" stroke="#b91c1c" stroke-width="1.6" marker-end="url(#ar2)"/>
    <text x="140" y="176" font-size="10" fill="#6b7280" text-anchor="middle">× s</text>

    <line x1="176" y1="184" x2="248" y2="184" stroke="#b91c1c" stroke-width="1.6" marker-end="url(#ar2)"/>
    <text x="212" y="176" font-size="10" fill="#6b7280" text-anchor="middle">clip′</text>

    <!-- 被绕过的 round -->
    <rect x="250" y="166" width="96" height="34" rx="5" fill="none" stroke="#d1d5db" stroke-width="1.4" stroke-dasharray="4 3"/>
    <text x="298" y="188" font-size="12" fill="#9ca3af" text-anchor="middle" text-decoration="line-through">round</text>
    <path d="M250 190 Q298 226 346 190" fill="none" stroke="#b91c1c" stroke-width="2.2"/>
    <text x="298" y="246" font-size="10" fill="#b91c1c" text-anchor="middle" font-weight="700">梯度绕行（导数为 0，跳过它）</text>

    <line x1="348" y1="184" x2="420" y2="184" stroke="#b91c1c" stroke-width="1.6" marker-end="url(#ar2)"/>
    <text x="384" y="176" font-size="10" fill="#6b7280" text-anchor="middle">÷ s</text>
    <text x="444" y="188" font-size="12" fill="#b91c1c" font-weight="700">∂L/∂x</text>
  </g>

  <rect x="490" y="150" width="150" height="72" rx="6" fill="#fef2f2" stroke="#fca5a5"/>
  <text x="502" y="170" font-size="10" fill="#991b1b" font-weight="700">一句话定义</text>
  <text x="502" y="188" font-size="10" fill="#991b1b">∂round(x)/∂x := 1</text>
  <text x="502" y="204" font-size="10" fill="#991b1b">有偏，但够用</text>

  <defs>
    <marker id="ar1" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 z" fill="#6b7280"/>
    </marker>
    <marker id="ar2" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 z" fill="#b91c1c"/>
    </marker>
  </defs>
</svg>
<p class="cap">图 2 · STE 的前向与反向。前向是诚实的：`round` 真的把值取整了，量化误差真实存在。反向是虚构的：假装那条阶梯是斜率为 1 的直线。整个 QAT 的可行性都建立在这个虚构上。</p>
</div>

**为什么这个"谎言"居然有效？** 直觉是：`round(x/s)` 在局部（一个格子的宽度内）确实与 `x/s` 只差一个常数偏移，而**我们希望权重朝"降低 loss"的方向移动，移动量级小于一个格子时，取整行为不变但输出的连续近似在变**。STE 给出的梯度方向在统计意义上是对的——它相当于在优化一个"把量化噪声当作固定偏移"的平滑代理目标。所以训练能收敛，收敛到的解还不错。这是**工程上极其成功的启发式**，不是数学上正确的推导。

### 2.3 STE 的三个已知问题

不严谨就得付代价。STE 的失败模式非常具体，而且每一个都对应后面某一节的修法：

**问题一：饱和区梯度不该有，但它有。**

`clip(x/s, -qmax, qmax)` 之后，如果 `x/s > qmax`（也就是这个元素超出了量化范围），输出是常数 `qmax`，对 x 的真实导数是 **0**——改它没用，反正都被截断。但 STE 的 backward 直接把 `∂L/∂ŷ` 传回去，等价于告诉优化器"这个元素还在正常参与计算，放心调"。

后果是具体的：**outlier 元素持续收到"往更大走"或"往更小走"的梯度，但 loss 并不因它改变**。于是这些元素要么被推得更极端（加剧 outlier），要么在错误的信号下漂移。而 outlier 恰恰是量化最敏感的东西（qt3 整篇都在讲它）。这是 PACT 要解决的第一个问题。

**问题二：梯度幅值不随 s 变化，学习率无法跨位宽复用。**

看反向路径：梯度从 `∂L/∂ŷ` 出发，乘以 s、跳过 round、除以 s——两个 s 相互抵消了。也就是说 **s 的梯度（对 s 求导的那一项）在朴素 STE 下拿不到正确的量级信息**。更麻烦的是，当位宽从 8 降到 4，同样的 s 相对变化对结果的影响大得多，但 STE 给出的梯度幅值不变，于是 4 bit 下就必须重新调学习率。这是 LSQ 要解决的核心问题。

**问题三：优化器状态建立在错误的梯度上。**

Adam 维护一阶矩和二阶矩。如果 `m` 和 `v` 累积的是 STE 的伪梯度而非真实梯度，那么自适应学习率的缩放因子 `1/√v` 就失去了它原本的统计意义。实践中这会导致 QAT 前期 loss 抖动明显大于普通训练，且对 `eps` 敏感。

<div class="warnbox">
<strong>一个常见的误判：</strong>QAT 训练曲线正常、loss 平稳下降，就认为量化没问题了。<strong>不对。</strong>STE 的有偏性会让模型学会"在训练集分布上绕过误差"，但这种绕过往往不泛化。站内 qt2 提过 GPTQ 的验收要看 MMLU / GSM8K 而不只是 PPL；QAT 更甚——<strong>PPL 几乎不掉但 GSM8K 掉 8 个点</strong>是 QAT 最典型的失败形态，因为多步推理对每一步的小误差累积最敏感。验收必须跑任务级指标。
</div>

## 3. PACT：把截断上界变成可学习参数

PACT（PArameterized Clipping acTivation，2018）的出发点很直接：既然 STE 在饱和区的梯度是错的，那就**让"饱和区在哪"本身变成可学的**。

### 3.1 做法

对激活（ReLU 之后的非负分布），引入一个可学习参数 α 作为截断上界：

```python
class PACT(nn.Module):
    """参数化截断。α 可学习，初始化后由梯度自己找到合适的截断点。"""

    def __init__(self, init_alpha=8.0, qmax=7):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.qmax = qmax

    def forward(self, x):
        # 前向：先截断到 [0, α]，再按 α 定 scale 做量化
        alpha = self.alpha.clamp(min=1e-6)          # 防止 α 塌到 0
        x_clip = torch.clamp(x, 0.0, alpha)
        s = alpha / self.qmax                       # scale 由 α 决定，不是由 absmax 决定
        q = torch.clamp(torch.round(x_clip / s), 0, self.qmax)
        return q * s
```

关键在 `s = α / qmax` 这一行：**scale 不再是 absmax 的函数，而是 α 的函数**。于是 α 一动，整个量化网格跟着动。

### 3.2 α 的梯度为什么只在被截断的样本上非零

对 α 求导：

```text
y = clip(x, 0, α)

∂y/∂α = 1   如果 x > α   （被截断的样本，"顶上"贴着 α）
      = 0   如果 x ≤ α   （未截断的样本，输出与 α 无关）
```

**只有被截断的样本能拉动 α。** 这个性质非常优雅，它让 α 的更新自动有了正确的语义：

- 如果 α 设得太小，大量样本被截断，它们持续把 α 往上推 → α 增大；
- 如果 α 设得太大，几乎没有样本被截断，α 收不到梯度 → 停住，不再膨胀；
- 平衡点是 loss 自己算出来的：**截断更多样本（牺牲分辨率换动态范围）与保留更多分辨率（牺牲动态范围）之间的最优权衡。**

对比一下 absmax 的做法：absmax 是**被单个 outlier 唯一决定的**（qt2 讲过，一个 118 的 outlier 会把 127 这个格子全部占掉）。PACT 把它换成了"由 loss 决定的统计折中点"——outlier 要么被切掉（损失一点信息，但换来全体的分辨率），要么把 α 拉大（保住它，代价是全体格子变粗）。**这个决策交给优化器，比交给 max() 合理得多。**

### 3.3 PACT 的实际效果与局限

PACT 在视觉模型（MobileNet、ResNet 系列）的 W4A4 上是标配，能显著好于 absmax。搬到 LLM 上有两个调整：

**① 只对激活用，不对权重用。** 权重的分布是对称的、厚重的、没有 ReLU 那种单边长尾，PACT 的单边上界语义不适用。对称量化用 LSQ（下一节）。

**② SwiGLU 的输出不是非负的。** LLaMA 系的 FFN 用 SwiGLU，中间激活有正有负。PACT 原本的 `clip(x, 0, α)` 要改成对称形式 `clip(x, -α, α)`，或者干脆用 LSQ 的对称参数化。

## 4. LSQ：学 step size，以及那个 1/√(N·q_max)

LSQ（Learned Step Size Quantization，ICLR 2020）是 LLM 量化里更常用的那个。它不学 α，直接学 **s**，核心贡献是给出了一个**梯度缩放因子**，让 s 的梯度在不同位宽、不同层规模下都落在合理量级。

### 4.1 s 的梯度推导

前向：`v̂ = s · round(clip(v/s, -Q_N, Q_P))`。对 s 求导，把 `round` 按 STE 处理：

```text
∂v̂/∂s = round(v/s) − v/s        （当 −Q_N < v/s < Q_P）
       = −Q_N                   （当 v/s ≤ −Q_N，下饱和）
       = Q_P                    （当 v/s ≥ Q_P，上饱和）
```

注意中间那一项 `round(v/s) − v/s`——它正好是**归一化舍入误差的相反数**，取值范围 `[-0.5, 0.5]`。这个结构很有解释力：**s 的梯度正比于"这个元素被舍入偏了多少"**。偏得多，说明这个 s 不合适，该调；偏得少，就不用动。

### 4.2 然后乘上一个不起眼但至关重要的因子

LSQ 论文在这里做了一件事：把上面这个梯度再乘一个缩放因子

```text
g = 1 / sqrt(N · Q_P)
```

`N` 是该张量的元素数（权重按整个层算，激活按 per-token 算），`Q_P` 是正向上限（int4 对称量化时是 7）。两个因子的意图各不相同，值得拆开看：

**`1/√N`：让 s 的梯度与层规模无关。**

s 是一个**标量**，但它影响 N 个元素的输出。所以 `∂L/∂s = Σ_i (∂L/∂v̂_i)(∂v̂_i/∂s)`，是 N 项之和。如果这 N 项的符号随机（舍入误差正负各半，确实如此），那么和的量级按 **√N** 增长而不是 N。不同层宽度差几十倍（hidden 4096 vs intermediate 11008 vs 词表 32000），不除这个因子，同一套学习率必然在某一层爆炸、在另一层停滞。`1/√N` 把这个尺度拉平了。

**`1/√Q_P`：让 s 的梯度随位宽降低而增大。**

位宽越低，`Q_P` 越小。除以 `√Q_P` 意味着**低位宽时 s 的梯度更大**。这符合直觉：4 bit 下 s 稍微不合适，相对精度损失就远大于 8 bit，所以 s 应该动得更积极。这个因子也是 LSQ 能在 2/3/4/8 bit 上**共用同一套超参**的直接原因——论文里的消融明确显示，去掉它以后每个位宽都要重新调学习率。

### 4.3 初始化：为什么不用 absmax

```text
s₀ = 2 · mean(|v|) / sqrt(Q_P)
```

用均值而不是最大值。原因和 PACT 那节同源：absmax 被单个 outlier 绑架，而 `mean(|v|)` 是分布主体位置的估计，对 outlier 不敏感。乘 2 是个经验性的补偿（`mean` 小于典型步长的合理值），`/sqrt(Q_P)` 保持与梯度缩放因子的一致性。

代价是：如果分布真的有长尾，这个初始化会让初始量化"切掉"尾部，前期 loss 会跳一下。但既然 s 是可学的，它会在几百步内自己找到位置——**从一个"主体合理"的点出发，比从一个"被 outlier 拖歪"的点出发要好。**

<div class="keybox">
<strong>PACT vs LSQ 一句话区分：</strong><br>
<strong>PACT</strong> 学<strong>截断上界 α</strong>，scale 由 α 导出（`s = α/qmax`）。适合<strong>非负激活</strong>（ReLU 后），语义是"截到哪里"。<br>
<strong>LSQ</strong> 直接学 <strong>step size s</strong>，配 `1/√(N·q_max)` 梯度缩放。适合<strong>对称分布</strong>（权重、SwiGLU 激活），语义是"格子多宽"。<br>
实践中 LLM 的 QAT 多是 LSQ 变体（因为权重对称、激活也无 ReLU），PACT 更多出现在视觉模型与早期工作中。
</div>

## 5. 完整的七步流程

原理讲完，落到工程。一个能跑通的 LLM QAT 流程是这样：

```python
import torch, torch.nn as nn
from torch.optim import AdamW

# ── 步骤 1a：先定义 STE 取整算子（全篇唯一一处「作弊」）────────────
class _STERound(torch.autograd.Function):
    """前向 round + 饱和截断，反向直通。"""

    @staticmethod
    def forward(ctx, x, qmax):
        return torch.round(x).clamp(-qmax - 1, qmax)

    @staticmethod
    def backward(ctx, g):
        return g, None          # 对 x 直通；qmax 是常数，无梯度


# ── 步骤 1b：构建 fake-quant 模块，替换 nn.Linear ─────────────────
class QATLinear(nn.Module):
    """权重用 LSQ 学 s，激活用 per-token 动态对称量化。"""

    def __init__(self, base: nn.Linear, w_bits=4, a_bits=8, group_size=128):
        super().__init__()
        self.base = base
        self.w_bits, self.a_bits, self.g = w_bits, a_bits, group_size
        self.wqmax = 2 ** (w_bits - 1) - 1
        self.aqmax = 2 ** (a_bits - 1) - 1

        out_f, in_f = base.weight.shape
        n_groups = in_f // group_size
        # 按 group 学 s：形状 [out_f, n_groups]，初始化用 2·mean(|w|)/√qmax
        w = base.weight.data.view(out_f, n_groups, group_size)
        s0 = 2.0 * w.abs().mean(dim=2) / (self.wqmax ** 0.5)
        self.w_scale = nn.Parameter(s0)

    def forward(self, x):
        # ── 权重侧：per-group LSQ ──
        w = self.base.weight
        out_f, in_f = w.shape
        s = self.w_scale.clamp(min=1e-8).unsqueeze(-1)          # [out_f, n_groups, 1]

        w_scaled = w.view(out_f, -1, self.g) / s                 # 缩放
        w_q = _STERound.apply(w_scaled, self.wqmax)              # STE 取整 + 截断
        w_dq = (w_q * s).view(out_f, in_f)                       # 反量化回原形状

        # ── 激活侧：per-token 动态，不需要学（absmax 运行时就有）──
        a_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.aqmax
        x_q = _STERound.apply(x / a_scale, self.aqmax)
        x_dq = x_q * a_scale

        return nn.functional.linear(x_dq, w_dq, self.base.bias)


# ── 步骤 2：只训 s（其余冻结），还是全参微调 ──────────────────────
# 两种策略，选择很关键：
#   (a) 只学 scale：几百步、单卡、几乎不掉精度 —— 95% 的场景够用
#   (b) 全参微调：几万步、多卡，能多救 0.1–0.3 PPL —— 端侧发布版才这么做
for name, p in model.named_parameters():
    p.requires_grad = ('w_scale' in name)     # 策略 (a)

# ── 步骤 3：优化器与学习率 ──────────────────────────────────────
# s 的梯度已经带了 1/√(N·qmax) 缩放，所以学习率可以跨位宽复用
opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, betas=(0.9, 0.99))

# ── 步骤 4：训练循环（用一小份校准+训练数据，1e8~1e9 token 量级）──
for step, batch in enumerate(loader):
    out = model(**batch)
    # 关键：QAT 的 loss 就是普通 LM loss，额外加一项 KD 会更稳
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # STE 梯度有偏，必须裁
    opt.step(); opt.zero_grad()

# ── 步骤 5：定期"真量化"评估（不能只看 fake-quant 的 loss）────────
# fake-quant 的 s 是 float，真部署时 s 要存成 fp16 → 多一次舍入
# 所以每 N 步要用真实的 int4 kernel 跑一遍，看 PPL 与任务指标

# ── 步骤 6：导出 ────────────────────────────────────────────────
# s 转 fp16 存盘；w_q 打包成 int32（每 8 个 int4 挤一个 int32，qt2 讲过打包格式）
# 注意：QAT 出来的 s 分布与 PTQ 不同，不要用 AWQ/GPTQ 的导出脚本硬套

# ── 步骤 7：验收（这一节最重要）─────────────────────────────────
# 必测：PPL（wiki-text）+ MMLU + GSM8K + HumanEval + 32k needle
# 红线：GSM8K 掉超过 2 个点就回退——这是 QAT 最常见的隐性失败
```

注意这两处细节，都是刻意的：

- **`w_scaled` 里 s 用 `.unsqueeze(-1)` 广播到 group 最后一维**，因为 LSQ 的 s 是逐 group 的标量，不是逐元素的。
- **`backward` 必须返回两个值**：`g` 给 `x`（直通），`None` 给 `qmax`（Python 整数不进计算图）。少写一个会直接报 grad 数量不匹配。

### 5.1 两个容易踩的坑

**坑一：fold BN / norm 的时机。** 视觉模型做 QAT 前必须先把 BN 折进卷积（否则量化后的 BN 参数会破坏 scale）。LLM 没有 BN，但有 RMSNorm——**RMSNorm 不折进线性层**（它作用在激活上，不是仿射变换），所以这一步在 LLM 上不存在。真正要注意的是**不要把 LayerNorm 的 γ 吸收进量化 scale**：γ 是逐通道的，会和 per-token 的激活 scale 打架。让 γ 保持 FP16，只量化它后面的线性层输入。

**坑二：激活的 per-token scale 不要学。** 权重的 s 是静态的（部署时存盘），所以必须学；激活的 s 是运行时按 token 现算的（`absmax(x)/qmax` 一行就出来），**让它可学反而有害**——训练时会漂到一个对训练分布最优但泛化更差的固定值，而且部署时你根本没有"学好的那个值"可用（每个 token 不一样）。这是 qt5 讲 KV 量化时同一个道理。

### 5.2 QAT 与 PTQ 在验收上的根本差别

PTQ 的验收相对简单：给定一组校准数据，跑一遍 PPL + 几个任务，好坏立判，而且**可复现**（同样的 checkpoint、同样的校准集，结果一样）。

QAT 的验收麻烦得多：

| 维度 | PTQ | QAT |
|---|---|---|
| 可复现性 | 完全可复现 | 受随机种子、数据顺序影响 |
| 验收成本 | 分钟级 | 小时级（要跑完整评测） |
| 失败形态 | PPL 明显上升 | **PPL 正常、任务指标掉**（隐性） |
| 过拟合风险 | 校准集过拟合 | 训练集过拟合 + 校准集过拟合 |
| 回退成本 | 换一组参数重跑 | 重训（几小时到几天） |

**最危险的是"失败形态"那一行。** QAT 训练时模型看到的是量化后的前向，所以它在训练分布上的 loss 会降得很好看；但 STE 的梯度有偏意味着它学到的是"绕过量化的技巧"而非"适应量化"。这些技巧换到 GSM8K 那种需要多步精确推理的任务上就露馅了。所以我把它写成红线：**GSM8K 掉超 2 点就回退**。

## 6. 取舍：QAT 贵 100 倍，什么时候值得

一张决策表，按场景给结论：

| 场景 | 建议 | 理由 |
|---|---|---|
| **W8A8 部署** | **不要 QAT** | PTQ 增量 0.01–0.05，落在噪声里，QAT 是浪费 |
| **W4A16，通用服务** | **AWQ 或 GPTQ** | 增量 0.1–0.3 已可接受，PTQ 几分钟搞定 |
| **W4A16，精度预算极紧** | QAT（仅学 scale） | 只学 scale 只需要几百步单卡，性价比尚可 |
| **W4A4，有 PTQ 落差不能接受** | **QAT 是唯一选择** | A/B/C 三条 PTQ 路线加起来也到不了这个区间 |
| **W2 / 三值 / 二值** | **必须 QAT** | PTQ 在这个位宽上直接崩，没有讨论空间 |
| **端侧发布版 int4 模型** | **QAT** | Gemma 3 QAT、Llama 3.2 1B/3B 都是这么来的；模型小、训练便宜、用户对质量零容忍 |
| **需要频繁换模型** | 不要 QAT | 每次换模型重训一遍，工程上不可持续 |

<div class="keybox">
<strong>一个常被忽略的成本项：</strong>QAT 的真正成本不是 GPU 小时，而是<strong>它把量化从"部署阶段的一个步骤"变成了"训练流程的一部分"</strong>。PTQ 可以在模型发布后、由部署方独立完成（这也是 AWQ/GPTQ 生态繁荣的原因）；QAT 必须由模型作者在训练阶段插入，且量化配置（位宽、group size、哪些层不量化）要在训练前就定死——<strong>改一个位宽就得重训</strong>。这是组织成本，不是算力成本。
</div>

所以我对 QAT 的判断是：**它是量化方法谱系里的"重武器"，不是"更好的默认选项"**。qt1 那张四路线图之所以把 D 放在最后，不是因为它弱，而是因为它的使用门槛最高——一旦用了，整条链路（训练、导出、验收、迭代）都得跟着改。而 A/B/C 三条 PTQ 路线可以随时插拔、随时回退，这是它们统治了绝大多数生产场景的真正原因。

**但反过来说**：如果你的目标是 W4A4，或者你在做端侧发布，那前面五篇讲的 PTQ 技巧都只是"把损失降到最小"，而 QAT 是"重新训练一个适合 4 bit 的模型"。这两件事不在一个量级上。

## 7. 系列导航

- 上一篇 **qt5 · KV Cache 量化与 FP4 前沿**：正交的压缩维度，以及格式如何改写算法。
- **qt4 · QuaRot / SpinQuant / DuQuant**：C 路线，旋转做 W4A4——本篇反复对比的那条基准线。
- **qt3 · SmoothQuant 与 AWQ**：B 路线，激活 outlier 的两条解法。PACT 与它们是同一个问题的不同代际回答。
- **qt2 · RTN 与 GPTQ**：A 路线，一切方法的基线；打包格式与验收口径在本篇被复用。
- **qt1 · 量化的基本语言**：四条路线地图——本篇是 D 路线的展开。
- **op2 · 模型量化总览**：系列导读与谱系总表。

站内相关：**qt7（量化的三个边界）** 会接着本篇追问——QAT 调出来的模型，在与投机解码叠加、换成 MLA 结构、跑到端侧 NPU 上之后，还成立吗；**fa4（DeepSeek V4）** 提到的量化训练配方与本篇的 STE/KD 组合有关；**fw5** 讲了推理引擎侧的 kernel 选择，QAT 导出的权重能否被高效执行取决于它。
