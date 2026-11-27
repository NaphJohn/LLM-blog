---
title: 前沿架构解码手记（十一）：LingBot-VLA 2.0——MoE 只在动作专家里，一场「宽度守恒」的容量游戏
description: "把 LingBot-VLA 2.0 的配置类与 real_robot.yaml 逐字段读一遍：6.38B 总参、Qwen3-VL-4B 稠密主干 + 768 宽稀疏动作专家，36 层逐层联合注意力。核心发现是一条宽度恒等式——32 个 512 宽专家取 Top-4 加 704 宽共享专家，恰好等于原稠密 FFN 的 2752，算力不变而 FFN 容量放大 6.2 倍。另拆 Dual-Query 蒸馏（实为 8+8 个 token 而非两个）、chunk 50 × 55 维、10 步流匹配，以及真机只执行前 25 步的重规划。"
pubDate: 2026-11-27
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa11-lingbot-vla2
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么看 LingBot-VLA 2.0

前沿架构手记写到第十一篇，一直在 LLM 里打转——MoE、稀疏注意力、线性注意力、残差连接。这次换个战场：**具身智能的 VLA**。

选 LingBot-VLA 2.0（蚂蚁灵波，arXiv:2607.06403）有两个理由。一是它把 LLM 侧最成熟的效率工具——**稀疏 MoE**——完整搬进了动作专家这一侧；二是它开源得足够彻底，`configs/vla/real_robot/real_robot.yaml` 里连 `routed_scaling_factor`、`router_z_loss_coeff` 这种旋钮都摊着，可以逐字段读，不用猜。

它和 VLA 系列的 π 家族（vla3）是同一条技术血脉：**稠密 VLM 主干 + 窄动作专家 + Flow Matching**。差别在于 π0 的动作专家是纯稠密 FFN，LingBot-VLA 2.0 把它换成了 32 专家的稀疏 MoE。

本文只回答一个问题：**这个 MoE 具体放在哪儿，以及它为什么这么设计。**

<div class="keybox">
<strong>先给结论：</strong>MoE 不在 VLM 主干里，只在动作专家的 FFN 里，36 层全覆盖；专家的宽度不是随手取的——<strong>Top-4 激活（4 × 512）+ 共享专家（704）= 2752，恰等于被替换掉的那个稠密 FFN 的 intermediate_size</strong>。这不是巧合，是「锁算力、放容量」的标准做法。
</div>

## 1. 整体形态：4B 稠密主干，挂一个 768 宽的动作专家

先把零件的尺寸摆出来。以下数字全部来自 HF 模型卡的 safetensors 精确计数与仓库配置，不是估算：

| 组件 | 规格 | 出处 |
|---|---|---|
| 总参数 | **6,375,907,511（≈6.38B）**，FP32 存储 | HF `robbyant/lingbot-vla-v2-6b` |
| VLM 主干 | Qwen3-VL-4B-Instruct，**36 层稠密** | README + `tokenizer_path` |
| 主干宽度 | hidden **2560**，32 Q / 8 KV，head_dim 128，FFN 9728 | Qwen3-VL-4B `config.json` |
| 动作专家 | Qwen2 架构，**36 层**，hidden **768** | `modeling_lingbot_vla_v2.py` `num_layers = 36` |
| 专家注意力 | 16 Q / 2 KV（GQA 8:1），head_dim 128 | `action_num_attention_heads: 16`、`action_num_key_value_heads: 2` |
| 动作 / 状态维度 | **55**（见第 7 节分解） | `action_dim: 55`、`max_state_dim: 55` |
| Action chunk | **50** 步 | `configuration_lingbot_vla.py` `chunk_size = 50` |
| 去噪步数 | **10** 步 Flow Matching | 同上 `self.num_steps = 10` |
| 推理延迟 | 约 **130 ms** / chunk（RTX 4090D） | README |

两塔宽度差距很大——2560 对 768，相差 3.3 倍。这个比例决定了后面所有设计的走向。

<div class="fig">
<svg viewBox="0 0 680 806" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="LingBot-VLA 2.0 完整数据流：左侧 Qwen3-VL-4B 稠密主干处理图像语言前缀，右侧 768 宽动作专家处理状态与加噪动作后缀，36 层逐层联合注意力，最后经动作输出投影生成速度场" id="fig-lb1">
  <style>
    #fig-lb1{--t1:#1a1a1a;--t2:#5F5E5A;--ln:#96918A;
      --vb:#E6F1FB;--vs:#185FA5;--vt:#0C447C;
      --gb:#E1F5EE;--gs:#1D9E75;--gt:#04342C;
      --nb:#F1EFE8;--ns:#5F5E5A;--nt:#444441;
      --ab:#FAEEDA;--as:#854F0B;--at:#633806;
      --pb:#FBEAF0;--ps:#993556;--pt:#72243E;}
    :root[data-theme="dark"] #fig-lb1{--t1:#F1EFE8;--t2:#D3D1C7;--ln:#727E8A;
      --vb:#042C53;--vs:#85B7EB;--vt:#B5D4F4;
      --gb:#04342C;--gs:#5DCAA5;--gt:#9FE1CB;
      --nb:#2C2C2A;--ns:#727E8A;--nt:#D3D1C7;
      --ab:#3a2e0c;--as:#F0C078;--at:#FBE0A8;
      --pb:#3d1220;--ps:#E08CA6;--pt:#F4C4D2;}
    #fig-lb1 text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
    #fig-lb1 .h1{font-size:15px;font-weight:700;fill:var(--t1)}
    #fig-lb1 .h2{font-size:12.5px;font-weight:700;fill:var(--t1)}
    #fig-lb1 .lb{font-size:11.5px;font-weight:700}
    #fig-lb1 .bd{font-size:11px;fill:var(--t2)}
    #fig-lb1 .mi{font-size:10.5px;fill:var(--t2)}
    #fig-lb1 .ct{text-anchor:middle;dominant-baseline:central}
  </style>
  <text class="h1" x="20" y="26">双塔结构：2560 宽的稠密主干 + 768 宽的稀疏专家</text>
  <rect x="40" y="44" width="290" height="88" rx="10" fill="var(--vb)" stroke="var(--vs)" stroke-width="1.2"/>
  <text class="lb ct" x="185" y="64" fill="var(--vt)">Prefix · 走 VLM 塔</text>
  <text class="bd ct" x="185" y="86">图像 + 指令 + 16 个对齐 query token</text>
  <text class="mi ct" x="185" y="106">token 数可变（由图像切块数决定）</text>
  <text class="mi ct" x="185" y="122">hidden 2560</text>
  <rect x="350" y="44" width="290" height="88" rx="10" fill="var(--gb)" stroke="var(--gs)" stroke-width="1.2"/>
  <text class="lb ct" x="495" y="64" fill="var(--gt)">Suffix · 走动作专家塔</text>
  <text class="bd ct" x="495" y="86">state 55 维 + 加噪动作 50 × 55 + time</text>
  <text class="mi ct" x="495" y="106">state_proj / action_in_proj → 768</text>
  <text class="mi ct" x="495" y="122">action_time_mlp 融合 time embedding</text>
  <line x1="185" y1="132" x2="185" y2="152" stroke="var(--vs)" stroke-width="1.5"/>
  <line x1="495" y1="132" x2="495" y2="152" stroke="var(--gs)" stroke-width="1.5"/>
  <rect x="40" y="158" width="600" height="46" rx="8" fill="var(--nb)" stroke="var(--ns)" stroke-width="1.2"/>
  <text class="h2 ct" x="340" y="175">两侧各自完成 Embedding / Projection，序列在这里分开</text>
  <text class="mi ct" x="340" y="193">左长度 = prefix 长度；右长度 = 1（state）+ 50（动作）</text>
  <line x1="185" y1="204" x2="185" y2="224" stroke="var(--vs)" stroke-width="1.5"/>
  <line x1="495" y1="204" x2="495" y2="224" stroke="var(--gs)" stroke-width="1.5"/>
  <rect x="40" y="230" width="600" height="278" rx="12" fill="none" stroke="var(--ln)" stroke-width="1.2" stroke-dasharray="5 4"/>
  <text class="h2" x="56" y="252">36 层 Transformer × 逐层联合注意力</text>
  <text class="mi" x="330" y="252">两塔层数必须相等，代码里是硬断言</text>
  <rect x="56" y="264" width="258" height="70" rx="8" fill="var(--vb)" stroke="var(--vs)" stroke-width="1"/>
  <text class="lb ct" x="185" y="282" fill="var(--vt)">Layer 0 – 35 · 稠密</text>
  <text class="bd ct" x="185" y="301">32 Q / 8 KV · FFN 9728</text>
  <text class="mi ct" x="185" y="320">Qwen3-VL-4B 主干权重原样保留</text>
  <rect x="366" y="264" width="258" height="70" rx="8" fill="var(--gb)" stroke="var(--gs)" stroke-width="1"/>
  <text class="lb ct" x="495" y="282" fill="var(--gt)">Layer 0 – 35 · MoE</text>
  <text class="bd ct" x="495" y="301">16 Q / 2 KV · FFN 换 32 专家</text>
  <text class="mi ct" x="495" y="320">每层 Top-4 + shared 704</text>
  <line x1="314" y1="299" x2="366" y2="299" stroke="var(--as)" stroke-width="2" stroke-dasharray="4 3"/>
  <text class="mi ct" x="340" y="289" fill="var(--at)">联合</text>
  <text class="mi ct" x="340" y="358">↑ 每层：两塔各自算 Q/K/V → 按序列维拼成一条做一次注意力 → 按 start/end 切回 ↓</text>
  <rect x="56" y="376" width="258" height="54" rx="8" fill="var(--vb)" stroke="var(--vs)" stroke-width="1" opacity="0.55"/>
  <text class="bd ct" x="185" y="395">Layer 1 … 34（同构重复 34 次）</text>
  <text class="mi ct" x="185" y="414">动作 token 全程可见视觉语言 token</text>
  <rect x="366" y="376" width="258" height="54" rx="8" fill="var(--gb)" stroke="var(--gs)" stroke-width="1" opacity="0.55"/>
  <text class="bd ct" x="495" y="395">Layer 1 … 34（同构重复 34 次）</text>
  <text class="mi ct" x="495" y="414">prefix 的 KV 只算一次并缓存</text>
  <rect x="56" y="440" width="258" height="50" rx="8" fill="var(--vb)" stroke="var(--vs)" stroke-width="1"/>
  <text class="bd ct" x="185" y="458">Layer 35 · 末端 hidden</text>
  <text class="mi ct" x="185" y="476">仅作条件，不输出 logits</text>
  <rect x="366" y="440" width="258" height="50" rx="8" fill="var(--gb)" stroke="var(--gs)" stroke-width="1"/>
  <text class="bd ct" x="495" y="458">Layer 35 · 末端 hidden</text>
  <text class="mi ct" x="495" y="476">取后 50 个位置送输出投影</text>
  <line x1="495" y1="490" x2="495" y2="516" stroke="var(--gs)" stroke-width="1.5"/>
  <rect x="350" y="522" width="290" height="58" rx="10" fill="var(--pb)" stroke="var(--ps)" stroke-width="1.2"/>
  <text class="lb ct" x="495" y="542" fill="var(--pt)">action_out_proj：768 → 55</text>
  <text class="bd ct" x="495" y="562">输出速度场 v_t，形状 50 × 55</text>
  <line x1="495" y1="580" x2="495" y2="606" stroke="var(--ps)" stroke-width="1.5"/>
  <rect x="350" y="612" width="290" height="86" rx="10" fill="var(--ab)" stroke="var(--as)" stroke-width="1.2"/>
  <text class="lb ct" x="495" y="632" fill="var(--at)">Flow Matching 去噪循环 × 10</text>
  <text class="bd ct" x="495" y="652">x_t ← x_t + dt · v_t，dt = −1/10</text>
  <text class="mi ct" x="495" y="670">t 从 1.0 递减，while t ≥ −dt/2</text>
  <text class="mi ct" x="495" y="687">prefix KV 复用 10 次，不重算</text>
  <rect x="40" y="716" width="600" height="76" rx="12" fill="var(--nb)" stroke="var(--ns)" stroke-width="1.2"/>
  <text class="h2" x="56" y="738">输出：50 × 55 的连续动作块</text>
  <text class="bd" x="56" y="759">55 = 14 臂 + 14 末端位姿 + 2 夹爪 + 12 手 + 4 腰 + 2 头 + 3 底盘 + 4 预留</text>
  <text class="mi" x="56" y="778">真机部署只执行前 25 步就重规划，chunk 50 是生成长度而非执行长度</text>
</svg>
<p class="fig-cap">图 1：LingBot-VLA 2.0 的完整数据流。两塔宽度相差 3.3 倍（2560 对 768），但层数严格相等（36），每层通过一次联合注意力让窄塔的动作 token 读取宽塔的视觉语言上下文。</p>
</div>

### ① 两塔层数相等是硬约束

`modeling_lingbot_vla_v2.py` 里有一行断言：

```python
models = [self.qwenvl.model.language_model, self.qwen_expert.model]
assert action_num_layers == num_layers, (
    "Action expert and VLM must have the same number of layers "
    f"(got action={action_num_layers}, vlm={num_layers})."
)
```

为什么必须相等？因为两塔是**逐层咬合**的——第 i 层的动作专家要接收第 i 层主干的注意力输出。层数不等，这套配对就断了。这和 π0 一脉相承：π0 的动作专家也是与 PaliGemma 层数对齐的同构窄塔。

### ② 联合注意力：一次计算，两边分食

每层的流程是：两塔各自用本层的 `q_proj` / `k_proj` / `v_proj` 算出 Q/K/V → 按序列维拼成一条 → 过**一次**注意力 → 再按 `start`/`end` 切回各自那段。代码骨架：

```python
for layer_idx in range(num_layers):
    for i, hidden_states in enumerate(inputs_embeds):
        q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True, ...)
        query_states.append(q.float())
        key_states.append(k.float())
        value_states.append(v.float())
    query_states = torch.cat(query_states, dim=1)   # 两塔拼接
    ...
    att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)
    # 之后按 start/end 把 att_output 切回两塔
```

一个容易误解的点：**两塔头数不同**（主干 32 Q / 8 KV，专家 16 Q / 2 KV），但因为宽度也不同（2560 对 768），投影后每塔的 QKV 各自匹配自己的头布局，拼接发生在序列维上。所以注意力矩阵是**跨塔稠密**的——动作 token 直接 attend 到每一个图像和语言 token，不需要额外的 cross-attention adapter。

### ③ 推理时的 KV 复用

`sample_actions()` 把 prefix 单独跑一遍 `fill_kv_cache=True`，拿到 `past_key_values`；随后的 10 步去噪只跑 suffix。这意味着**图像编码和语言理解在整个去噪循环里只做一次**，10 步的边际成本几乎全在 768 宽的窄塔上。

这是 130ms 延迟能成立的关键——如果每步都重跑视觉塔，4090D 上不可能做到。

### ④ AdaNorm：时间条件不走 cross-attention

`adanorm_time: true` 是配置里的一个开关。时间步 embedding 不通过 cross-attention 注入，而是直接调制 LayerNorm 的缩放平移参数（`replace_lnorm_with_adanorm`），并且末端还有一个 `final_norm_adanorm`。

这和 π0 系的做法同源——用 AdaNorm 注入时间条件比 cross-attention 便宜得多，而且不占序列位置。对 10 步去噪的小模型来说，这点开销省得有意义。

## 2. MoE 放在哪儿：只在动作专家的 FFN 里

这是最容易读错的地方。不少解读写「LingBot-VLA 2.0 是稀疏 MoE 模型」，听起来像主干也走 MoE。**实际不是。**

论文第 4.1 节的原文说得很清楚：

> we instantiate sparse MoE layers inside the action expert, which replace the feed-forward network (FFN) in the action expert… all action-expert transformer blocks are equipped with MoE FFNs

代码层面也能验证——`_install_moe_blocks()` 只改 `qwen_expert` 的 MLP，主干的 `qwenvl` 完全不动：

```python
for idx in token_moe_layers:
    self.qwen_expert.model.layers[idx].mlp = Qwen2TokenMoeBlock(token_config)
```

所以准确的说法是：**Qwen3-VL-4B 主干是稠密的，动作专家的 36 层 FFN 全部换成了稀疏 MoE。**

<div class="warnbox">
<strong>一个必须辨析的常见错误：</strong>「Sparse MoE Action Expert 32 / Top-4」这句话数字没错（配置里确实是 <code>token_num_experts: 32</code>、<code>token_top_k: 4</code>），但它只描述动作专家，<strong>不能推广到整个模型</strong>。Qwen3-VL-4B 主干仍是稠密 FFN，intermediate 9728。
</div>

`real_robot.yaml` 里的完整 MoE 配置：

```yaml
use_moe: true
token_moe_layers: [0,1,2,...,35]      # 36 层全覆盖，无一层豁免
token_num_experts: 32
token_top_k: 4
token_moe_intermediate_size: 512     # 单个专家宽度
token_shared_intermediate_size: 704  # 共享专家宽度
router_activation: "sigmoid"
routed_scaling_factor: 4.0
bias_update_speed: 0                 # 关闭 bias 修正
sequence_wise_loss_coeff: 1e-3
router_z_loss_coeff: 1e-4
use_shared_expert_gate: false
```

### 那条宽度恒等式

上面这两行——`512` 和 `704`——看起来是随手取的。但把它们和配置类的默认值放在一起，会发现一件事：

```
Top-4 激活专家 + 共享专家 = 4 × 512 + 704 = 2752
被替换的稠密 FFN intermediate_size  = 2752
                                      相等 ✓
```

`configuration_lingbot_vla.py` 里 `expert_intermediate_size: int = 2752` 就是那个被替换掉的稠密 FFN 宽度。MoE 化之后，**每次前向实际激活的 FFN 宽度与原稠密版一模一样**。

<div class="fig">
<svg viewBox="0 0 680 580" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="宽度恒等式：一个 2752 宽的稠密 FFN 被替换为 32 个 512 宽专家，Top-4 激活 2048 加共享专家 704，恰好等于 2752；每层参数从 6.34M 涨到 39.37M" id="fig-lb2">
  <style>
    #fig-lb2{--t1:#1a1a1a;--t2:#5F5E5A;--ln:#96918A;
      --db:#F1EFE8;--ds:#5F5E5A;--dt:#444441;
      --eb:#E6F1FB;--es:#185FA5;--et:#0C447C;
      --onb:#185FA5;--shb:#1D9E75;--offb:#CDC9BC;
      --okb:#E1F5EE;--oks:#1D9E75;--okt:#04342C;}
    :root[data-theme="dark"] #fig-lb2{--t1:#F1EFE8;--t2:#D3D1C7;--ln:#727E8A;
      --db:#2C2C2A;--ds:#727E8A;--dt:#D3D1C7;
      --eb:#042C53;--es:#85B7EB;--et:#B5D4F4;
      --onb:#378ADD;--shb:#28B589;--offb:#5F5E5A;
      --okb:#04342C;--oks:#5DCAA5;--okt:#9FE1CB;}
    #fig-lb2 text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
    #fig-lb2 .h1{font-size:15px;font-weight:700;fill:var(--t1)}
    #fig-lb2 .h2{font-size:12px;font-weight:700;fill:var(--t1)}
    #fig-lb2 .mi{font-size:10.5px;fill:var(--t2)}
    #fig-lb2 .ct{text-anchor:middle;dominant-baseline:central}
  </style>
  <text class="h1" x="20" y="26">宽度恒等式：算力不变，容量放大 6.2 倍</text>
  <text class="h2" x="40" y="56">替换前 · 稠密 FFN</text>
  <rect x="40" y="66" width="600" height="56" rx="6" fill="var(--db)" stroke="var(--ds)" stroke-width="1.2"/>
  <text class="ct" x="340" y="86" font-size="12.5" font-weight="700" fill="var(--dt)">intermediate_size = 2752</text>
  <text class="ct" x="340" y="106" font-size="11" fill="var(--t2)">每层 3 × 768 × 2752 = 6.34 M 参数</text>
  <line x1="340" y1="122" x2="340" y2="148" stroke="var(--ln)" stroke-width="1.5"/>
  <text class="mi ct" x="352" y="135">替换</text>
  <text class="h2" x="40" y="172">替换后 · 32 专家 MoE（Top-4 激活）</text>
  <rect x="40" y="182" width="600" height="32" rx="6" fill="var(--db)" stroke="var(--ds)" stroke-width="0.8" stroke-dasharray="3 2"/>
  <text class="mi ct" x="340" y="198">32 个专家，每个 512 宽 —— 权重都在，但每次只挑 4 个（两行各 16 个）</text>
  <g>
    <rect x="40" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="78" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="116" y="226" width="30" height="30" rx="3" fill="var(--onb)"/>
    <rect x="154" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="192" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="230" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="268" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="306" y="226" width="30" height="30" rx="3" fill="var(--onb)"/>
    <rect x="344" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="382" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="420" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="458" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="496" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="534" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="572" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="610" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="40" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="78" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="116" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="154" y="262" width="30" height="30" rx="3" fill="var(--onb)"/>
    <rect x="192" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="230" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="268" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="306" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="344" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="382" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="420" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="458" y="262" width="30" height="30" rx="3" fill="var(--onb)"/>
    <rect x="496" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="534" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="572" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="610" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
  </g>
  <text class="mi" x="40" y="312">4 个蓝色 = 本轮被 Top-4 选中的专家（4 × 512 = 2048）</text>
  <text class="mi" x="40" y="328">28 个灰色 = 未激活，权重占显存但本轮不参与计算</text>
  <rect x="40" y="344" width="600" height="44" rx="6" fill="var(--okb)" stroke="var(--oks)" stroke-width="1.2"/>
  <text class="ct" x="340" y="359" font-size="11.5" font-weight="700" fill="var(--okt)">共享专家 704 · 始终激活，不在这 32 个之内</text>
  <text class="ct" x="340" y="377" font-size="10.5" fill="var(--okt)">兜住跨本体通用的那部分先验</text>
  <rect x="40" y="400" width="600" height="58" rx="8" fill="var(--okb)" stroke="var(--oks)" stroke-width="1.2"/>
  <text class="ct" x="340" y="420" font-size="12.5" font-weight="700" fill="var(--okt)">4 × 512 + 704 = 2752 = 原稠密 FFN 宽度</text>
  <text class="ct" x="340" y="441" font-size="11" fill="var(--okt)">每次前向的激活宽度完全一致，FLOPs 几乎不变</text>
  <text class="h2" x="40" y="484">代价与收益</text>
  <rect x="40" y="496" width="290" height="66" rx="8" fill="var(--eb)" stroke="var(--es)" stroke-width="1"/>
  <text class="ct" x="185" y="514" font-size="11.5" font-weight="700" fill="var(--et)">参数：6.34 M → 39.37 M / 层</text>
  <text class="ct" x="185" y="534" font-size="10.5" fill="var(--t2)">36 层 FFN 合计 0.228 B → 1.42 B</text>
  <text class="ct" x="185" y="551" font-size="10.5" fill="var(--t2)">显存占用涨 6.2 倍</text>
  <rect x="350" y="496" width="290" height="66" rx="8" fill="var(--eb)" stroke="var(--es)" stroke-width="1"/>
  <text class="ct" x="495" y="514" font-size="11.5" font-weight="700" fill="var(--et)">算力：不变</text>
  <text class="ct" x="495" y="534" font-size="10.5" fill="var(--t2)">激活宽度锁定 2752</text>
  <text class="ct" x="495" y="551" font-size="10.5" fill="var(--t2)">推理 FLOPs 与稠密版同级</text>
</svg>
<p class="fig-cap">图 2：宽度恒等式。32 个 512 宽专家取 Top-4 得到 2048，加上 704 宽的共享专家，总和正好等于被替换的稠密 FFN 宽度 2752。每层参数从 6.34M 涨到 39.37M，但每次前向的激活宽度不变。</p>
</div>

这个设计的意图很清楚：**用显存换容量**。32 个专家让模型有能力为不同本体、不同任务准备专门的知识分区，而每次前向付出的算力成本与调大前的稠密模型相同。论文的说法是「fine-grained expert segmentation and shared expert isolation allow universal priors and specialized embodiment/task patterns to coexist under the same active compute budget」——粒度细到 512 宽，是为了组合更灵活；共享专家独立，是为了兜住跨本体通用的那部分先验。

<div class="keybox">
<strong>为什么 512 和 704 值得记：</strong>如果专家宽度不是 512，这条恒等式就不成立。反过来说，看到这个恒等式就能推断：设计师先定了「FFN 总宽保持 2752」这个约束，再倒推出「32 专家 × 512 + 共享 704」。这是 MoE 改造里最稳妥的做法——<strong>先锁算力，再放大容量</strong>，避免动结构顺带把推理成本推上天。
</div>

### 路由侧的几个非默认选择

对比配置类默认值，`real_robot.yaml` 在路由上做了四处偏离：

| 字段 | 配置类默认 | real_robot.yaml | 含义 |
|---|---|---|---|
| `router_activation` | `softmax` | **`sigmoid`** | 每个专家独立打分，不强制归一化 |
| `routed_scaling_factor` | `1.0` | **`4.0`** | 路由输出放大 4 倍，补偿 Top-4 摊薄 |
| `bias_update_speed` | `0.001` | **`0`** | 关闭 bias 修正，纯靠辅助损失均衡 |
| `use_shared_expert_gate` | `true` | **`false`** | 共享专家直接相加，不学门控 |

sigmoid + scaling 4.0 是 DeepSeek 系 MoE 的经典组合：sigmoid 让专家之间不争份额，scaling 把激活值量级拉回来。`bias_update_speed: 0` 值得注意——它放弃了 inference-time 的专家负载修正，改用两个辅助损失来管：

- `sequence_wise_loss_coeff: 1e-3`，模式 `per_sequence`：按**单条序列内**的专家使用分布做均衡，而非整个 batch
- `router_z_loss_coeff: 1e-4`：约束 router logits 量级，防止路由过度自信

按序列均衡是个有意思的选择。VLA 的 batch 里混着不同本体、不同任务的数据，按 batch 全局均衡会让「某条序列霸占某几个专家」被平均掉；按序列均衡则能压住这种偏斜。

## 3. Dual-Query 蒸馏：不是两个 query，是 8 + 8 个 token

论文把这块写成 `[Q_t, Q_{t+T}]` 两个 query，看起来像两个 token。**实际上每组是 8 个 token，两组共 16 个。**

配置里 `num_task_tokens: 8`，代码里 `_get_align_tokens()` 把可学习 embedding 表按 8 组取均值，输出 8 个 token：

```python
def _get_align_tokens(tokens):
    tk_weights = tokens.view(self.num_task_tokens,
                             tokens.shape[0] // self.num_task_tokens,
                             tokens.shape[1])
    tk_weights = tk_weights.mean(dim=1)
    return tk_weights            # 形状 (8, hidden)
```

然后这两组 token 被插进 prefix 序列参与联合注意力：

```python
for segment_name in prefix_query_segments(use_depth_align=True, use_future_depth=..., ...):
    if segment_name == "current_depth":   _append(align_embs, ...)
    elif segment_name == "future_video":  _append(future_video_align_embs, ...)
    elif segment_name == "future_depth":  _append(future_align_embs, ...)
```

所以「Dual」指的是**两组**（当前帧 + 未来帧），不是「两个 token」。概念的符号化表述（`Q_t` / `Q_{t+T}`）和实现的 token 数不是一回事，写技术文档时这点必须分清。

<div class="fig">
<svg viewBox="0 0 680 452" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dual-Query 蒸馏结构：prefix 序列由图像 token、8 个当前帧 query、语言 token、8 个未来帧 query 组成，两组 query 分别从 LingBot-Depth 蒸馏几何、从 DINO-Video 蒸馏时序" id="fig-lb3">
  <style>
    #fig-lb3{--t1:#1a1a1a;--t2:#5F5E5A;--ln:#96918A;
      --ib:#F1EFE8;--is:#5F5E5A;--it:#444441;
      --qb:#B5D4F4;--qs:#185FA5;--qt:#0C447C;
      --lb2:#E1F5EE;--ls:#1D9E75;--lt:#04342C;
      --tb:#FAEEDA;--ts:#854F0B;--tt:#633806;
      --wb:#FBEAF0;--ws:#993556;--wt:#72243E;}
    :root[data-theme="dark"] #fig-lb3{--t1:#F1EFE8;--t2:#D3D1C7;--ln:#727E8A;
      --ib:#2C2C2A;--is:#727E8A;--it:#D3D1C7;
      --qb:#185FA5;--qs:#85B7EB;--qt:#B5D4F4;
      --lb2:#04342C;--ls:#5DCAA5;--lt:#9FE1CB;
      --tb:#3a2e0c;--ts:#F0C078;--tt:#FBE0A8;
      --wb:#3d1220;--ws:#E08CA6;--wt:#F4C4D2;}
    #fig-lb3 text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
    #fig-lb3 .h1{font-size:15px;font-weight:700;fill:var(--t1)}
    #fig-lb3 .h2{font-size:12px;font-weight:700;fill:var(--t1)}
    #fig-lb3 .lb{font-size:11px;font-weight:700}
    #fig-lb3 .bd{font-size:10.5px;fill:var(--t2)}
    #fig-lb3 .ct{text-anchor:middle;dominant-baseline:central}
  </style>
  <text class="h1" x="20" y="26">Prefix 序列的构成：16 个可学习 query token 藏在这里</text>
  <rect x="40" y="46" width="150" height="52" rx="6" fill="var(--ib)" stroke="var(--is)" stroke-width="1.2"/>
  <text class="ct" x="115" y="64" font-size="11.5" font-weight="700" fill="var(--it)">图像 token</text>
  <text class="ct" x="115" y="83" font-size="10" fill="var(--t2)">长度可变</text>
  <rect x="196" y="46" width="118" height="52" rx="6" fill="var(--qb)" stroke="var(--qs)" stroke-width="1.4"/>
  <text class="ct" x="255" y="64" font-size="11.5" font-weight="700" fill="var(--qt)">Q_t × 8</text>
  <text class="ct" x="255" y="83" font-size="10" fill="var(--qt)">当前帧几何</text>
  <rect x="320" y="46" width="150" height="52" rx="6" fill="var(--ib)" stroke="var(--is)" stroke-width="1.2"/>
  <text class="ct" x="395" y="64" font-size="11.5" font-weight="700" fill="var(--it)">语言 token</text>
  <text class="ct" x="395" y="83" font-size="10" fill="var(--t2)">上限 72</text>
  <rect x="476" y="46" width="118" height="52" rx="6" fill="var(--qb)" stroke="var(--qs)" stroke-width="1.4"/>
  <text class="ct" x="535" y="64" font-size="11.5" font-weight="700" fill="var(--qt)">Q_t+T × 8</text>
  <text class="ct" x="535" y="83" font-size="10" fill="var(--qt)">未来帧演化</text>
  <text class="bd" x="40" y="118">↑ 段顺序由 prefix_query_segments 决定，上图是一种可行排布</text>
  <line x1="255" y1="124" x2="255" y2="152" stroke="var(--qs)" stroke-width="1.5"/>
  <line x1="535" y1="124" x2="535" y2="152" stroke="var(--qs)" stroke-width="1.5"/>
  <rect x="40" y="158" width="270" height="96" rx="10" fill="var(--lb2)" stroke="var(--ls)" stroke-width="1.2"/>
  <text class="ct" x="175" y="178" font-size="12" font-weight="700" fill="var(--lt)">LingBot-Depth（教师）</text>
  <text class="ct" x="175" y="200" font-size="10.5" fill="var(--t2)">单目深度估计</text>
  <text class="ct" x="175" y="219" font-size="10.5" fill="var(--t2)">loss = L1，权重 0.004</text>
  <text class="ct" x="175" y="238" font-size="10.5" fill="var(--t2)">与 query 做 8 token 对齐 → 式 (9)</text>
  <rect x="372" y="158" width="268" height="96" rx="10" fill="var(--wb)" stroke="var(--ws)" stroke-width="1.2"/>
  <text class="ct" x="506" y="178" font-size="12" font-weight="700" fill="var(--wt)">DINO-Video（教师）</text>
  <text class="ct" x="506" y="200" font-size="10.5" fill="var(--t2)">303.13 M · 5 M clips · 16 帧</text>
  <text class="ct" x="506" y="219" font-size="10.5" fill="var(--t2)">loss = 平方 Frobenius，权重 0.004</text>
  <text class="ct" x="506" y="238" font-size="10.5" fill="var(--t2)">block-wise 因果注意力 + 3D-RoPE</text>
  <rect x="40" y="272" width="600" height="76" rx="10" fill="var(--tb)" stroke="var(--ts)" stroke-width="1.2"/>
  <text class="ct" x="340" y="292" font-size="12" font-weight="700" fill="var(--tt)">蒸馏方向：教师 → query token，query 参与主干前向</text>
  <text class="ct" x="340" y="313" font-size="10.5" fill="var(--t2)">推理阶段教师全部卸载，只留下这 16 个 token 的权重</text>
  <text class="ct" x="340" y="332" font-size="10.5" fill="var(--t2)">作用：把「当前场景几何 + 未来场景演化」压进主干能读的位置</text>
  <rect x="40" y="364" width="600" height="72" rx="10" fill="var(--ib)" stroke="var(--is)" stroke-width="1.2"/>
  <text class="ct" x="340" y="384" font-size="12" font-weight="700" fill="var(--it)">一个易混点：num_task_tokens = 8 是「每组」的 token 数</text>
  <text class="ct" x="340" y="405" font-size="10.5" fill="var(--t2)">论文写 Q_t 与 Q_t+T 两个符号，指的是两组 query，不是两个 token</text>
  <text class="ct" x="340" y="424" font-size="10.5" fill="var(--t2)">video.share_future_depth_query: true → 未来视频与未来深度共用同一组</text>
</svg>
<p class="fig-cap">图 3：Dual-Query 蒸馏。两组各 8 个可学习 query token 混在 prefix 里参与联合注意力，训练时分别从 LingBot-Depth 与 DINO-Video 蒸馏；推理时教师卸载，只留 query 权重。</p>
</div>

### 为什么用 query 而不是直接喂特征

如果目标是「让主干知道当前深度和未来画面」，最直接的做法是把深度图 / 未来帧当额外输入喂进去。但这样有两个问题：输入序列变长（深度图和视频帧都不便宜），且推理时必须真的跑教师模型。

query 蒸馏绕开了这两点：**知识被压进 16 个 token 的权重里**——教师只在训练时出场，推理时完全不需要。代价是这 16 个 token 得挤过联合注意力被主干读到，信息带宽有限，所以三个蒸馏权重都设得很小（`depth_loss_weight`、`future_depth_loss_weight`、`future_video_loss_weight` 全是 **0.004**），避免辅助任务压过动作主损失。

`video.share_future_depth_query: true` 是另一处细节：未来视频和未来深度**共用同一组** query，而不是各给 8 个。`real_robot.yaml` 里 `future_video_share_future_depth_query: true` 配 `use_future_depth: true`，代码里还有校验：

```python
if (not self.use_future_depth and getattr(self, "use_future_video", False)
        and getattr(self, "future_video_share_future_depth_query", False)):
    raise ValueError("share_future_depth_query=True requires depth.use_future_depth=True.")
```

共享的理由是：未来深度和未来视频描述的是**同一件事的两个侧面**（几何 vs 外观），共用 query 能强制它们落在同一语义空间，也省 8 个位置。

## 4. 动作生成：Flow Matching，10 步，chunk 50

`sample_actions()` 的循环很朴素，就是显式欧拉：

```python
dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
x_t = noise
time = torch.tensor(1.0, dtype=dtype, device=device)
while time >= -dt / 2:
    v_t = predict_velocity_fn(state, prefix_pad_masks, past_key_values,
                              x_t, expanded_time, ...)
    x_t += dt * v_t
    time += dt
```

`num_steps = 10` 是 `configuration_lingbot_vla.py` 里硬编码的：

```python
self.num_steps = 10
self.n_obs_steps = 1
```

`dt = -0.1`，`t` 从 1.0 递减，`while t >= 0.05` —— 数下来正好 10 次。与 README 的「10 denoising steps / 130 ms」完全对得上。

<div class="warnbox">
<strong>一个坑：</strong><code>deploy/lingbot_vla_v2_policy.py</code> 第 309 行有一句被注释掉的 <code># config.num_steps = 4</code>。有人会据此以为默认是 4 步。不是——那是某次实验留下的，注释状态不生效，实际走配置类的 10 步。
</div>

训练侧是标准 Flow Matching 插值：

```python
x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions      # 目标速度场
```

注意时间方向：`t=1` 对应纯噪声，`t=0` 对应干净动作，所以推理时 `dt` 取负、`t` 从 1.0 往 0 走。这个约定和 π0 一致。

Action chunk 长度是 **50**，写在配置类里：

```python
chunk_size: int = 50,
...
self.chunk_size = chunk_size
self.n_action_steps = chunk_size
```

`real_robot.yaml` 没有覆盖这个字段，所以就是 50。采样时噪声形状用 `(bsize, n_action_steps, max_action_dim)`，即 `(B, 50, 55)`。

## 5. 参数量核算：6.38B 装了什么

HF 模型卡的 safetensors 计数是 **6,375,907,511**，这是最权威的口径（全部 FP32 统计，对应权重文件约 25.5 GB）。拆开看：

| 组件 | 计算式 | 参数量 |
|---|---|---|
| Qwen3-VL-4B 主干 | 36 层，hidden 2560，FFN 9728 | ≈ 4.0 B |
| 动作专家注意力（36 层） | 36 × (768×2048 + 2 × 768×256 + 2048×768) | ≈ 0.13 B |
| 动作专家 MoE FFN（36 层） | 36 × (32 × 3 × 768 × 512 + 3 × 768 × 704) | 1.42 B |
| 输入输出投影与时间 MLP | 55×768 ×2 + 768×55 + 1536×768 + 768×768 | ≈ 0.002 B |
| 视觉塔与 deepstack 等 | Qwen3-VL 视觉侧 | ≈ 0.8 B |
| **合计** | | **≈ 6.35 B** |

与 HF 精确值 6.376B 差约 0.03B，缺口来自 RMSNorm 权重、router gate 与各类 bias。

MoE FFN 那 1.42B 是全场最大的单项增量。作为对照，如果这 36 层还是稠密 FFN（2752 宽），只有 0.228B——**MoE 化多花 1.19B 参数，换来 6.2 倍的 FFN 容量**。

<div class="keybox">
<strong>激活参数量怎么看：</strong>按 per-token 激活算，MoE 部分只走 Top-4 + 共享，激活宽度等于「4 × 512 + 704」，和稠密版完全一样。所以这个模型的<strong>总参数 6.38B，激活规模也在 6.38B 量级</strong>——它不是「总参 100B、激活 5B」那种稀疏大模型。MoE 在这里的作用不是省算力，是<strong>在固定算力下扩容</strong>。这一点和 fa3 / fa10 里 MiniMax 用 MoE 的目的完全不同。
</div>

## 6. 训练配方

`real_robot.yaml` 的 `train` 段落信息密度很高，挑几个值得记的：

| 字段 | 值 | 说明 |
|---|---|---|
| `optimizer` | `muon` | 非 AdamW |
| `lr` | 5e-5，`constant` | 不衰减 |
| `micro_batch_size` / `global_batch_size` | 32 / 256 | 梯度累积 8 步 |
| `max_steps` | 60,000 | |
| `vlm_causal` | **true** | 主干改为因果注意力（VLA 常规做法） |
| `attention_implementation` | `flex_cached` | FlexAttention + KV 缓存 |
| `enable_fp32` | **true** | 权重 FP32，对应 HF 的 F32 统计 |
| `data_parallel_mode` | `fsdp2` | |
| `tokenizer_max_length` | 72 | 指令长度上限 |
| `freeze_vision_encoder` | false | 视觉塔参与训练 |

### Distributed Muon：37.6% 的训练提速

README 给了一组单机 H20 的实测：

| 优化器 | Clean 成功率 | Randomized | 迭代耗时 |
|---|---|---|---|
| Muon | 93.52 | 92.80 | 5.93 s/it |
| Distributed Muon | 91.56 | 91.34 | **4.31 s/it** |

速度提升 **37.6%**（5.93 → 4.31），代价是成功率掉约 2 个点。官方把参数配置放在 `robotwin_dist_muon.yaml` 里供对照——这种把 trade-off 摆明面的做法，比只报最优值更有参考价值。

Distributed Muon 基于 TorchTitan 的 FlexShard 实现，要求 FSDP2 且至少 2 个 data-parallel rank。

### 预训练数据

- 约 **60,000 小时**，其中 **50,000 小时机器人数据，覆盖 20 种构型**
- **10,000 小时第一人称视频**（人类操作视角）

20 种构型 × 统一 55 维动作空间，这是「一脑多机」的数据基础。

## 7. 55 维：跨本体的统一接口

这是整个设计里最「工程」的部分。不同机器人的自由度差异极大——单臂 6 轴、双臂 14 轴、带灵巧手再加 12 轴、带底盘再加 3 轴。LingBot-VLA 2.0 的做法是**定一个足够大的统一空间，用不满的维度填零**：

| 段 | 维度 | 内容 |
|---|---|---|
| 臂关节位置 | 14 | 双臂，每臂 7 轴 |
| 末端位姿 | 14 | 双臂末端，每臂 7 维（位置 3 + 姿态 4） |
| 夹爪 | 2 | 两指开合 |
| 灵巧手关节 | 12 | 每手 6 轴 |
| 腰部 | 4 | |
| 头部 | 2 | |
| 底盘 / 移动信号 | 3 | |
| 预留 | 4 | |
| **合计** | **55** | |

配置里三个字段全部锁在 55：

```yaml
action_dim: 55
max_action_dim: 55
max_state_dim: 55
```

这样不同本体共享同一套输入输出投影（`state_proj: 55 → 768`、`action_out_proj: 768 → 55`），权重完全复用。代价是简单本体（比如 7 轴单臂）会浪费大量维度——但和「每个本体一套投影」相比，这个浪费换来的跨本体知识共享更值。

<div class="keybox">
<strong>和 π0 的对比：</strong>π0 用 14 维（双臂各 7 轴），π0.5 也在这个量级。LingBot-VLA 2.0 直接拉到 55 维，把灵巧手、腰部、头部、底盘全塞进来——这是「一脑多机」叙事在数据结构上的具体体现。<strong>维度留得够宽，才装得下不同形态的本体。</strong>
</div>

## 8. 部署：生成 50 步，只执行 25 步

这一节讲一个最容易搅混的点。

README 的开环评估命令写 `--use_length 50`，真机部署用 `--use_length 25`。这不是两个不同的 chunk 长度，而是**同一个 50 步 chunk，执行时只取前 N 步**：

```python
if self.use_length > 0:
    assert self.use_length <= unnormalized_actions[output_key].shape[1]
    unnormalized_actions[output_key] = unnormalized_actions[output_key][:, :self.use_length]
```

这就是 receding horizon 控制：**每次生成完整的 50 步动作块，但只执行前 25 步，然后重新观测、重新规划。** 越靠后的动作预测误差越大，截断掉能提升鲁棒性。

`use_length` 还控制重规划频率：

```python
if ... or self.global_step % self.use_length == 0 or self.use_length == -1:
```

`use_length = -1` 意味着每步都重规划。开环评估用 50 是因为要在离线数据上算完整轨迹的误差。

所以回答「chunk 是 50 还是 25」：**生成长度 50，执行长度 25，两者不矛盾。**

## 9. 性能数据

### 真机 · GM-100 双臂操作（进展分 / 成功率）

| 平台 | GR00T N1.7 | π0.5 | LingBot-VLA-1.0 | LingBot-VLA 2.0 |
|---|---|---|---|---|
| AgileX Cobot Magic | 36.3 / 17.8 | 59.1 / 32.2 | 58.2 / 30.0 | **66.2 / 34.4** |
| Galaxea R1Pro | 16.4 / 5.6 | 27.4 / 8.9 | 32.7 / **15.6** | **34.6 / 15.6** |

### 真机 · 长程移动操作

| 本体 | 任务 | 设定 | LingBot-VLA 2.0 | π0.5 |
|---|---|---|---|---|
| Astribot S1 | 冰箱整理 | 域内 | **77.1 / 60.0** | 65.3 / 46.7 |
| Astribot S1 | 冰箱整理 | 域外 | **37.0 / 13.3** | 30.3 / 6.7 |
| Cobot Magic-ARX X5 | 灶台清洁 | 域内 | **84.3 / 66.7** | 79.9 / 60.0 |
| Cobot Magic-ARX X5 | 灶台清洁 | 域外 | **67.5 / 40.0** | 62.5 / 33.3 |

### 仿真 · RoboTwin 2.0 成功率

| 任务 | π0.5 | LingBot-VLA-1.0 | LingBot-VLA 2.0 |
|---|---|---|---|
| Clean | 82.74% | 88.56% | **93.52%** |
| Randomized | 76.76% | 86.68% | **92.80%** |

<div class="warnbox">
<strong>读数时要看清的两点：</strong>第一，这些全是<strong>厂商自报口径</strong>，没有独立复现；第二，Galaxea R1Pro 上 2.0 的成功率与 1.0 完全持平（15.6），提升全在进展分（32.7 → 34.6）——说明在这个平台上它「做得更细」但没有「做得更成」，别把进展分和成功率混着看。
</div>

Clean 与 Randomized 只差 0.72 个点（93.52 对 92.80），这个泛化 gap 相当小，说明 MoE 那 32 个专家确实学到了对视觉扰动不敏感的表征，而不是记住了干净数据。

## 10. 数据存疑处

写这份拆解时有几处口径对不齐，列出来比含糊过去更有用：

| 疑点 | 现象 | 我的判断 |
|---|---|---|
| `num_task_tokens` 语义 | 论文写 `Q_t`、`Q_{t+T}` 两个符号 | **每组 8 个 token**，两组共 16。符号是概念表述，不是 token 计数 |
| 未来帧间隔 T | 论文只说「future」 | **官方未披露具体秒数**。yaml 里有 `use_future_image: true` 但无间隔字段 |
| Dual-Query 的双分支 | 材料描述为「Understanding Branch / Action Branch」 | 论文措辞是**因果 VLM + Action Expert**，两塔逐层联合注意力，不是两条独立分支 |
| `num_steps` | README 说 10，deploy 脚本有注释 `# config.num_steps = 4` | **10 是生效值**，注释不执行 |
| 专家负载分布 | 无公开 expert utilization 曲线 | `bias_update_speed: 0` 关闭了 bias 修正，靠 `1e-3` 序列均衡损失管，实际均衡度未知 |
| 参数量 6.38B | 官方「6B」为概数 | HF safetensors 精确值 **6,375,907,511**，官方 6B 是四舍五入 |

## 11. 与系列文章的衔接

- **vla3（π 系列演进）**：本文与 π0 / π0.5 共享同一套骨架（稠密 VLM + 窄动作专家 + Flow Matching），差别在动作专家的 FFN 是否 MoE 化。读 vla3 再看本文，能看清「同一架构骨架上的两种效率路线」。
- **fa3 / fa10（MiniMax M3 系列）**：那些文章里 MoE 的目的是**省算力**（128 专家 Top-4，激活远小于总参）；本文的 MoE 目的是**固定算力下扩容**。同一个工具，两种用法，这个对比本身就是 MoE 设计空间的注脚。
- **fa7 / fa8（Qwen3.8 双 checkpoint）**：本文主干就是 Qwen3-VL-4B，Qwen 系的稠密架构在这里作为固定底座出现，可以对照看 Qwen 在 LLM 与 VLM 两侧的宽度选择。
- **fw9（Wan2.2 双专家）**：那篇讲的是**按扩散时间步路由**的两个大专家（高噪 / 低噪各一个 14B）；本文讲的是**按 token 路由**的 32 个小专家。同为「专家」，路由维度完全不同——一个是时间轴上的分段，一个是 token 轴上的选择。

<div class="keybox">
<strong>一句话收尾：</strong>LingBot-VLA 2.0 的 MoE 是一个「保守但正确」的设计——它没有追求参数量的漂亮数字，而是先用宽度恒等式把算力锁死，再把容量放大 6.2 倍。对一个要跑在机器人本体上的模型来说，<strong>可预测的推理延迟比参数量重要得多</strong>。130ms 的底气，一半来自 KV 复用，一半来自这条恒等式。
</div>
