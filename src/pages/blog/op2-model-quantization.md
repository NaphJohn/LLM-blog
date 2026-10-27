---
title: '算子讲解手记（二）：模型量化——从精度谱系到四条路线的总览'
description: '量化手记系列（qt1–qt5）的导读入口。保留原有的精度谱系表、QAT 路线、KV Cache 与推理引擎落地清单，另外补一张完整的「量化方法谱系总表」——把 RTN / GPTQ / AWQ / SmoothQuant / QuaRot / SpinQuant / DuQuant / KIVI 按「改什么」分成四条路线，标清各自的上限位宽、计算成本、校准需求，以及哪两两可以叠加。想深入某一个方法，直接跳到对应手记。'
pubDate: 2026-09-08
series: 算子讲解手记
lang: zh
altLang: en
altHref: /en/blog/op2-model-quantization
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

模型量化是"用精度换显存/带宽/算力"的艺术——它和**蒸馏**（用容量换体积）是正交的两件事，可以叠加（先蒸馏小模型再量化部署）。

本篇是**导读**：给出全景谱系和选型清单，具体方法展开在「量化手记」系列五篇里。

<div class="keybox">
<strong>系列索引：</strong><br>
<strong>qt1 · 量化的基本语言</strong>——量化公式、粒度（per-tensor/channel/group/token）、对称性、校准、四条路线地图<br>
<strong>qt2 · RTN 与 GPTQ</strong>——A 路线「只改舍入」，含 OBS 框架完整推导与 GPTQ 伪代码<br>
<strong>qt3 · SmoothQuant 与 AWQ</strong>——B 路线「改分布·缩放」，激活 outlier 的两条相反解法<br>
<strong>qt4 · QuaRot / SpinQuant / DuQuant</strong>——C 路线「改分布·旋转」，W4A4 的唯一出路，四个插入位置 R1–R4<br>
<strong>qt5 · KV Cache 量化与 FP4</strong>——正交维度，以及 FP8/FP4 格式如何改变选型逻辑
</div>

## 1. 精度谱系与压缩比

| 精度 | 位宽 | 相对 FP16 压缩 | 典型损失 | 主战场 |
|------|------|---------------|----------|--------|
| FP16 / BF16 | 16 | 1× | 基准 | 训练、高保真推理 |
| FP8 (E4M3 / E5M2) | 8 | **2×** | ~0.3% | 权重+激活、KV Cache |
| INT8 | 8 | 2× | 略高于 FP8 | 边缘、老硬件 |
| INT4 (W4A16) | 4 | **4×** | ~0.5% | 端侧/板载部署，当前主流 |
| INT4 (W4A4) | 4 | 4× | 需旋转补偿 | 显存极限场景 |
| FP4 (NVFP4/MXFP4) | 4.5 | 3.6× | ~0.1%（原生硬件） | Blackwell 及之后 |

- **BF16 vs FP16**：BF16 指数位多、动态范围大，训练更稳；FP16 尾数更细但易溢出。
- **FP8 两种格式**：`E4M3`（4 指数 3 尾数，动态范围 ±448，适合权重/激活，代码数学友好）vs `E5M2`（5 指数 2 尾数，动态范围 ±57344，适合长文本/多语言）——选型直接决定掉点。细节见 **qt5 §3.1**。
- **为什么 FP4 是 4.5 bit**：微缩放格式每 16 或 32 个元素要共享一个 8 位 scale，摊到每元素约 0.5 bit。这 0.5 bit 换来的是动态范围，非常值——见 **qt5 §3.2**。

## 2. 量化方法谱系总表

这是本篇最核心的一张表。所有方法按「**它改了什么东西**」分四条路线：

| 路线 | 改什么 | 代表方法 | 位宽上限 | 计算成本 | 需校准 | 可叠加 |
|---|---|---|---|---|---|---|
| **A** | 只改**舍入** | RTN | W4A16 | 秒级 | 否 | 任意 |
| **A** | 只改**舍入** | GPTQ | W4A16 / W4A8 | 分钟~小时 | 是（Hessian） | AWQ / 旋转 |
| **B** | 改**分布·缩放** | SmoothQuant | **W8A8** | 秒级 | 是（act max） | GPTQ |
| **B** | 改**分布·缩放** | AWQ | W4A16 | 分钟级 | 是（mean\|X\|） | GPTQ |
| **C** | 改**分布·旋转** | QuaRot / SpinQuant / DuQuant | **W4A4** | 分钟级（+训练） | 是 / 需训练 | 可先旋转再 A/B |
| **D** | 改**训练** | FP4 QAT | W4A4 / FP4 | 小时~天 | 需训练栈 | 与上述正交 |
| **KV** | 正交维度 | KIVI / FP8 KV / KV 驱逐 | KV4~KV8 | 在线（极小） | K 需离线统计 | 与所有叠加 |

三句话把这张表读透：

```text
① A 路线（RTN/GPTQ）不改变数据分布，只在给定网格上让误差最小 → 上限由网格决定，通常 W4A16
② B/C 路线（缩放/旋转）改变数据分布，让网格更容易覆盖 → B 只能救激活，C 能同时救激活和 outlier → 上 W4A4
③ D 路线（QAT）让模型主动适应网格 → 突破 PTQ 的天花板，但要付出训练成本
```

**关键误区纠正**：很多资料说「AWQ 是 GPTQ 的替代品」，这是错的。它们的正交性很强——AWQ 决定 `s`（分布），GPTQ 决定 `round`（舍入）。生产配方 `AWQ 定 scale → GPTQ 定舍入` 能同时拿到两者收益（详见 **qt3 §4**）。

## 3. 四条路线的直觉对比

<div class="fig">
<svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="量化四条路线的直觉对比">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">同一组数据（含一个 outlier），四条路线怎么处理</text>

  <!-- 原始分布 -->
  <text x="20" y="52" font-size="11" font-weight="700" fill="#1a1a1a">原始激活 X（每列是全部元素，红柱是 outlier）</text>
  <g>
    <rect x="30" y="62" width="8" height="20" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="42" y="62" width="8" height="24" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="54" y="62" width="8" height="18" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="66" y="62" width="8" height="22" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="78" y="72" width="8" height="10" fill="#dc2626" stroke="#991b1b"/>
    <rect x="90" y="62" width="8" height="21" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="102" y="62" width="8" height="19" fill="#bfdbfe" stroke="#60a5fa"/>
    <rect x="114" y="62" width="8" height="23" fill="#bfdbfe" stroke="#60a5fa"/>
  </g>
  <text x="132" y="80" font-size="10" fill="#b91c1c">absmax = 118 → INT4 把其余全压成 0</text>

  <line x1="20" y1="98" x2="660" y2="98" stroke="#e5e7eb"/>

  <!-- A 路线 -->
  <text x="20" y="120" font-size="11" font-weight="700" fill="#1d4ed8">A · RTN / GPTQ：网格不动，改舍入</text>
  <g>
    <rect x="30" y="130" width="8" height="20" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="42" y="130" width="8" height="24" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="54" y="130" width="8" height="18" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="66" y="130" width="8" height="22" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="78" y="140" width="8" height="10" fill="#dc2626" stroke="#991b1b"/>
    <rect x="90" y="130" width="8" height="21" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="102" y="130" width="8" height="19" fill="#dbeafe" stroke="#60a5fa"/>
    <rect x="114" y="130" width="8" height="23" fill="#dbeafe" stroke="#60a5fa"/>
  </g>
  <text x="132" y="140" font-size="10" fill="#374151">outlier 还在 → 上限 W4A16（干脆不量化激活）</text>
  <text x="132" y="154" font-size="10" fill="#6b7280">GPTQ 用 Hessian 补偿敏感通道，但不改变"谁大"</text>

  <!-- B 路线 -->
  <text x="20" y="180" font-size="11" font-weight="700" fill="#7c3aed">B · SmoothQuant / AWQ：缩放到别的地方</text>
  <g>
    <rect x="30" y="190" width="8" height="22" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="42" y="190" width="8" height="26" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="54" y="190" width="8" height="20" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="66" y="190" width="8" height="24" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="78" y="190" width="8" height="16" fill="#f59e0b" stroke="#b45309"/>
    <rect x="90" y="190" width="8" height="23" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="102" y="190" width="8" height="21" fill="#ede9fe" stroke="#a78bfa"/>
    <rect x="114" y="190" width="8" height="25" fill="#ede9fe" stroke="#a78bfa"/>
  </g>
  <text x="132" y="200" font-size="10" fill="#374151">SQ：outlier 搬到权重（权重本来就均匀）</text>
  <text x="132" y="214" font-size="10" fill="#6b7280">AWQ：给显著通道放大权重，不搬移 → 上限 W4A16 / W8A8</text>

  <!-- C 路线 -->
  <text x="20" y="240" font-size="11" font-weight="700" fill="#15803d">C · QuaRot / SpinQuant / DuQuant：旋转摊平</text>
  <g>
    <rect x="30" y="250" width="8" height="20" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="42" y="250" width="8" height="21" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="54" y="250" width="8" height="19" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="66" y="250" width="8" height="22" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="78" y="250" width="8" height="20" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="90" y="250" width="8" height="21" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="102" y="250" width="8" height="19" fill="#bbf7d0" stroke="#4ade80"/>
    <rect x="114" y="250" width="8" height="22" fill="#bbf7d0" stroke="#4ade80"/>
  </g>
  <text x="132" y="260" font-size="10" fill="#374151">所有柱子一样高（能量均摊到 d 个通道）</text>
  <text x="132" y="274" font-size="10" fill="#15803d" font-weight="700">唯一能上 W4A4 的路线</text>

  <!-- D 路线 -->
  <text x="20" y="300" font-size="11" font-weight="700" fill="#ea580c">D · QAT：模型自己学会适应格子</text>
  <g>
    <rect x="30" y="310" width="8" height="21" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="42" y="310" width="8" height="22" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="54" y="310" width="8" height="20" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="66" y="310" width="8" height="23" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="78" y="310" width="8" height="18" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="90" y="310" width="8" height="21" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="102" y="310" width="8" height="22" fill="#ffedd5" stroke="#fb923c"/>
    <rect x="114" y="310" width="8" height="20" fill="#ffedd5" stroke="#fb923c"/>
  </g>
  <text x="132" y="320" font-size="10" fill="#374151">前向时伪量化，梯度看到真实误差</text>
  <text x="132" y="334" font-size="10" fill="#6b7280">outlier 在训练中被"自家吸收"，突破 PTQ 天花板</text>

  <!-- 底部对比 -->
  <rect x="20" y="352" width="640" height="38" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="32" y="369" font-size="10" fill="#1d4ed8" font-weight="700">A 不动分布 · B 搬移难点 · C 摊平尖峰 · D 训练适应</text>
  <text x="32" y="383" font-size="10" fill="#1d4ed8">四条路线全部正交，理论上可以四重叠加（生产里常见的是 B+C+A，即先旋转、再 AWQ 定 scale、最后 GPTQ 定舍入）</text>
</svg>
<p class="cap">图 1：同一组数据，四条路线的处理方式。注意只有 C 路线真正让"所有柱子一样高"——这就是它能做 W4A4 的原因。</p>
</div>

## 4. 训练后量化（PTQ）：不发梯度的压缩

PTQ 在已训模型上直接量化，核心是"怎么分配缩放因子"。四个主流方法，各归各路线：

- **RTN（舍入最近）**：逐元素 round-to-nearest，最快但大激活通道易被压坏。→ **qt2 §1**
- **GPTQ**：逐层 OBS 风格二阶量化——量化某列时用 Hessian 补偿其余列，最小化 `‖WX − ŴX‖²`。适合 GPU 上逐层离线压缩。→ **qt2 §2–3**
- **SmoothQuant**：`s = amax_X^α / amax_W^(1−α)`，把激活的 outlier 搬到权重。**只为 W8A8 而生**。→ **qt3 §2**
- **AWQ（激活感知）**：先算激活显著性 `s = mean(|X|)^α`，给重要通道"放大权重"再量化，几乎零额外开销。是当前 LLM 端侧部署主流。→ **qt3 §3**
- **per-group 量化**：`group_size=128` 是工业界默认粒度，在细粒度保真和查找表开销间取得平衡。→ **qt1 §2**

## 5. 旋转量化：W4A4 的支线

它会改变数据的**分布**而非**舍入**：对任意正交矩阵 `R`（`R Rᵀ = I`），有

```text
XW = (X R)(Rᵀ W)
```

乘积不变，但数值分布被"搅匀"——原本集中在少数通道的激活异常值（outlier）被摊平到所有通道，4-bit 量化不再被 outlier 撑爆动态范围。

三个代表：

- **QuaRot（NeurIPS 2024）**：随机化 Hadamard + 随机置换，在四个位置（R1 残差流 / R2 attention out / R3 SwiGLU 后 / R4 Q·K）施加旋转，首次在 LLaMA-2/3 上做到 **W4A4KV4** 近乎无损，无需训练。
- **SpinQuant（arXiv:2405.16406）**：把旋转从"随机固定"升级为"可学习"——用 Cayley 变换保证正交给约束，在 Stiefel 流形上优化，LLaMA-3 上比 QuaRot 多挽回显著精度。
- **DuQuant**：指出全局 Hadamard 会**制造新的 outlier**，改用块对角旋转 + zigzag 通道置换，在小模型和顽固 outlier（LLaMA-3、Qwen2）上比前两者更稳。

<div class="warnbox">
<strong>一个重要的现实修正：</strong>旋转路线在通用 GPU 上的收益是<strong>省显存</strong>而非提速度——因为 H100/H200 的 Tensor Core 没有原生 INT4×INT4 MMA，W4A4 只能"解包再算"，吞吐上并不比 W4A16 快。只有 Blackwell 的原生 NVFP4 才真正解锁 W4A4 的算力优势。完整分析见 <strong>qt4 §4</strong> 与 <strong>qt5 §3.3</strong>。
</div>

> VLA 实战：OpenVLA 做 INT4 量化后约 4GB，成功率近乎无损，可在 Jetson Orin 上跑，给感知/规划/控制栈留出余量——这是"大模型进物理世界"的必经之路。

## 6. 量化感知训练（QAT）：把误差放进反向传播

PTQ 到 INT4/FP4 损失开始变大，需要 QAT：

- **FP4 QAT（DeepSeek V4 / V4-Flash 路线）**：对 MoE 的 routed expert 权重和 CSA indexer 的 QK 路径做 FP4 伪量化——**训练时伪量化到 FP4 再反量化回高精度计算**，采样/rollout 用 native FP4。这样梯度看到的是量化后的真实分布，部署即所见。
- 前置基础：混合精度训练（BF16 AMP）+ FP8 KV Cache。QAT 往往和 Muon 优化器、SwiGLU Clamping 等工程化技巧配合，才能把 4-bit 的掉点压到可用区间。

## 7. KV Cache 量化与驱逐：长上下文的显存救命线

长上下文推理的主要瓶颈不是算力而是 **KV Cache 显存**。三条正交路径：

- **量化（压精度）**：FP16→FP8 2× ~0.3% 损失；FP16→INT4 需注意 **K 比 V 敏感一个量级**（softmax 指数放大），KIVI 的方案是 K per-channel + V per-token + 保留最近 128 token 全精度。→ **qt5 §1–2**
- **驱逐（减 token 数）**：H2O / ThinKV 等按重要性裁剪缓存，可再砍 3~244×。
- **叠加前沿**：RDKV（2026）率失真统一量化+驱逐，2.48% 缓存→97.81% 准确率；ThinKV（ICLR 2026 Oral）思想自适应 KV 压缩，<5% KV 保留近乎无损、5.8× 吞吐。
- **坑**：E4M3 vs E5M2 选型、INT2 实用下限、量化+驱逐叠加需调参、H2O 额外 5~10% 计算开销。→ **qt5 §1.3** 的 PagedAttention block 对齐约束

## 8. 落地到推理引擎与推测解码

- **vLLM / SGLang**：FP8 KV Cache + AWQ 部署是量化在推理引擎的落地点；HiSparse（见本博客 sys6）进一步把 HBM 当缓存、主机 DRAM 存权威全量 KV，是"量化+分层"思路在系统层的延伸。
- **推测解码**：量化给投机解码腾出显存余量——KV Cache 压缩后 batch 更大、draft model 可更重，端到端加速更明显。
- **训练侧**：Megatron-LM `--fp8-format hybrid` + BF16 AMP 是当前大模型训练的常态配置。

## 9. 部署选型清单

1. **先定位瓶颈**：显存墙 → KV Cache 量化/驱逐；带宽墙 → 权重 INT4/AWQ；算力墙 → FP8 前向。
2. **看硬件定格式**：A100 → INT8/AWQ W4A16；H100 → FP8（不需要 SmoothQuant）；B200 → NVFP4（不需要旋转）。**硬件决定格式，格式决定算法。**
3. **精度优先 FP8**（2×、~0.3% 损失），端侧/板载再下探 INT4（AWQ，4×）。
4. **W4A4 只在显存极限且必须时用**，通用 GPU 上没有速度收益。
5. **FP4 优先原生硬件**（B200），否则必须配 QAT。
6. **长文本选 E5M2、代码/数学选 E4M3**。
7. **INT2 以下基本不可用**，别为了噱头牺牲可用性。

**串起来看**：蒸馏缩容量、量化压精度、KV 驱逐砍长度、分层缓存（HiSparse）把三者放进同一显存预算——这就是今天把 70B+ 模型塞进单卡长上下文服务的完整工具箱。

*注：本文整合自 2026-07-18（KV Cache 量化与驱逐）、2026-08-01（FP8/INT4 部署）、2026-08-04（FP4 QAT）的每日技术追踪；2026-10-27 拆出「量化手记」五篇独立展开，本篇改为导读与谱系总表。*
