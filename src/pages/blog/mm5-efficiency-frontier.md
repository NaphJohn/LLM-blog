---
title: 多模态解码手记（五）：高效系统与前沿推理——注意力 / 推理优化 / 端侧 / RL
description: 收官篇把视野拉到底座：支撑所有多模态模型的"优化手段"。从注意力五件套（FlashAttention/RoPE/GQA/MLA/RMSNorm），到高效训练与推理系统（混合精度/KV Cache/PagedAttention/PD分离/投机解码/SwiGLU），再到端侧智能（LoRA/QLoRA/On-Device深研/EcoSpec）与前沿推理RL（OAT/Ring-Zero/SPS/HiLS-Attention/探索悖论/AMVL）。
pubDate: 2026-07-31
series: 多模态解码手记
lang: zh
altLang: en
altHref: /en/blog/mm5-efficiency-frontier
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么需要这一章

（一）~（四）讲的是多模态"能做什么"；（五）讲的是"怎么让它跑得起、跑得快、跑在端侧"。**推理成本每降一档，VLA + 世界模型（四章）就多一分端侧实时可行**。这一章把它拆成四层。

## 1. 注意力五件套：主流 Decoder LLM 的标配

- **FlashAttention（FA-1→FA-4）**：分块 + 在线 softmax + 算子融合；N×N 注意力矩阵始终留在 SRAM，不落 HBM。bit-exact（非近似），A100 上 2~4× 加速、128K 上下文不 OOM；H100 FA-3 FP8 达 740 TFLOPs/s。Qwen3/DeepSeek V3-V4/MiniMax/GLM/Kimi/Llama 5 全默认装配。
- **RoPE（旋转位置编码）**：在 2D 子空间旋转 Q/K 编码**相对位置**，使注意力只依赖偏移 `m−n`。`θ_i = 10000^(−2i/d)`；零参数显式相对位置 + NTK/YaRN 扩长上下文；**V 不旋转**是常见坑。Qwen3/DeepSeek V4/MiniMax/Llama5/GLM/Kimi 在用。
- **GQA（分组查询注意力）**：H 个 Query Head 分 G 组、**组内共享 KV Head**，4~8× 压缩 KV Cache。G=8 甜点（质量损失 < 0.5%）。Llama2/3、Qwen 全系、Mistral、DeepSeek V2-V4、MiniMax 在用。
- **MLA（多头潜在注意力）**：把 K/V **联合压成低维潜向量 c**，只缓存 c，使用时上投影还原 → 显存降至 1/4~1/10 几乎不掉精度。**需解耦 RoPE**。DeepSeek V2/V3/V4 在用。
- **RMSNorm**：LayerNorm 简化版（去均值中心化与 β），约 7~15% 加速、效果持平，对端侧 VLA 部署 latency 优化直接有价值。Llama/Qwen3/DeepSeek V4/MiniMax/Mistral/Gemma 在用。

> 长上下文 / 低成本推理的胜负手在 KV Cache：GQA/MLA"省显存"、FlashAttention"省 IO"、RMSNorm"省计算"——三者叠加构成今天的效率基线。

## 2. DeepSeek-V4 架构深读：压缩 KV Cache 的四级演进（MLA → NSA → DSA → CSA+HCA）

> 把（一）的"注意力五件套"落到一个旗舰模型上：DeepSeek 用三代（V2→V3→V4）把"KV Cache 压缩"做成了工程主线。理解这条线，就理解了长上下文推理为何能降本一个数量级——这也是评估任何长上下文/低成本推理标的的技术锚点。

### 2.1 一条主线：把 KV Cache 压到极限

KV Cache 是长上下文推理的成本锚点：每多一个 token，就要多存一份 K/V。DeepSeek 的压缩有**两个正交方向**：

- **压宽度（MLA）**：把高维 K/V 压成低维潜向量 `c`，只缓存 `c`，用时上投影还原 → 显存降到 1/4~1/10、几乎不掉精度，但需解耦 RoPE（V2/V3/V4 在用）。
- **压长度（NSA→CSA+HCA）**：不再缓存每个 token 的 K/V，而是只保留"被压缩后的少量记忆块" → token 数本身下降，长度维度被压缩。

图1 给出两个方向的对照。

<figure class="fig">
<svg viewBox="0 0 680 220" role="img" aria-label="KV Cache 两个压缩方向">
  <defs>
    <marker id="arr1" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--fg)"></path>
    </marker>
  </defs>
  <text x="165" y="20" text-anchor="middle" fill="var(--fg)" font-size="14" font-weight="600">压宽度 · MLA</text>
  <g fill="var(--accent)" opacity="0.85">
    <rect x="36" y="38" width="13" height="116"></rect>
    <rect x="54" y="38" width="13" height="116"></rect>
    <rect x="72" y="38" width="13" height="116"></rect>
    <rect x="90" y="38" width="13" height="116"></rect>
    <rect x="108" y="38" width="13" height="116"></rect>
  </g>
  <text x="93" y="172" text-anchor="middle" fill="var(--muted)" font-size="11">每个 token 全量 K/V</text>
  <path d="M132 96 H176" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr1)"></path>
  <rect x="186" y="88" width="22" height="18" rx="4" fill="var(--accent)"></rect>
  <text x="197" y="130" text-anchor="middle" fill="var(--fg)" font-size="13">c</text>
  <text x="197" y="150" text-anchor="middle" fill="var(--muted)" font-size="10">低维潜向量</text>

  <line x1="340" y1="28" x2="340" y2="186" stroke="var(--border)"></line>
  <text x="510" y="20" text-anchor="middle" fill="var(--fg)" font-size="14" font-weight="600">压长度 · NSA→CSA+HCA</text>
  <g fill="var(--accent)" opacity="0.5">
    <rect x="356" y="50" width="16" height="16"></rect>
    <rect x="378" y="50" width="16" height="16"></rect>
    <rect x="400" y="50" width="16" height="16"></rect>
    <rect x="422" y="50" width="16" height="16"></rect>
    <rect x="444" y="50" width="16" height="16"></rect>
    <rect x="466" y="50" width="16" height="16"></rect>
    <rect x="488" y="50" width="16" height="16"></rect>
    <rect x="510" y="50" width="16" height="16"></rect>
    <rect x="532" y="50" width="16" height="16"></rect>
    <rect x="554" y="50" width="16" height="16"></rect>
    <rect x="576" y="50" width="16" height="16"></rect>
    <rect x="598" y="50" width="16" height="16"></rect>
    <rect x="620" y="50" width="16" height="16"></rect>
  </g>
  <text x="488" y="86" text-anchor="middle" fill="var(--muted)" font-size="11">全部 token 细粒度 K/V</text>
  <path d="M510 110 H548" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr1)"></path>
  <g fill="var(--accent)">
    <rect x="556" y="100" width="26" height="22" rx="4"></rect>
    <rect x="588" y="100" width="26" height="22" rx="4"></rect>
    <rect x="620" y="100" width="26" height="22" rx="4"></rect>
  </g>
  <text x="606" y="146" text-anchor="middle" fill="var(--fg)" font-size="11">少量记忆块</text>
</svg>
<figcaption>图1　KV Cache 的两个压缩方向：压宽度（MLA，把每列的 K/V 压成细 latent c）与压长度（NSA→CSA+HCA，把多列合并成少数记忆块）。两者正交、可叠加。</figcaption>
</figure>

### 2.2 四级演进

| 阶段 | 代号 | 做法 | 省了什么 | 备注 |
|---|---|---|---|---|
| ① | MLA | K/V→低维潜向量 `c` | 压**宽度** | V2/V3/V4 基线 |
| ② | NSA（2025.02） | 三分支 cmp/slc/win，原生可训练 | 压**长度** | 硬件对齐（GQA 式分组，算术强度均衡） |
| ③ | DSA（V3.2 过渡） | Lightning Indexer | 省**算力** | 省算不省存，承上启下 |
| ④ | CSA+HCA（V4） | 三级记忆 | 压**长度**到极致 | 1M 上下文仅 ~7800 记忆 |

- **NSA（Native Sparse Attention）**：compressed / selected / window 三分支**原生可训练**；按 GQA 式分组做算术强度均衡，能在 Tensor Core 上跑满——不是"后处理剪枝"，而是训练时就稀疏。
- **DSA（V3.2 过渡）**：加入 Lightning Indexer，用轻量打分快速挑出重要 token；**省算力但 KV 仍要存**（"省算不省存"），是通往 V4 的桥梁。
- **CSA+HCA（V4 三级记忆）**：把"长短记忆"显式分成三级（见 2.3）。

### 2.3 V4 三级记忆：SWA + CSA + HCA

<figure class="fig">
<svg viewBox="0 0 680 280" role="img" aria-label="V4 三级记忆">
  <defs>
    <marker id="arr2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--fg)"></path>
    </marker>
  </defs>
  <text x="14" y="20" fill="var(--fg)" font-size="13" font-weight="600">V4 三级记忆（越久越粗，越近越细）</text>

  <text x="14" y="58" fill="var(--accent)" font-size="13" font-weight="600">SWA 短期</text>
  <g fill="var(--accent)">
    <rect x="120" y="44" width="13" height="13"></rect>
    <rect x="137" y="44" width="13" height="13"></rect>
    <rect x="154" y="44" width="13" height="13"></rect>
    <rect x="171" y="44" width="13" height="13"></rect>
    <rect x="188" y="44" width="13" height="13"></rect>
    <rect x="205" y="44" width="13" height="13"></rect>
    <rect x="222" y="44" width="13" height="13"></rect>
    <rect x="239" y="44" width="13" height="13"></rect>
    <rect x="256" y="44" width="13" height="13"></rect>
  </g>
  <text x="290" y="56" fill="var(--muted)" font-size="11">滑动窗口 n_win = 128（细粒度局部 KV）</text>

  <text x="14" y="118" fill="var(--accent)" font-size="13" font-weight="600">CSA 中期</text>
  <g fill="var(--accent)" opacity="0.6">
    <rect x="120" y="104" width="18" height="18"></rect>
    <rect x="144" y="104" width="18" height="18"></rect>
    <rect x="168" y="104" width="18" height="18"></rect>
    <rect x="192" y="104" width="18" height="18"></rect>
  </g>
  <path d="M220 113 H250" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr2)"></path>
  <rect x="258" y="104" width="22" height="18" rx="4" fill="var(--accent)"></rect>
  <text x="296" y="118" fill="var(--muted)" font-size="11">4→1 压缩 + Lightning Indexer 取 top-k = 1024</text>

  <text x="14" y="178" fill="var(--accent)" font-size="13" font-weight="600">HCA 长期</text>
  <rect x="120" y="162" width="120" height="24" rx="4" fill="var(--accent)" opacity="0.25"></rect>
  <text x="180" y="178" text-anchor="middle" fill="var(--muted)" font-size="11">128 个 chunk</text>
  <path d="M248 174 H278" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr2)"></path>
  <rect x="286" y="162" width="26" height="24" rx="4" fill="var(--accent)"></rect>
  <text x="324" y="179" fill="var(--fg)" font-size="11">128→1 稠密（1M→~7800）</text>

  <text x="14" y="250" fill="var(--muted)" font-size="11">三级叠加：1M token 最终仅约 7800 个记忆单元（≈ 原 KV 的 0.8%）</text>
</svg>
<figcaption>图2　V4 三级记忆：短期 SWA（窗口 n_win=128）→ 中期 CSA（4 个 chunk 压成 1 个 + Lightning Indexer 取 top-k=1024）→ 长期 HCA（128→1 稠密，1M 上下文压到约 7800 个记忆）。</figcaption>
</figure>

- **SWA（Sliding Window Attention，短期）**：仅看最近 `n_win=128` 个 token 的细粒度 KV，覆盖局部依赖。
- **CSA（Compressed Sliding Attention，中期）**：把每 4 个 chunk 压成 1 个压缩块，再用 Lightning Indexer 从全局长上下文里挑 `top-k=1024` 个最关键位置 → 中期记忆既压缩又"有选择"。
- **HCA（Heavy-Cluster Attention，长期）**：把 128 个块稠密压成 1 个"长期记忆头"，1M token 上下文最终只留约 **7800** 个记忆单元（≈ 原 KV 的 0.8%）。

> 三级记忆的妙处：越久越远的信息越"粗"，越近越"细"。近处要精确、远处只要语义——这正是对人记忆的建模，也让 1M 上下文的推理成本可控。

### 2.4 两条产品线：V4-Pro 与 V4-Flash

| 指标（相对稠密基线） | V4-Pro | V4-Flash |
|---|---|---|
| 计算量 FLOPs | ≈ 27% | ≈ 10% |
| KV 显存 | ≈ 10% | ≈ 7% |

V4-Flash 把计算量压到约 1/10、KV 压到约 1/14，是面向"极致吞吐 / 边缘部署"的版本；V4-Pro 在效果与成本间取更平衡的点。

### 2.5 推理侧：Mooncake 与 DistServe

压缩把"单卡能装下"解决了，但**高并发长上下文**还要靠 PD 分离。

- **Mooncake（KV-Cache 中心化的 P/D 分离）**：把 KV Cache 抽成一个共享池，Prefill 实例算完把 KV 存进池，Decode 实例按需取用，显存跨实例复用、不被重复算。

<figure class="fig">
<svg viewBox="0 0 680 250" role="img" aria-label="Mooncake P/D 分离">
  <defs>
    <marker id="arr3" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--fg)"></path>
    </marker>
  </defs>
  <rect x="30" y="40" width="120" height="40" rx="6" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="90" y="65" text-anchor="middle" fill="var(--fg)" font-size="13">Prefill 实例 ×N</text>
  <rect x="30" y="90" width="120" height="40" rx="6" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="90" y="115" text-anchor="middle" fill="var(--fg)" font-size="13">Prefill 实例</text>

  <rect x="250" y="55" width="180" height="120" rx="10" fill="var(--accent)" opacity="0.12" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="340" y="100" text-anchor="middle" fill="var(--fg)" font-size="13" font-weight="600">KVCache 池</text>
  <text x="340" y="122" text-anchor="middle" fill="var(--muted)" font-size="11">KV 只算一次</text>
  <text x="340" y="140" text-anchor="middle" fill="var(--muted)" font-size="11">跨实例复用</text>

  <rect x="530" y="40" width="120" height="40" rx="6" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="590" y="65" text-anchor="middle" fill="var(--fg)" font-size="13">Decode 实例 ×M</text>
  <rect x="530" y="90" width="120" height="40" rx="6" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="590" y="115" text-anchor="middle" fill="var(--fg)" font-size="13">Decode 实例</text>

  <path d="M150 60 H248" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr3)"></path>
  <path d="M432 100 H528" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr3)"></path>
  <text x="340" y="200" text-anchor="middle" fill="var(--muted)" font-size="11">以 KVCache 为中心的 P/D 分离：显存跨实例共享，不被重复计算</text>
</svg>
<figcaption>图3　Mooncake：以 KVCache 池为中心的 Prefill/Decode 分离，KV 只算一次、多处复用。</figcaption>
</figure>

- **DistServe（Goodput，OSDI'24）**：把 Prefill 与 Decode 拆到**不同资源池**，各自按自己的 SLO（TTFT / TPOT）优化，用 "Goodput"（满足 SLO 的有效 token 数）而非吞吐来度量——避免"为了高吞吐牺牲首 token 延迟"的坑。

<figure class="fig">
<svg viewBox="0 0 680 220" role="img" aria-label="DistServe">
  <defs>
    <marker id="arr4" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="var(--fg)"></path>
    </marker>
  </defs>
  <rect x="40" y="50" width="200" height="80" rx="8" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="140" y="85" text-anchor="middle" fill="var(--fg)" font-size="13" font-weight="600">Prefill 池</text>
  <text x="140" y="108" text-anchor="middle" fill="var(--muted)" font-size="11">优化 TTFT</text>

  <rect x="440" y="50" width="200" height="80" rx="8" fill="none" stroke="var(--accent)" stroke-width="1.5"></rect>
  <text x="540" y="85" text-anchor="middle" fill="var(--fg)" font-size="13" font-weight="600">Decode 池</text>
  <text x="540" y="108" text-anchor="middle" fill="var(--muted)" font-size="11">优化 TPOT</text>

  <path d="M240 90 H438" stroke="var(--fg)" stroke-width="2" marker-end="url(#arr4)"></path>
  <text x="340" y="82" text-anchor="middle" fill="var(--muted)" font-size="11">KV 传递</text>

  <rect x="200" y="160" width="280" height="34" rx="6" fill="var(--accent)" opacity="0.12"></rect>
  <text x="340" y="182" text-anchor="middle" fill="var(--fg)" font-size="12">目标：Goodput = 满足 SLO 的有效 token 数</text>
</svg>
<figcaption>图4　DistServe：Prefill 池与 Decode 池分离，分别优化 TTFT 与 TPOT，用 Goodput（满足 SLO 的有用 token）做目标。</figcaption>
</figure>

> 端到端成本 = 压缩（MLA/NSA/CSA+HCA 降显存+降 FLOPs）× 调度（Mooncake/DistServe 提并发）。DeepSeek V3 全量分离栈约 545 output tok/s/GPU，正是这俩乘出来的结果——**这也是"推理降本→云推理厂商/轻量 VLA 部署门槛下降"投资逻辑的技术底座**。

## 3. 高效训练与推理系统

- **混合精度训练（FP16/BF16/FP8 Hybrid）**：仅 GEMM 输入用低精度（显存减半），master weights 保持 FP32。FP16 需 Gradient Scaling 防下溢；BF16 指数位同 FP32 无需 scaling；FP8 Hybrid = E4M3 前向 + E5M2 反向（H100+）。DeepL 172B 吞吐 400→550 TFLOP/s（1.4×）。
- **KV Cache 量化与驱逐**：量化压精度（2~4×）、驱逐削 token 数（3~244×）。vLLM/SGLang 默认 FP8 KV（0.3% 损失）；DeepSeek V4 MLA+FP8 双重压缩，1M 上下文仅 ~20GB；RDKV（2026）2.48% 缓存→97.81% 准确率；ThinKV（ICLR 2026 Oral）<5% KV 近乎无损、5.8× 吞吐。
- **PagedAttention**：KV Cache 按 16-token block 分页，内存利用率 20~40%→95%+，并发 2~4×。vLLM（SOSP 2023）、SGLang；也是"多机器人并发 VLA 推理"的底层支撑。
- **PD 分离（Prefill-Decode Disaggregation）**：Prefill（算力密集）与 Decode（访存密集）拆到不同实例，降 TTFT。vLLM/SGLang/Mooncake/DistServe；DeepSeek V3 全量分离栈 ~545 output tok/s/GPU；**仅在长 prompt + 长输出 + 高并发时回本**。
- **投机解码（EAGLE / MTP）**：草稿-验证循环，draft 先猜 K token、主模型一次前向验证，严格不改输出分布（lossless）。DeepSeek V3 H200 batch=1 40→60 tok/s（1.8×）；EAGLE-3 在 Llama-3 8B 最高 6.5×。**坑：接受率必须监控（< 30% 应关掉）；INT4 主 + FP16 草稿精度不匹配会暴跌。**
- **SwiGLU（门控 FFN）**：双支路 `SiLU(xW_gate) ⊙ (xW_val) · W_down` 引入逐元素门控。参数量持平 ReLU FFN，但 `d_ff` 须改为 `(8/3)d`。Llama/Qwen/DeepSeek V4/GLM-5/MiniMax/Gemma4/Mistral 在用。

> 推理成本 = 显存（GQA/MLA + 量化 KV）+ IO（FlashAttention）+ 调度（PagedAttention + PD 分离）+ 解码效率（投机解码）四层叠加。评估推理栈时看是否"四层全开"。

## 4. 端侧智能：小资源也能用大模型

- **LoRA / QLoRA（PEFT）**：冻结 `W₀`、只训低秩增量 `ΔW = B×A`（r≪d），微调成本降 50~100×。QLoRA 叠加 4-bit NF4 + 双重量化，让 65B 单卡可微调；社区用 QLoRA 微调 MiniMax-8B 成本 < $20。LLaVA 自身就用 LoRA（仅 0.5% 参数）；Qwen3.5 推荐 `r=64 + DoRA`。把"大模型定制"从土豪游戏变成平民技能。
- **On-Device Deep Research at 4B（arXiv:2607.12257）**：**4B 小模型也能做"深度研究"**——用"暴露边界"（faithfulness）管幻觉、用"检索覆盖边界"管遗漏。意味着**端侧 / 本地就能跑带检索的研究助手**，利好端侧芯片 / 边缘算力叙事。
- **EcoSpec：面向 MoE 的成本感知投机解码（arXiv:2607.12696）**：MoE 里**专家分散**带来隐性激活成本；在草稿阶段把激活成本纳入考量、减少专家搬运 → 不牺牲质量下更快。利好 DeepSeek-V3/V4 这类稀疏 MoE 的落地推理。
- **测试时计算（Test-Time Compute）**：把"够好的模型"放本地跑（4B 深研、端侧 VLA），用多步验证 / 自进化验证器弥补参数不足。省云成本、保隐私、降时延——**机器人端侧实时推理正依赖此路线**。

## 5. 前沿推理与强化学习（2026 夏）

- **OAT：从"成功之流"溯源 Agent 失败（arXiv:2607.12747）**：用神经微分方程 NCDE 从成功轨迹反学"失败发生在哪一步"，比行为监控更早发现轨迹走偏。可迁移到"机器人操作失误定位"——提升自主系统可调试性。
- **Ring-Zero：Zero-RL 推到 1T（人大×蚂蚁, 2607.12395）**：把 Zero-RL（无需 SFT 冷启动、直接从 RL 学推理）推到 1T 参数时，**高级思维策略自发涌现**，弱化对海量 SFT 数据的依赖。
- **SPS：状态-预测分离（康奈尔×哈佛, 2607.01218）**：表征与预测解耦训练，效率 2.6×，更易复用与缩放——高效世界模型 / 预言机的工程思路。
- **HiLS-Attention：8K 训练外推 4M（腾讯混元, 2607.02980）**：注意力改进使 8K 训练可外推 4M，prefill 提速 13.5×，已开源。长上下文"训短用长"的性价比路线。
- **探索悖论 RL（字节 Seed×MSU, 2607.06987）**：修正探索中的重要性采样偏差，缓解长程推理 RL 的熵崩溃 / 探索不足。
- **AMVL：潜空间连续推理（上交×蚂蚁, 2607.00461）**：在潜空间做连续推理，BLINK 基准 +10.83，呼应"扩散 / 连续表征做推理"路线。

> 这组工作共同指向：**推理能力正从"堆参数"转向"更好的训练信号 + 更好的长上下文 / 潜空间表征"**。

## 6. 全系列收束

```
（一）基础与对齐范式  → 多模态=把生成空间从文本外推到图像/动作
（二）VLM 演进        → ViT/CLIP/LLaVA：看+说的标准范式
（三）生成模型        → 扩散/GAN：画出来的那一支
（四）VLA+世界模型    → RT-2→π0.7→HiF-VLA：动手+先想再动
（五）高效+前沿       → 注意力/推理优化/端侧/前沿RL：跑得起跑得快的底座
```

**投资视角闭环**：大模型降本（本篇）↔ 具身实时可行（四篇）↔ 端侧芯片 / 边缘算力（本篇端侧）是同一逻辑链上的三个支点。评估任何一家多模态 / 具身公司，都可以沿这条"基础范式 → 模型 → 生成 → 行动 → 效率"的轴线逐项打分。
