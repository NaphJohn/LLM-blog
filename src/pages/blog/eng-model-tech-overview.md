---
title: 模型技术总览 · 结构与原理解读
description: 从五大主题速览大模型技术栈：架构、注意力、高效计算、生成模型与推理加速。
pubDate: 2026-07-31
series: 大模型工程实践
lang: zh
layout: ../../layouts/BlogPost.astro
---

> 文档定位：把近期模型方向的 5 大主题，从**结构（由哪些模块组成、怎么串起来）**到**原理（关键机制/公式/为什么有效）**统一讲清楚，并标注每个主题对应的**已有素材**（svg 图、知乎科普、长文档、AI 知识库条目），方便直接跳读或继续深化。
> 这是 **A 形态（统一总览）**。另有 **B 形态**（6 篇独立「结构+原理」短文档，已归入 `模型技术库/` 下各分类目录）与 **C 形态**（资产索引 + 知识地图，见 `模型技术资产索引与知识地图.md`）。
> 数据来源：工作区 `AI知识库.md`、`知乎科普/`、`模型技术库/06_大模型与多模态/`（`Llama5_技术架构与演进.md`、`Step3-VL_多模态大模型.md`、`国内外最强大模型技术架构_2026.md`）及各 svg 图，截止 2026-07。

---

## 〇、五大主题速览

| # | 主题 | 一句话定位 | 结构核心 | 原理关键词 | 对应素材 |
|---|------|-----------|----------|-----------|----------|
| 1 | **Stable Diffusion / LDM 潜扩散** | 把图像生成从像素空间搬到「潜空间」压缩后再扩散 | VAE + U-Net + CLIP(Cross-Attn) 三件套 | 感知压缩、DDPM、CFG、潜空间去噪 | 架构图+去噪动画已内嵌 `04_生成模型/StableDiffusion_LDM_潜扩散模型.md`、`知乎科普/2026-07-16_StableDiffusion.md` |
| 2 | **FlashAttention（FA-1→FA-4）** | 让标准 Attention 不必物化 N×N 大矩阵，化解「显存墙」 | 分块入 SRAM + 在线 softmax + 算子融合 | IO 感知、tile、m/l/o 三状态、bit-exact | `算子讲解_sent.md`、`att_b64_*`、`AI知识库.md` |
| 3 | **Speculative Decoding（EAGLE / MTP）** | 草稿–验证循环，无损加速逐 token 解码 | draft 模型 + target 验证 + 接受率 | 并行验证、接受率 α、EAGLE-3、MTP、tree-attn | `知乎科普_投机解码与SSM状态.md`、`AI知识库.md` |
| 4 | **Transformer / BERT / ViT** | 统一序列建模架构，跨语言/视觉通用 | Encoder-Decoder / Encoder-only / Patch+Encoder | 自注意力、位置编码、MLM/NSP、patch token | 三张架构图已内嵌 `01_基础架构/Transformer_BERT_ViT.md`、`知乎科普/2026-07-15_ViT.md` |
| 5 | **Llama5 / Step3-VL / 国内外大模型架构** | 2026 前沿大模型架构全景与多模态落地 | MoE / VLM(Encoder-Projector-Decoder) / 多厂商格局 | MoE 路由、RSI、投影对齐、PaCoRe、原生多模态 | `模型技术库/06_大模型与多模态/Llama5_技术架构与演进.md`、`模型技术库/06_大模型与多模态/Step3-VL_多模态大模型.md`、`模型技术库/06_大模型与多模态/国内外最强大模型技术架构_2026.md` |

> 阅读建议：想快速通览看本文件；想逐主题深挖看 B；想知道「我手里到底有哪些可复用素材」看 C。

---

## 一、Stable Diffusion / LDM 潜扩散模型

### 1.1 定位
并非在像素上「画画」，而是先在 VAE 压缩后的**潜空间**（4×64×64，相对像素 3×512×512 压缩约 **48×**）做扩散，再把结果解压回高清图。这让扩散模型能在消费级 GPU（如 4090）十几秒出图，是开源文生图的「地基」。

### 1.2 结构：三件套
```
文本 prompt ──CLIP Text Encoder──┐
                                  ├─► Cross-Attention 注入 ─► U-Net εθ（潜空间去噪）
像素图像 ──VAE Encoder──► 潜码 z ─┘                                      │
                                                                         ▼
                                                               VAE Decoder ─► 高清图
```
- **VAE**：感知压缩器，像素↔潜码双向。
- **U-Net ε_θ**：在潜空间执行加噪/去噪循环的去噪网络，通过 Cross-Attention 接收文本条件。
- **CLIP 文本编码器**：把文字翻成提示向量 τ(c)。
- 配图：架构图与去噪动画已 base64 内嵌于 `04_生成模型/StableDiffusion_LDM_潜扩散模型.md`（见原文「整体结构」与「动画演示」两节，自包含不依赖独立 svg 文件）。

### 1.3 原理（关键公式）
- 训练目标（LDM）：`L_LDM = E[ ‖ε − ε_θ(z_t, t, τ(c))‖² ]`
- 无分类器引导（CFG）：`ε̂ = ε_u + w·(ε_c − ε_u)`，**w∈[5,12]** 甜区；w 越大越贴 prompt 但多样性↓、易过饱和。
- 训练技巧：条件 dropout **10%**（CFG 成立前提）；SD1.x 平方余弦调度 → SDXL EDM → SD3 rectified flow。
- 影响：SDXL/SD3/HunyuanDiT/Wan-Video/FLUX 全沿用此模板；LoRA/ControlNet/IP-Adapter 都建在 LDM 之上。

### 1.4 与素材对应
- 架构图与去噪动画：已 base64 内嵌于 `04_生成模型/StableDiffusion_LDM_潜扩散模型.md` 正文（自包含，不再依赖独立 svg 文件）。
- 通俗版：`知乎科普/2026-07-16_StableDiffusion.md`（含「工程哲学」串联）。
- 知识库条目：`AI知识库.md` 的 `2026-07-16` 段。

### 1.5 与其它主题关联
- 与 **FlashAttention**：SD 用 U-Net（不存 KV），与 FA 无直接交集；但下游文生视频（Wan/SD3 视频版）已逐步接入 MLA 思路压缩 Cross-Attn 上下文。
- 与 **大模型架构**：「先找好的压缩通道再生成」这一哲学，与 LLM 的 MoE（参数压缩）、MLA（KV 压缩）同构。

---

## 二、FlashAttention（FA-1 → FA-4）

### 2.1 定位
**IO 感知 + 在线 softmax + 算子融合**，让标准 Attention 不必物化 N×N 注意力矩阵，化解「显存墙」。它是今天几乎所有前沿大模型（Qwen3/3.5、DeepSeek V3/V4、MiniMax、GLM、Kimi、Llama 5）的标配算子。

### 2.2 结构
```
标准 Attention：  Q·K^T → /√d → softmax → ·V     （N×N 矩阵需物化在 HBM，巨慢）
FlashAttention：  把 Q/K/V 切 128×128 小方块 → 每块载入 SRAM → 算完即丢，全程不在 HBM 物化大表
```
- 不依赖外部大矩阵，显存从 O(N²) 降到 O(N)。

### 2.3 原理（关键机制）
分块（tiling）入 SRAM 后，跨块维护三个在线状态保证结果与朴素实现 **bit-exact（不是近似！）**：
- `m_i`：当前块的行最大值（max）
- `l_i`：分母累加器（running denominator）
- `o_i`：输出累加器（output accumulator）

在线 softmax 用递推方式更新上述三值，数学上等价于先算全表再 softmax。

### 2.4 收益与在用模型
- A100 上 **2~4×** 加速、长序列 **128K** 不 OOM；H100 FA-3 FP8 + 异步流水线吞吐达 **740 TFLOPs/s**。
- 使用者：Qwen3/3.5（FA-2/3+GQA）、DeepSeek V3/V4（FA-2/3+MLA）、MiniMax（Lightning+FA）、GLM/Kimi/Llama 5（FA-3 BF16/FP8）。

### 2.5 坑
自定义 mask 触发慢路径；FP8 长序列需 block-wise scaling；旧 GPU（A100 前）只能用 FA-1。

### 2.6 与素材对应
- 算子记录：`算子讲解_sent.md`（已讲解：MLA、FlashAttention）。
- 图资源：`att_b64_oneline.txt` / `att_b64_wrapped.txt` / `b64.txt`（注意力相关 base64，可转图嵌入）。
- 知识库：`AI知识库.md` `2026-07-16` 段「当日算子」。

### 2.7 与其它主题关联
- 与 **Speculative Decoding**：在 vLLM/SGLang 深度协同——FA-3 加速主模型前向，Spec Decoding 提升单请求 token/s，叠加是当前最优推理栈。
- 与 **大模型架构**：MLA / GQA / MTP 等都依赖高效注意力基座，FA 是它们的底层算力前提。

---

## 三、Speculative Decoding（EAGLE / MTP）

### 3.1 定位
**草稿–验证循环，无损（lossless）加速逐 token 解码**——batch=1 在线对话最确定的提速手段。本质：让小模型一次猜 K 个 token，主模型一次前向验证 K+1 个位置的 logits，顺序接受与分布一致的 prefix。

### 3.2 结构
```
Draft 模型： 预测 K 个候选 token
        │
        ▼
Target 模型： 一次前向拿 K+1 位置 logits
        │
        ▼
Rejection Sampler： 顺序比对，接受最长一致子串（接受率 α）
        │
        ▼
每步「换得」 τ = (1 − α^(r+1)) / (1 − α) 个 token
```

### 3.3 原理
- 验证是**并行**的：猜 2 个或 8 个，主模型都只跑一遍，所以「多猜」几乎不增延迟，猜得越准越快。
- 两条主流路线：
  - **EAGLE 系（北大 SafeAILab）**：用主模型**倒数第二层特征**当草稿输入；EAGLE-3 用倒数第二层特征+多层融合，Llama-3 8B 最高 **~6.5×**。
  - **MTP（Multi-Token Prediction，DeepSeek V3/R1）**：训练时多挂 MTP 头，推理作 draft 喂 EAGLE，**草稿与主模型天然同分布**，省去单独训练。

### 3.4 收益与生态
- DeepSeek V3 H200 TP8：batch=1 ~40→60 tok/s（**1.8×**），batch=32 ~30 tok/s（1.5×）；**batch 越小收益越大**。
- SGLang 首发 EAGLE-3，AMD ROCm 验证 **1.25~2.11×**；阶跃 JetSpec、vLLM v0.25（NEXTN/MTP）跟进。

### 3.5 坑（生产必看）
- 接受率必须监控（正常 60~90%，<30% 要关）；
- INT4 主 + FP16 草稿精度不匹配 → 接受率暴跌；
- batch 大时接受率下降是常态，可结合 tree-attention；
- **混血模型（注意力 + SSM/Mamba，如 Qwen3Next）专属坑**：被接受的 token 数 `nacc` 直接决定 SSM 状态递推步数，`nacc` 必须保持 `len(generated_token_ids) − 1`，改算法会导致 cu_seqlens 与 nacc 错位（nacc length mismatch）或输出乱码。详见 `知乎科普_投机解码与SSM状态.md`。

### 3.6 与素材对应
- 通俗+工程深版：`知乎科普_投机解码与SSM状态.md`（含 cu_seqlens/nacc 对齐 mermaid 图、K 递增验证建议）。
- 知识库：`AI知识库.md` `2026-07-16` 段「当日性能优化」。

### 3.7 与其它主题关联
- 与 **FlashAttention**：算力层与调度层协同（见上）。
- 与 **PD 分离**：Prefill 实例不需 spec decode（首 token 不受益），Decode 实例可重度配置 EAGLE draft。
- 与 **大模型架构**：MTP 是 DeepSeek V3/R1/Llama5 等原生训练技巧，已成前沿模型标准组件。

---

## 四、Transformer / BERT / ViT

### 4.1 定位
一套**通用序列建模架构**，跨语言与视觉成立。原生 Transformer 是 Encoder-Decoder；BERT 取其 Encoder-only 做理解；ViT 把图像切成 patch 当 token，也只吃 Encoder。

### 4.2 结构（三种变体）
- **原生 Transformer（2017）**：`Encoder(N×) → Decoder(N×) → Linear → Softmax`。
  - Encoder：Multi-Head Self-Attn → Add&Norm → FFN → Add&Norm。
  - Decoder：Masked Self-Attn → Cross-Attn(K,V 来自 Encoder) → FFN。
  - 配图：Encoder/Decoder 堆叠、Cross-Attn 的 K,V 红线（图已内嵌于 `01_基础架构/Transformer_BERT_ViT.md` 正文）。
- **BERT（Encoder-only）**：输入 → BERT（多层 Self-Attn+FFN）→ 接 **MLM** 与 **NSP** 两个预训练头。
  - 配图：In → BERT → MLM / NSP（图已内嵌于该文档正文）。
- **ViT（视觉，Encoder-only）**：`图像 → 16×16 patch → 展平+线性 → [CLS]+位置编码 → Transformer Encoder → CLS → 分类头`。
  - 224/16 → 14×14 = **196** 个 patch token（远小于 50176 像素）。
  - 配图：Img → 16x16 → Transf → CLS（图已内嵌于该文档正文）。

### 4.3 原理（关键公式）
- ViT 前向：
  - `z₀ = [x₁E; …; x_N E] + E_pos`
  - `z'ₗ = MSA(LN(z_{ℓ−1})) + z_{ℓ−1}`
  - `zₗ = MLP(LN(z'ₗ)) + z'ₗ`
  - `y = LN(z₀ᴸ) → head`
- 归纳偏置弱：小数据不如 CNN，大规模预训练（JFT-300M）后反超；催生 DeiT / Swin / CLIP / DALL·E 视觉骨干。

### 4.4 与素材对应
- 三张架构图（Transformer / BERT / ViT）已 **base64 内嵌**进 `01_基础架构/Transformer_BERT_ViT.md`，单文件自包含，不再单独保留 svg。
- 通俗版：`知乎科普/2026-07-15_ViT.md`（含 MLA/PD 分离延伸）、`email_bert_body.html`。
- 知识库：`AI知识库.md` `2026-07-15` 段（ViT + MLA + PD 分离）。

### 4.5 与其它主题关联
- ViT 是 **Step3-VL / 多模态模型**的视觉骨干（主题五）。
- Self-Attention 是 **FlashAttention / MLA / GQA** 的优化对象。
- 「patch=token」思想与「把图当句子读」直接启发了多模态拼接式架构。

---

## 五、Llama5 / Step3-VL / 国内外大模型架构

### 5.1 定位
2026 大模型架构全景：前沿模型**全面 MoE 化 + 原生多模态 + 超长上下文 + 强化推理**；多模态落地走「拼接式（ViT+投影+LLM）」与「原生多模态」两派。

### 5.2 结构 A：Llama 5（MoE 旗舰）
- 600B+ MoE（16 experts × ~38B），激活 ~60B；**5M** 上下文；五模态；**RSI（递归自改进）** + 原生 Agentic 训练。
- MoE Block：`Input → Attn(GQA) → Add&Norm → Router → Top-2 Experts(+Shared Expert) → Add&Norm → LM Head`。
- RSI vs CoT：RSI 检测矛盾会**回退重算**（自纠错），适合多步逻辑/代码调试/研究综合。
- 演进主线：L1(Dense/RoPE) → L2(+GQA/RLHF) → L3(15T tokens) → L4(首发 MoE) → L5(600B+RSI)。

### 5.3 结构 B：Step3-VL-10B（VLM 拼接式）
```
图像 → 多裁剪预处理(728全局+504局部) → PE-lang ViT(1.8B)
     → 双 stride=2 卷积 Projector：16× 空间压缩 + 维度对齐到 4096
     → 文本 embedding 同维度
     → 拼接为统一一维多模态序列 → Qwen3-8B Decoder（可选 PaCoRe）
     → 图文回答
```
- **投影层是关键翻译层**：压缩 token 数（16×）+ 对齐维度（到 D_llm=4096）。
- 推理提效：**GQA**（Qwen3，非 MLA）+ **RadixAttention**（SGLang 基数树共享前缀 KV）。
- **PaCoRe**：测试时 N=16 路并行推理 + 协调聚合（RLVR 训练），等效 128K 推理窗，且 16 路共享同一份视觉前缀 KV。

### 5.4 结构 C：国内外格局
- 四巨头（OpenAI/Anthropic/Google/Meta）+ 国产四强（DeepSeek/Qwen/GLM/Kimi）+ 阶跃星辰。
- **MoE 激活比从 2024 Mixtral ~25% 降到 2026 的 2–5%**（模型越来越稀疏）。
- **注意力三路线**：MLA（Kimi/GLM/DeepSeek 压 KV）、CSA/HCA 混合（DeepSeek V4）、Gated DeltaNet 线性注意力（Qwen 3.5）。
- **多模态两派**：原生多模态（Gemini/Kimi/Llama5，每层交互）vs 拼接式（Qwen/多数国产，成本低）。

### 5.5 与素材对应
- `模型技术库/06_大模型与多模态/Llama5_技术架构与演进.md`（MoE/RSI/演进时间线/部署）。
- `模型技术库/06_大模型与多模态/Step3-VL_多模态大模型.md`（Encoder-Projector-Decoder 全链路 + PaCoRe + RadixAttention）。
- `模型技术库/06_大模型与多模态/国内外最强大模型技术架构_2026.md`（格局/选型速查/六小龙分化/阶跃实力）。
- 知识库：`AI知识库.md` `2026-07-15/16` 段交叉引用。

### 5.6 与其它主题关联
- 视觉骨干 = **ViT**（主题四）；推理加速 = **FlashAttention + Speculative Decoding + PD 分离**（主题二/三）；
- DeepSeek V4 / Llama5 的 MTP 与 **Speculative Decoding** 同源；**MLA/GQA** 是注意力优化，与 FlashAttention 互补。

---

## 六、工程哲学串联 & 能力地图

五大主题反复出现同一个思想：**把「看似必须的事」重新拆解、重新组合**——
- SD 拆掉「必须在像素空间扩散」；
- FlashAttention 拆掉「必须物化 N×N 矩阵」；
- Speculative Decoding 拆掉「必须每步只生成 1 个 token」；
- MoE 拆掉「必须激活全部参数」；
- ViT 拆掉「必须先有卷积先验」。

**能力依赖地图（谁支撑谁）**：
```
Transformer 自注意力 ──优化──► FlashAttention / MLA / GQA
        │
        ├─► ViT（视觉骨干）──► Step3-VL / 多模态大模型
        │
        └─► MoE（参数稀疏化）──► Llama5 / DeepSeek / Qwen / Kimi
                                      │
                                      ├─► MTP ──► Speculative Decoding(EAGLE)
                                      └─► 推理栈：FlashAttention + Spec Decoding + PD 分离
        │
        └─► 扩散（LDM）──► SD/SDXL/FLUX/Wan-Video（与 LLM 共享「压缩+生成」哲学）
```

---

## 七、下一步
- 想逐主题深挖 → 见 **B 形态**（6 篇独立文档，已归入 `模型技术库/` 各分类）：01_基础架构 / 02_注意力机制 / 03_高效注意力计算 / 04_生成模型 / 05_推理加速 / 06_大模型与多模态（详见 `模型技术资产索引与知识地图.md` 分类清单）。
- 想盘点复用素材 → 见 **C 形态**：`模型技术资产索引与知识地图.md`（含 svg/科普/长文档/知识库的关系 mermaid 图）。
- 想出科普/邮件/周报 → 直接引用对应 svg（base64 已备好）与知乎科普版。

*整理于 2026-07-17，依据工作区各源文件核对。*
