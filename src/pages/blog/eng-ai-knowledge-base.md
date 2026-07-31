---
title: AI 知识库 · 大模型技术每日沉淀
description: Vision Transformer、MLA、PD 分离等大模型技术要点的每日沉淀知识库。
pubDate: 2026-07-31
series: 大模型工程实践
lang: zh
layout: ../../layouts/BlogPost.astro
---

## 2026-07-15 · Vision Transformer (ViT, 2020) ｜ 算子：MLA (Multi-head Latent Attention) ｜ 性能优化：PD 分离 (Prefill-Decode Disaggregation)

### 论文核心思想
- 将图像切分为 P×P 非重叠 patch（默认 16×16），展平+线性投影为 token，加 [CLS] 与可学习 1D 位置编码，送入纯 Transformer Encoder，取 CLS 做分类。
- patch 数 N = (H·W)/(P·P) = 196（224/16），远小于像素数 50176，使自注意力 O(N²·D) 可行。
- 关键公式：z0 = [x1E; …; xN E] + E_pos；z'ℓ = MSA(LN(z{ℓ-1})) + z{ℓ-1}；zℓ = MLP(LN(z'ℓ)) + z'ℓ；y = LN(z0^L) → head。
- 归纳偏置弱：小数据不如 CNN，大规模预训练（JFT-300M）后反超。催生 DeiT / Swin / 多模态（CLIP、DALL·E 视觉骨干）。

### 当日算子：MLA（DeepSeek V2/V3/V4）
- 定位：低秩潜变量压缩 KV Cache，显存随序列长度近乎不增长。
- 机制：c_t = W_DK·h_t ∈ R^r（r≪d）；k_t = W_UK·c_t，v_t = W_UV·c_t；仅缓存 c_t。
- 对比朴素 MHA：缓存从 2·n_h·d_h·L 降到 r·L（约一个数量级），长上下文/高并发受益。
- 在用模型：DeepSeek-V2/V3/V4（与解耦 RoPE 配合）；V4 已在 vLLM/SGLang 适配。

### 当日性能优化：PD 分离
- 定位：Prefill（算力密集、易批处理）与 Decode（访存密集、受带宽限制）拆到不同实例，各自最优扩缩。
- 机制：分离后 Prefill 高吞吐、Decode 低延迟，经 KV 传输层（NIXL/Mooncake）+ router 连接。
- 收益：decode TPOT 更稳定、prefill 吞吐更高、GPU 利用率↑、成本↓。
- 代表工作：DistServe、Mooncake；vLLM/SGLang 推进中。
- 坑：依赖稳定 KV 跨实例传输层与路由策略，否则增加尾延迟。

### 与已有条目关联
- 关联 2026-07-15 vLLM/SGLang 社区跟踪：vLLM v0.25 移除 PagedAttention、MRv2 默认；SGLang Spec V2；DeepSeek V4 / Step 3.7 Flash 适配——均与 MLA、PD 分离、投机解码等主题交叉。

### 当日产业关联
- 小米机器人进厂 98% 成功率；它石智航×地瓜机器人千台级具身部署；Optimus V3 量产；DeepSeek 融资；VLA 模型 2–3 年替代工厂精细手工活。视觉骨干（ViT 类）+ Transformer 是具身智能地基。

## 2026-07-22 · GAN (Generative Adversarial Nets, Goodfellow 2014) ｜ 算子：SwiGLU ｜ 性能优化：混合精度训练

### 论文核心思想
- 将生成问题重新定义为「生成器 G vs 判别器 D」的零和博弈：min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]。
- D 输出 [0,1] 标量判真伪；G 从噪声 z 映射到数据空间；交替优化。
- 理论上 p_g = p_data 时 G 达到全局最优，D* = 1/2（完全分不出）。
- 关键技巧：G 损失改用 -log D(G(z)) 防早期梯度消失；Adam β1=0.5；Label Smoothing。
- 坑：Mode Collapse（只产少数模式）、训练震荡。解决：WGAN/WGAN-GP（用 Wasserstein 距离替换 JS 散度）、SN-GAN。

### 当日算子：SwiGLU（门控 FFN）
- 定位：双支路门控激活函数，替代 ReLU/GELU 成为大模型 FFN 标配。
- 机制：SwiGLU(x) = (SiLU(xW_gate)) ⊙ (xW_val) · W_down；SiLU(x) = x·σ(x)。
- 对比 ReLU FFN：参数量持平（d_ff 调为 (8/3)d 替代原 4d），但引入乘法门控——高阶复合非线性，逐元素决定信息通过/抑制。
- 在用模型：Llama 全系、Qwen 全系、DeepSeek V4（+SwiGLU Clamping 钳制到 [-10,10]）、GLM-5、MiniMax、Gemma 4、Mistral。
- 坑：自己实现时 d_ff 必须调整为 (8/3)d，否则参数量比 ReLU FFN 多 50%。

### 当日性能优化：混合精度训练
- 定位：仅 GEMM 输入（权重+激活）用低精度，master weights/optimizer 保持 FP32，显存减半+吞吐 1.4~2×。
- 三条路线：① FP16 AMP（需 Gradient Scaling 防下溢）；② BF16 AMP（指数位同 FP32，无需 scaling）；③ FP8 Hybrid（E4M3 前向 + E5M2 反向，H100+）。
- 实测收益：DeepL 172B 吞吐 400→550 TFLOP/s（1.4×）；LLM-jp 172B 前 7000 步 BF16 后切 FP8 速度 1.4× loss 不崩。
- 在用：Megatron-LM `--fp8-format hybrid`；DeepSeek V4 Muon 优化器配 BF16 AMP；vLLM/SGLang FP8 量化部署 KV Cache 减半。
- 坑：FP16 记在 scaler.step(optimizer) 前先 unscale_() 否则梯度裁剪失效；BF16 训练 LR 需比 FP32 高 10-20%。

### 与已有条目关联
- 关联 2026-07-15 vLLM/SGLang 社区跟踪：DeepSeek V4 SwiGLU Clamping 是 SwiGLU 算子的工程化应用；混合精度训练是 FP8 量化部署的前置基础。
- 关联 2026-07-15 当日性能优化 PD 分离：PD 分离的 KV Cache 压缩与混合精度推理（FP8 KV Cache）配合使用。

### 当日产业关联
- WAIC 2026 落幕：具身智能从"炫技"转"实干"，行业进入规模化落地快车道；京东 JoyAI 7 款基础模型+精灵 G2 Max 落地仓库；DeepSeek V4 技术报告更新（SwiGLU Clamping/Muon/mHC）；Kwai 3D-Vision 多模态融合 3D；百台机器人全国物流中心常态化上岗。

## 2026-07-16 · Stable Diffusion (LDM, CVPR 2022) ｜ 算子：FlashAttention (FA-1→FA-4) ｜ 性能优化：Speculative Decoding (EAGLE / MTP)

### 论文核心思想
- 把图像生成从像素空间搬到 VAE 压缩后的潜空间：4×64×64（48×↓），让扩散在消费级 GPU 上跑起来。
- 两阶段：① VAE(KL-reg) 感知压缩像素→潜码；② U-Net+Cross-Attention 在潜空间做 DDPM 扩散，注入文本 τ(c)；最后 VAE-Dec 还原像素。
- 关键公式：L_LDM = E[‖ε - ε_θ(z_t, t, τ(c))‖²]；CFG ε̂ = ε_u + w·(ε_c - ε_u)，w∈[5,12]。
- 训练/推理要点：条件 dropout 10%（CFG 前提）；SD1.x 平方余弦调度 → SDXL EDM → SD3 rectified flow。
- 影响：SDXL/SD3/HunyuanDiT/Wan-Video/FLUX 全沿用此模板；LoRA/ControlNet/IP-Adapter 整条下游微调链都站在 LDM 肩膀上。

### 当日算子：FlashAttention（Tri Dao et al.）
- 定位：IO 感知 + 在线 softmax + 算子融合，让标准 Attention 不必物化 N×N，化解"显存墙"。
- 机制：分块（Q_i,K_j,V_j）入 SRAM，跨块维护 m_i（max）、l_i（denominator）、o_i（累加器）—在线 softmax 保证 bit-exact。
- 收益：A100 上 2~4× 加速、长序列 128K 不 OOM；H100 FA-3 FP8 + 异步流水线，吞吐达 740 TFLOPs/s。
- 在用模型：Qwen3/3.5（FA-2/3 + GQA）、DeepSeek V3/V4（FA-2/3 + MLA + vLLM/SGLang）、MiniMax（Lightning+FA 混合）、GLM/Kimi/Llama 5（FA-3 BF16/FP8）。
- 坑：自定义 mask 触发慢路径；FP8 长序列需 block-wise scaling；旧 GPU（A100 前）只能用 FA-1。

### 当日性能优化：Speculative Decoding（EAGLE / MTP）
- 定位：草稿-验证循环，无损加速 decode — batch=1 在线对话最确定的提速手段。
- 机制：draft 模型预测 K 个候选，主模型一次前向拿 K+1 位置 logits，顺序接受与分布一致的 prefix（接受率 α），每步换得 τ = (1-α^(r+1))/(1-α) 个 token。
- 两条路线：① EAGLE 系（北大 SafeAILab）— EAGLE-3 用倒数第二层特征+多层融合，Llama-3 8B 最高 ~6.5×；② MTP（DeepSeek V3/R1）— 训练多挂 MTP 头，推理作 draft 喂 EAGLE，天然同分布。
- 收益：DeepSeek V3 H200 TP8 batch=1 ~40→60 tok/s（1.8×），batch=32 ~30 tok/s（1.5×）。
- 集成：SGLang 首发 EAGLE-3，AMD ROCm 验证 1.25~2.11×；阶跃 JetSpec、vLLM v0.25（NEXTN/MTP）跟进。
- 坑：接受率生产环境必监控（正常 60~90%，<30% 要关）；INT4 主+FP16 草稿精度不匹配暴跌；batch 大时接受率下降是常态，可结合 tree-attention。

### 与已有条目关联
- 与 07-15 MLA+PD 分离：LDM 用 U-Net（不存 KV 故与 MLA 无直接交集），但下游文生视频（Wan-Video、SD3 视频版）已逐步接入 MLA 思路压缩 Cross-Attention 上下文。
- FlashAttention 与 Speculative Decoding 在 vLLM/SGLang 已深度协同：FA-3 加速主模型前向，Spec Decoding 提升单请求 token/s，两者叠加是当前最优推理栈。
- 投机解码可与 PD 分离叠加：Prefill 实例不需 spec decode（首 token 不受益），Decode 实例可重度配置 EAGLE draft 模型。

### 当日产业关联
- 小米开源 380B 具身生成大模型 U0（WorldArena 第一，FlashAR+ 推理 1024² 数据生成 450.77s→5.44s 82.9×↑，π₀.₅ 策略 OOD 任务 36.9%→63.2%）。
- 小米 MiMo 端侧大模型通过国家备案，机器人进厂柔性工件成功率 90%、自攻螺母双侧 98%。
- Thinking Machines（OpenAI 前 CTO Mira Murati）首发开放权重 Inkling：MoE 975B / 激活 41B / 1M context / 45T token / Apache 2.0；Design Arena 1257 分与 Claude Opus 4.6 持平。
- 腾讯混元 Hy3 发布一周调用量较 Hy2 增长 68×，OpenRouter 全球总榜第一；阿里千问集成国行 Apple 智能；字节豆包 AI 手机 WAIC 亮相。
- WAIC 2026 明日上海开幕：东方算芯 3D 近存算 AI 芯片 DF1000、矩阵超智第三代全尺寸人形机器人（今年 1000/明年 10000 台）、商汤 Seko 创编一体 AI 视频智能体（5s 生 5s）。
- 行业主旋律：具身智能从"会演"走向"能干"（具身数据工厂+芯片自主+原生 OS 合流）；大模型进入"拼成本/拼可改写"阶段。

## 2026-07-18 · GPT (Improving Language Understanding by Generative Pre-Training, 2018) ｜ 算子：RoPE (Rotary Position Embedding) ｜ 性能优化：KV Cache 量化与驱逐

### 论文核心思想
- 首次证明「生成式预训练 + 下游微调」范式可横扫 NLU 任务：先在 BookCorpus 上做因果 LM（L₁=Σ log P(uᵢ|u<ᵢ;Θ)），再迁移 Θ 加线性头 W_y 微调，辅以 λ·L₁ 防遗忘。
- Decoder-only Transformer：12层, d=768, h=12, ctx=512, 117M params, 仅因果 mask。
- 统一输入格式：通过 Start/Delim/Extract token 构造多任务输入，不改架构即可适配分类、蕴涵、相似度、问答等。
- 消融要点：去掉辅助 LM 损失↓1~3%，去掉预训练大幅↓，去掉 Transformer 改 LSTM 显著↓。
- 影响：确立 GPT 路线 → GPT-2/3/4/ChatGPT；与 BERT（双向掩码）分庭抗礼，GPT 的生成能力最终胜出。

### 当日算子：RoPE（Rotary Position Embedding, Su et al. 2021）
- 定位：在 2D 子空间内旋转 Q/K 编码相对位置，使注意力分数仅依赖偏移 m−n，是所有主流 Decoder LLM 的默认位置编码。
- 核心公式：θ_i=10000^{-2i/d}；(q_{2i}',q_{2i+1}')=R(mθ_i)·(q_{2i},q_{2i+1})；⟨Q_m',K_n'⟩=f(q,k,m-n)。
- 对比朴素绝对位置编码：零参数、显式相对位置、+NTK/YaRN可扩展到百万级 token；计算开销约0.01% FLOPs。
- 在用模型：Qwen3/3.5（NTK-aware scaling扩展至128K/1M）、DeepSeek V4（解耦RoPE+MLA）、MiniMax（Lightning+RoPE混合）、Llama 5/GLM/Kimi（默认RoPE）。
- 坑：V不旋转；直接外推超训练长度会输出垃圾需调base；base需≥6.4×10⁵（32K上下文）；与ALiBi不混淆。

### 当日性能优化：KV Cache 量化与驱逐
- 定位：KV Cache 是长上下文推理的主要显存瓶颈——量化压缩存储精度（2~4×），驱逐削减缓存 token 数（3~244×），两者叠加可让百万级 token 序列在有限 GPU 上跑起来。
- 量化维度：FP16→FP8(E4M3/E5M2) 2× 压缩 ~0.3%损失；FP16→INT4混合(K对称+V非对称) 4× ~0.5%损失；per-group(group_size=128)主流。
- 驱逐维度：Window Attention 保留最近N个token ~244:1 对话场景；Heavy Hitter(H2O)保留20~30%高注意力token 质量损失可控；叠加淘汰50%+INT4→总压缩比8×。
- 前沿进展：RDKV(2026)率失真统一量化+驱逐，2.48%缓存→97.81%准确率；ThinKV(ICLR 2026 Oral)思想自适应KV压缩，<5%KV保留近乎无损，5.8×吞吐提升。
- 代表工作：vLLM/SGLang 默认FP8 KV；DeepSeek V4 MLA+FP8 KV双重压缩1M序列仅需~20GB；RDKV 128K下4.5×加速+1.9×显存下降。
- 坑：FP8 E5M2(长文本多语言) vs E4M3(代码数学)选择；INT2实用下限之下；量化+驱逐叠加需调参；H2O额外5~10%计算开销。

### 与已有条目关联
- 与 07-15 MLA：MLA压缩KV维度(8192→128)，RoPE仅对解耦k_rot部分应用——两者在DeepSeek V4深度配合。
- 与 07-15 PD分离：KV Cache量化让Decode实例更省显存可装更多并发，PD分离让Prefill实例不需量化。
- 与 07-16 FlashAttention：FA-3加速注意力计算，KV Cache量化加速存储——计算和存储两条线协同优化。
- 与 07-16 Speculative Decoding：KV Cache量化让Decode实例可装更多draft+主模型KV，Spec Decoding提升单请求token/s。
- GPT论文与已有Transformer(07-14/ papers_sent)、BERT(07-14)条目直接关联：GPT用Decoder-only(单向因果)，BERT用Encoder(双向掩码)，两者同源于Transformer。

### 当日产业关联
- WAIC 2026 上海开幕：具身智能升格核心赛道，200+企业参展；蚂蚁灵波LingBot-VLA2.0「具身原生」（6万小时真实数据，17厂商20构型）；智元远征A3Ultra十大镇馆之宝；宇树展示无人产线方案；穹彻智能药房落地沈阳。
- OpenAI发布ChatGPT Agent（整合Operator+Deep Research自主完成任务）。
- Google DeepMind推出MoR新架构（推理速度2×，KV内存减半）。
- 腾讯Robotics X×越疆跑通Physical AI产线落地（真实化妆品产线95%成功率）。
- 优必选Walker S2全球首创自主换电（7×24小时不间断作业）。

## 2026-07-25 · 【具身智能】RT-2 (Vision-Language-Action Models, Google DeepMind 2023) ｜ 算子：RMSNorm ｜ 性能优化：PagedAttention

### 论文核心思想
- 首个大规模 VLA（视觉-语言-动作）模型：把预训练 VLM（PaLI-X 55B / PaLM-E 12B）直接微调成端到端机器人策略。
- 关键创新：将 7-DoF 机械臂动作按 256-bin 离散化为文本 token，与语言共享 tokenizer 与自回归输出空间。
- 训练目标：L = -Σ_t log P(a_t | o_≤t, q_≤t; Θ)，采用 co-fine-tuning 混合网页图文数据与 RT-1 机器人演示数据。
- 推理：相机图像 + 语言指令 → VLM 自回归生成动作字符串 → 反离散化 → 机械臂执行；支持 chain-of-thought 先规划再动作。
- 效果：unseen 任务成功率 RT-1 32% → RT-2 62%；scale 越大泛化越好，奠定 OpenVLA / π0 / GR00T 等后续 VLA 路线。

### 当日算子：RMSNorm
- 定位：LayerNorm 的简化版，去掉均值中心化和 β 偏置，只保留 RMS 重缩放 + 可学习 γ。
- 公式：RMS(x) = sqrt(mean(x²))；RMSNorm(x) = γ · x / RMS(x)。
- 对比：约 7–15% 加速、参数量略减、效果持平；残差连接可吸收 mean shift。
- 在用模型：Llama 全系、Qwen3/3.5、DeepSeek V4、MiniMax、Mistral、Gemma、PaLM。
- 坑：Llama 源码变量名仍叫 layernorm 但类为 LlamaRMSNorm；端侧部署时 eps 不可省略。
- 具身关联：VLA 端侧推理每 block 省计算，堆叠 32–64 层后 latency 显著下降。

### 当日性能优化：PagedAttention
- 定位：解决 LLM 服务中 KV Cache 的碎片与过度预留，提升内存利用率与并发。
- 机制：把 KV Cache 切成 16-token block，逻辑连续 → 物理非连续，通过 per-sequence block table 映射；按需分配、用完回收、支持 prefix caching / copy-on-write。
- 对比：内存利用率 20–40% → 95%+；vLLM 在 ShareGPT 上比 Orca 高 1.7–2.7× 请求率，比 FasterTransformer 高 2–4× 吞吐。
- 代表工作：vLLM（SOSP 2023）、SGLang；已成为 DeepSeek V4 / Qwen3 / Llama 5 主流推理后端。
- 坑：block_size 需权衡；prefix caching 要求 block 哈希对齐；swap vs recompute 需按场景选择。
- 具身关联：多机器人并发 VLA 推理时，PagedAttention + continuous batching 可提升单卡服务机器人数量、降低单位推理成本。

### 与已有条目关联
- 关联 2026-07-19 LLaVA：LLaVA 把指令微调引入多模态，RT-2 把多模态模型直接用于机器人控制，是多模态理解的下游延伸。
- 关联 2026-07-16 Stable Diffusion / 2026-07-15 ViT：视觉编码器（ViT/CLIP）是 VLM 视觉骨干，也是 VLA 的“眼睛”。
- 关联 2026-07-18 RoPE / 2026-07-19 GQA / 2026-07-22 SwiGLU：RMSNorm 与这些算子共同构成现代 Decoder LLM 的标准积木。
- 关联 2026-07-15 PD 分离 / 2026-07-16 Speculative Decoding / 2026-07-18 KV Cache 量化：PagedAttention 与这些技术在同一推理栈中协同，共同服务长上下文与低成本部署。

### 当日产业关联
- 小鹏人形机器人开启小批量试生产；智元创新启动赴港上市；成都与宇树科技签署战略协议；两部门发文推动“人工智能+交通运输”。
- 清华李升波：自动驾驶是第一类具身智能；享道 Robotaxi 累计订单 35 万+/安全里程 400 万+公里。
- Anthropic Claude Opus 5 发布，性能接近 Fable 5 价格减半，ARC-AGI-3 达 30.2%。
- 百度开源 LoongForge，支持 LLM/VLM/扩散模型/具身智能训练。
- 腾讯 QClaw 并入 WorkBuddy、阿里整合办公 Agent；阶跃/努比亚智能体手机亮相 WAIC。

## 2026-07-19 · LLaVA (Visual Instruction Tuning, NeurIPS 2023 Oral) ｜ 算子：GQA (Grouped-Query Attention) ｜ 性能优化：LoRA / QLoRA (参数高效微调)

### 论文核心思想
- 首次将指令微调（Instruction Tuning）范式引入多模态领域：用纯文本 GPT-4 基于 COCO captions 生成 158K 条多模态指令数据（对话/描述/推理），冻结 CLIP ViT-L/14 + 可训练投影层 W + Vicuna LLM 做两阶段训练。
- 架构极简：Image → CLIP ViT（冻结，256 patch token）→ Linear Projection W（可训）→ Vicuna Decoder（Stage 2 解冻）→ 自回归文本回答。
- 关键公式：H_v = W · g(X_img)，Input = Concat(H_v, Embed(Tokens))，Loss = -Σ_t log P(a_t | H_v, x_<t)。W ∈ R^{d_LLM × d_v}。
- 两阶段训练：S1 特征对齐（仅训 W，CC3M 595K，~1h/8×A100）；S2 指令微调（W+LLM，158K 对，~10h/8×A100）。
- 影响：GPT-4-as-Judge 达 85.1% 相对分；ScienceQA 92.53% SOTA；NeurIPS 2023 Oral；开源催生 LLaVA-1.5/NeXT/CogVLM/InternVL/Qwen-VL 整条多模态 VLM 生态。

### 当日算子：GQA（Grouped-Query Attention, Ainslie et al. 2023）
- 定位：解决 MHA 的 KV Cache 显存爆炸——将 H 个 Query Head 分为 G 组，每组共享一个 KV Head，以 4~8× KV Cache 压缩换取接近 MHA 的质量。
- 核心公式：group_id(h) = ⌊h·G/H⌋；K_g = x·W_K^g, V_g = x·W_V^g（g ∈ {1..G}）；Attention_i = softmax(Q_i·K_g^T/√d_head)·V_g；KV Cache = 2·L·G·d_head（vs MHA 的 2·L·H·d_head，压缩 H/G 倍）。
- 在用模型：Llama 2/3（70B H=64 G=8, 8:1）、Qwen2/2.5/3 全系（H=32~64 G=8）、Mistral/Mixtral（H=32 G=8 4:1, 与滑动窗口层交替）、DeepSeek V2-V4（GQA + MLA 双重压缩）、MiniMax（GQA + Lightning 混合）。G=8 为业界甜点。
- 坑：G 过小（MQA G=1）质量掉 2~3% 已废弃；生产环境 repeat_interleave 在 FA kernel 内隐式共享，零额外访存。

### 当日性能优化：LoRA / QLoRA（参数高效微调 PEFT）
- 定位：解决大模型全参数微调的显存/算力/版本管理三重瓶颈——冻结 W₀，仅训练低秩 ΔW = B·A（r ≪ d），50~100× 降成本，质量仅损失 1~3%；QLoRA 将 W₀ 量化为 4-bit NF4 使 65B 在单卡可微调。
- 核心机制：h = W₀·x + (α/r)·B·A·x；B ∈ R^{d×r}, A ∈ R^{r×d}（2·r·d ≪ d²）；QLoRA 叠加 4-bit NF4 + 双重量化(SQ) + paged optimizers → 7B 模型仅需 12-16 GB VRAM。
- 代表工作与收益：HuggingFace PEFT 一站式支持；Unsloth 2~5× 加速；LLaVA 自身用 LoRA 微调（0.5% 参数达相当性能）；Qwen3.5 推荐 r=64 + DoRA；DeepSeek V4 社区数百 LoRA 适配器；QLoRA 微调 MiniMax-8B 成本 < $20。
- 坑：r=8→16→64 三档策略（实验→标准→深度）；α=2r 强默认；QLoRA 稳定性差于 LoRA 建议先 LoRA 验证；DoRA 质量高 1~3% 仅额外 ~10% 开销；PiSSA 初始化加速收敛 2× 需注意 FSDP 兼容性。

### 与已有条目关联
- 与 07-18 GPT：LLaVA 的 LLM 骨干 Vicuna 基于 LLaMA（GPT 路线），印证 GPT 解码器范式是通用多模态基座。
- 与 07-16 Stable Diffusion：同为视觉-语言模型，LDM 是「文→图」生成，LLaVA 是「图→文」理解，两者共享 CLIP 视觉编码器。
- 与 07-15 ViT/MLA：ViT 是 LLaVA 视觉骨干的基础；MLA 是 GQA 之后的进一步 KV 压缩路线（DeepSeek 同时使用两者）。
- GQA 与 07-16 FlashAttention 协同：FA 加速计算，GQA 压缩存储，组合是当前最优推理栈。
- LoRA/QLoRA 与 07-16 Speculative Decoding 互补：LoRA 降低训练成本，Spec Dec 降低推理时延。

### 当日产业关联
- WAIC 2026 Day 2：腾讯 TairosAgent 具身智能体框架发布，WorkBuddy 三端独立 App 上线（多模态 Agent 落地加速）。
- Kimi K3（2.8T 参数开源）引发"DeepSeek 2.0 时刻"，美股科技股大幅波动（开源路线持续冲击闭源商业逻辑）。
- Grok 4.6 下周完成训练（2T 参数）→ 多模态能力竞赛全面展开。
- 蚂蚁 LingBot-Vision 开源（具身智能视觉感知） / 京东 JoyAI 模型矩阵亮相 WAIC。
- AliCloud KV Cache Store + Agentic FS 基础设施发布，日均 Token 调用突破 140 万亿次。

## 2026-07-31 · 【具身智能】HiF-VLA（WAM 世界动作模型）
- 论文核心：Motion-centric 双向时空推理(Hindsight-Insight-Foresight)，联合专家同时输出未来 Motion 预测 + 动作序列，实现「边想边做」。
- 关联算子：MLA（KV 压缩，支撑长上下文）；关联优化：PD 分离（支撑多机器人并发推理降本）。
- 当日产业：三家车企量产、阶跃+千里装车50万、融资935亿；主线=具身智能引擎。
