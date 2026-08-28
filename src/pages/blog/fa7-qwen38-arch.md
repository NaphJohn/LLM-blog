---
title: 前沿架构解码手记（七）：Qwen3.8 双 checkpoint 对比——稠密 27B vs 稀疏 Flash-Next（360GB 权重只为换 6B 激活）
description: 基于两个仓库 master 分支的模型卡、config.json、权重索引与 safetensors 头部扫描，拆解 Qwen3.8-Flash-Next（Qwen4Exp 稀疏 MoE，125B 主模型 / 6B 激活 / 360GB BF16）与 Qwen3.8-27B（Qwen3_5 稠密，55.6GB BF16）的架构与权重组织差异——前者用更大的静态存储换更低的 token 级激活，后者权重文件小而直接。附 3 张自绘架构/容量拆解图。
pubDate: 2026-08-28
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa7-qwen38-arch
layout: ../../layouts/BlogPost.astro
---

## 0. 结论先行

这两个 checkpoint **不是同一模型的大小版本**，而是两套架构与权重组织方式明显不同的权重：

- **Qwen3.8-Flash-Next** = `Qwen4ExpForConditionalGeneration` 架构的**稀疏 MoE**。主模型按模型卡口径 **125B**，每 token 激活约 **6B**；另有约 **51B N-gram embedding** 与 **MTP** 权重。仓库含 **131 个 BF16 safetensors 分片**，权重约 **360 GB（335.276 GiB）**。
- **Qwen3.8-27B** = `Qwen3_5ForConditionalGeneration` 架构的**稠密**模型。仓库含 **18 个 BF16 safetensors 分片**，权重约 **55.6 GB（51.747 GiB）**。

Flash-Next 的权重文件是 27B 的 **6.48 倍**，但它**并不是每个 token 都计算全部权重**——大量存储来自 512 个专家和 N-gram 查表参数。"Flash" 主要体现**计算/访问效率设计**，不代表下载文件更小。两者 tokenizer、视觉预处理、生成配置多项相同，但 config/索引/权重全不同，**不能混用分片**。

> **型号勘误更新（2026-08-28）**：本系列 fa6（Gated DeltaNet）曾据当时快照判断"`qwen3.8-27B` 不存在、Qwen3.8 为 2.4T-A95B 这类 MoE"。本文基于 2026-08-28 的仓库直接扫描确认：**`Qwen3.8-27B` 作为一个稠密 checkpoint 真实存在**（55.6GB BF16，架构 `Qwen3_5ForConditionalGeneration`）。fa6 的判断在其当时快照下成立，本仓库现已可见 27B 稠密 checkpoint，特此校正。两仓库 revision 短 ID：Flash-Next `2741eec1`、27B `1098534a`。

## 1. 架构总览：同样词表/上下文，骨架相反

两者共享 **248,320 词表**（padding 后）与 **262,144 原生上下文**，但"堆容量"的方式完全相反：Flash-Next 走**窄主干 + 专家/N-gram 查表**，27B 走**宽主干 + 单层稠密 FFN**。

<figure class="arch-fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Flash-Next 与 27B 架构总览对比">
  <defs>
    <style>
      .ttl{font:700 14px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .nm{font:700 16px -apple-system,'PingFang SC',sans-serif}
      .kv{font:12px -apple-system,'PingFang SC',sans-serif;fill:#374151}
      .sub{font:11px -apple-system,'PingFang SC',sans-serif;fill:#6b7280}
      .blue{fill:#eff6ff;stroke:#1d4ed8;stroke-width:1.5}
      .terra{fill:#fff7ed;stroke:#c2410c;stroke-width:1.5}
      .chip{fill:#fff;stroke:#cbd5e1;stroke-width:1}
      .chipb{fill:#dbeafe;stroke:#1d4ed8;stroke-width:1}
      .chipg{fill:#ecfdf5;stroke:#047857;stroke-width:1}
      .chipa{fill:#fef9c3;stroke:#a16207;stroke-width:1}
    </style>
  </defs>
  <text x="340" y="22" text-anchor="middle" class="ttl">同样 248,320 词表 · 262,144 上下文，骨架与容量哲学相反</text>
  <rect class="blue" x="20" y="40" width="300" height="216" rx="10"/>
  <text x="170" y="66" text-anchor="middle" class="nm" fill="#1d4ed8">Qwen3.8-Flash-Next</text>
  <text x="170" y="84" text-anchor="middle" class="sub">Qwen4Exp · 稀疏 MoE · 125B 主模型</text>
  <rect class="chip" x="36" y="96" width="268" height="30" rx="6"/>
  <text x="170" y="115" text-anchor="middle" class="kv">窄主干 宽度 2,560 · 48 层</text>
  <rect class="chipg" x="36" y="134" width="268" height="30" rx="6"/>
  <text x="170" y="153" text-anchor="middle" class="kv">512 专家 MoE（每 token 选 10+1）</text>
  <rect class="chipa" x="36" y="172" width="268" height="30" rx="6"/>
  <text x="170" y="191" text-anchor="middle" class="kv">51B N-gram 查表（128 分片）</text>
  <rect class="chipb" x="36" y="210" width="268" height="30" rx="6"/>
  <text x="170" y="229" text-anchor="middle" class="kv">每 token 激活 ≈ 6B · 权重 360 GB</text>
  <rect class="terra" x="360" y="40" width="300" height="216" rx="10"/>
  <text x="510" y="66" text-anchor="middle" class="nm" fill="#c2410c">Qwen3.8-27B</text>
  <text x="510" y="84" text-anchor="middle" class="sub">Qwen3_5 · 稠密 · 27B</text>
  <rect class="chip" x="376" y="96" width="268" height="30" rx="6"/>
  <text x="510" y="115" text-anchor="middle" class="kv">宽主干 宽度 5,120 · 64 层</text>
  <rect class="chip" x="376" y="134" width="268" height="30" rx="6"/>
  <text x="510" y="153" text-anchor="middle" class="kv">每层一套稠密 FFN（中间维 17,408）</text>
  <rect class="chip" x="376" y="172" width="268" height="30" rx="6"/>
  <text x="510" y="191" text-anchor="middle" class="kv">无专家 · 无 N-gram 额外轴</text>
  <rect class="chipb" x="376" y="210" width="268" height="30" rx="6"/>
  <text x="510" y="229" text-anchor="middle" class="kv">每 token 激活 = 全部 27B · 权重 55.6 GB</text>
</svg>
<figcaption>图：两者词表/上下文相同，但 Flash-Next 用"窄主干 + 512 专家 + 51B N-gram 查表"堆容量，27B 用"宽主干 + 单层稠密 FFN"直接撑容量。</figcaption>
</figure>

## 2. 精度与容量口径

两份仓库**都是 BF16，不是 FP8**。config 均声明 `text_config.dtype = bfloat16`（`mamba_ssm_dtype: float32` 仅 SSM 计算用，不代表权重 FP32）。头部扫描确认：Flash-Next 为 1,655 个 BF16 张量 + 3 个 I64 元数据；27B 为 1,199 个 BF16 张量。两仓库均无 `quantization_config`。独立量化仓库：`Flash-Next-FP8` ≈ 185.5 GB、`27B-FP8` ≈ 30.9 GB。

| 项目 | Flash-Next | 27B |
|---|---|---|
| 分片数 | 131 | 18 |
| 索引 payload | 359.999963 GB / 335.276 GiB | 55.562856 GB / 51.747 GiB |
| 实际 .safetensors 总和 | 360.000193 GB / 335.276 GiB | 55.563007 GB / 51.747 GiB |
| 模型卡参数量口径 | 主 125B + 51B N-gram + 4B MTP | 27B |
| 许可证 | qwen-community-1.0 | Apache-2.0 |

容量差异基本全部来自权重本身（非权重文件两者均约 23 MB，几乎相同）。

## 3. 架构差异（来自 config.json）

| 维度 | Flash-Next | 27B | 对权重的影响 |
|---|---|---|---|
| 语言模型宽度 | 2,560 | 5,120 | Flash 投影更窄，但专家更多 |
| 语言层数 | 48 | 64 | 27B 层数更多 |
| 混合层布局 | 12×(3×[GDN→MoE]→1×[QSA→MoE]) | 16×(3×[GDN→FFN]→1×[Gated Attn→FFN]) | 都是 3 线性 + 1 全注意力周期 |
| 线性注意力层 | 36（48 V head/16 QK head/dim 128） | 48 | 结构相同 |
| 全注意力层 | 12 个 QSA 层 | 16 个 Gated Attention 层 | Flash 有 QSA indexer 权重 |
| FFN 形式 | 512 专家；每 token 10 routed + 1 shared | 每层一套稠密 FFN | Flash 存全部专家，27B 只存一套 |
| 专家/FFN 中间维度 | routed/shared = 640 | 稠密 FFN = 17,408 | Flash 靠专家数扩展，27B 靠单 FFN 宽度 |
| N-gram embedding | 有；基准词表 2,000,000，拆 128 表分片，接第 2 层 | 无 | Flash 多出约 51B 查表参数 |
| Residual | Gated Residual，4 branch，rank 320；hyper-connection 张量 | 无对应 | Flash 多一组残差混合权重 |
| 视觉→语言投影输出 | 2,560 | 5,120 | 预处理相同，连接层权重不同 |

## 4. 权重索引里的组织方式

**Flash-Next 的专家权重按专家打包成大张量**：`layers.0.mlp.experts.gate_up_proj` 形状 `[512,1280,2560]`，第一维 512 对应专家，大张量再分布到 131 分片。每 token 只路由 10 个 routed + 1 shared，但 **checkpoint 必须保存全部 512 个专家**——计算量接近激活的少数专家，磁盘/可用存储仍接近全部专家。

**N-gram 表是额外的容量轴**：索引含 128 个 `ple_embedding.ngram_embedding.shard_*.weight`，形状 `[2,500,012, 160]`，合计 **51.2B 参数 / 102.4 GB**。官方说明强调这类参数主要靠局部 n-gram 查表获取、不进每 token 常规矩阵乘法预算，设计目标之一是更易放 Host Memory 异步预取；但下载时不能省略这些分片，能否高效 offload 仍取决于推理框架。

**27B 用普通稠密 FFN**：`gate_proj/up_proj/down_proj` 形状 `[17408,5120]`/`[5120,17408]`，每层一套，容量由 64 层 + 17,408 中间维度构成，无需保存数百专家副本。

## 5. 权重存储拆解（按张量名称归类）

<figure class="arch-fig">
<svg viewBox="0 0 680 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Flash-Next 与 27B 权重存储占比拆解">
  <defs>
    <style>
      .ttl{font:700 14px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .lab{font:12px -apple-system,'PingFang SC',sans-serif;fill:#374151}
      .sm{font:10.5px -apple-system,'PingFang SC',sans-serif;fill:#fff}
      .smk{font:10.5px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
    </style>
  </defs>
  <text x="20" y="20" class="ttl">权重去哪了？Flash-Next 95% 在「专家 + N-gram」，27B 62% 在「稠密 FFN」</text>
  <!-- Flash-Next bar -->
  <text x="20" y="52" class="lab" fill="#1d4ed8">Flash-Next · 360 GB（等宽比例条）</text>
  <rect x="60" y="60" width="402.6" height="34" fill="#047857"/>
  <text x="261" y="81" text-anchor="middle" class="sm">Routed 专家 241.6GB · 67.1%</text>
  <rect x="462.6" y="60" width="170.4" height="34" fill="#a16207"/>
  <text x="547.8" y="81" text-anchor="middle" class="sm">N-gram 102.4GB · 28.4%</text>
  <rect x="633" y="60" width="27" height="34" fill="#94a3b8"/>
  <text x="646" y="108" class="smk">其他≈16GB (MTP/视觉/注意力)</text>
  <!-- 27B bar -->
  <text x="20" y="132" class="lab" fill="#c2410c">27B · 55.6 GB（等宽比例条）</text>
  <rect x="60" y="140" width="369.6" height="34" fill="#c2410c"/>
  <text x="244.8" y="161" text-anchor="middle" class="sm">稠密 FFN 34.2GB · 61.6%</text>
  <rect x="429.6" y="140" width="230.4" height="34" fill="#94a3b8"/>
  <text x="544.8" y="161" text-anchor="middle" class="smk">其他≈21.3GB (MTP/视觉/注意力)</text>
  <!-- legend -->
  <rect x="60" y="200" width="14" height="14" fill="#047857"/><text x="80" y="211" class="lab">Routed 专家 FFN</text>
  <rect x="200" y="200" width="14" height="14" fill="#a16207"/><text x="220" y="211" class="lab">N-gram 查表</text>
  <rect x="320" y="200" width="14" height="14" fill="#c2410c"/><text x="340" y="211" class="lab">稠密 FFN</text>
  <rect x="440" y="200" width="14" height="14" fill="#94a3b8"/><text x="460" y="211" class="lab">其他（MTP/视觉/注意力等）</text>
  <text x="20" y="250" class="lab">要点：Flash-Next 大文件主要不是 attention/视觉塔，而是</text>
  <text x="20" y="268" class="lab">全部 routed 专家 + N-gram 表；27B 更小不仅因层数少，</text>
  <text x="20" y="286" class="lab">更因它没有 512 专家与 51B N-gram 这两块「额外轴」。</text>
</svg>
<figcaption>图：按张量名称归类的存储拆解（比例条等宽，便于比较结构占比）。Flash-Next 的 Routed 专家占 67% + N-gram 占 28% ≈ 95%；27B 主要是稠密 FFN（62%）。</figcaption>
</figure>

## 6. 哪些文件相同 / 不能混用

**SHA256 完全相同（可复用）**：`chat_template.jinja`、`configuration.json`、`generation_config.json`、`merges.txt`、`preprocessor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`video_preprocessor_config.json`、`vocab.json`——tokenizer、词表、图像/视频预处理、生成配置高度一致。

**必须视为模型专属（不可交叉替换）**：`config.json`、`model.safetensors.index.json`、全部 `model-*.safetensors`、README、LICENSE、`.gitattributes`。两仓库分片均用 `model-00001-of-...` 通用命名，但分片总数、索引映射、内部张量名完全不同——**不能把一个仓库的第 1 分片与另一个仓库的索引配套**。注意：视觉预处理文件相同 ≠ 视觉权重相同（语言隐藏维度 2,560 vs 5,120，连接层不同）。

## 7. 对推理部署的含义

<figure class="arch-fig">
<svg viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="存储与激活参数对比：Flash-Next 大存储换低激活">
  <defs>
    <style>
      .ttl{font:700 14px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .lab{font:12px -apple-system,'PingFang SC',sans-serif;fill:#374151}
      .v{font:700 12px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .note{font:12px -apple-system,'PingFang SC',sans-serif;fill:#b45309}
    </style>
  </defs>
  <text x="20" y="20" class="ttl">核心原理：用「更大静态存储」换「更低 token 级激活」</text>
  <!-- 存储 -->
  <text x="20" y="52" class="lab">静态存储（BF16，GiB）</text>
  <rect x="170" y="40" width="460" height="26" fill="#e5e7eb"/>
  <rect x="170" y="40" width="460" height="26" fill="#1d4ed8"/>
  <text x="640" y="58" text-anchor="end" class="v" fill="#fff">335.3</text>
  <rect x="170" y="72" width="70.3" height="26" fill="#c2410c"/>
  <text x="246" y="90" text-anchor="start" class="v">51.7</text>
  <text x="20" y="78" class="lab" fill="#1d4ed8">Flash</text>
  <text x="20" y="110" class="lab" fill="#c2410c">27B</text>
  <!-- 激活 -->
  <text x="20" y="150" class="lab">每 token 激活（B 参数）</text>
  <rect x="170" y="138" width="480" height="26" fill="#e5e7eb"/>
  <rect x="170" y="138" width="480" height="26" fill="#c2410c"/>
  <text x="656" y="156" text-anchor="end" class="v" fill="#fff">27</text>
  <rect x="170" y="170" width="106.7" height="26" fill="#1d4ed8"/>
  <text x="282" y="188" text-anchor="start" class="v">6</text>
  <text x="20" y="166" class="lab" fill="#1d4ed8">Flash</text>
  <text x="20" y="198" class="lab" fill="#c2410c">27B</text>
  <rect x="170" y="218" width="460" height="34" rx="8" fill="#fffbeb" stroke="#f59e0b"/>
  <text x="184" y="240" class="note">存储 Flash 是 27B 的 6.48×，但激活仅 6B（≈27B 的 22%）。"Flash" 省的是每 token 计算/显存访问，不是下载体积。</text>
</svg>
<figcaption>图：同刻度对比（存储以 335 GiB 为满刻、激活以 27B 为满刻）。Flash-Next 用 6.48× 的静态存储换到远低于 27B 的 token 级激活。</figcaption>
</figure>

仅按静态权重（不含运行时/激活/KV cache/框架开销/碎片）：27B ≈ **51.75 GiB**（接近单卡/少卡部署）；Flash-Next ≈ **335.28 GiB**（通常需多卡、多机、量化或 Host Memory offload）。实际所需设备内存不能等价于"激活 6B"——专家权重与 N-gram 表仍需在某处可访问。

- **分片数（131/18）≠ GPU 数 / TP 度**：运行时切分由推理框架另定。
- **框架兼容**：工作区 `vllm-v0.21.0` 有 `qwen3_5` / `qwen3_5_mtp` 路径（贴近 27B），但**未检索到 `qwen4_exp` / Flash-Next 专用实现**——Flash-Next 须按模型卡最新 vLLM/SGLang recipe 实测，不能假设直接加载。
- **长上下文**：两者原生 262,144 tokens，均可用 YaRN 扩至 1M；YaRN 不改静态权重大小，但显著增大 KV cache 与运行时内存。

## 8. 选型判断

| 选择倾向 | 更适合 | 原因 |
|---|---|---|
| 本地部署门槛、存储、显存 | **Qwen3.8-27B** | ≈55.6 GB BF16，稠密结构直接 |
| 更大条件容量、更低每 token 激活 | **Qwen3.8-Flash-Next** | 125B 主模型 + N-gram 扩展，≈6B 激活 |
| 最小化下载/显存 | 对应 **-FP8** 版 | 本文两 URL 为非量化 BF16 |

<div class="keybox">
<strong>一句话记忆：</strong>Qwen3.8-27B 是"单套稠密 FFN 的 27B 模型"；Qwen3.8-Flash-Next 是"较窄主干 + 512 专家 + 51B N-gram 查表 + 特殊残差/注意力"的大容量稀疏体。前者权重小而直接，后者用更大的静态存储换更低的 token 级激活——文件 6.48×、激活 0.22×。
</div>

---

*下一篇预告：Flash-Next 的 512 专家 + N-gram 查表在推理框架里如何切分与 offload（EP / Host Memory 预取 / 量化），以及它和 DeepSeek V4、MiniMax M3 的"注意力改造"路线有何本质不同。*
