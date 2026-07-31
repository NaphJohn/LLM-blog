---
title: （六）支持的模型与选型指南：什么场景该选谁
description: 模型矩阵、硬件矩阵、量化矩阵与选型决策树，附 Step 3.7 Flash、GLM-5.2、DeepSeek-V4 等主流模型的实际部署要点。
pubDate: 2026-08-01
series: vLLM 与 SGLang 框架解码手记
lang: zh
altLang: en
altHref: /en/blog/fw6-model-support-selection
layout: ../../layouts/BlogPost.astro
---

## 1. 模型支持矩阵

先说结论：**两家对主流模型的覆盖都很好，差异在"新模型多快能用"和"深度调优做到什么程度"。**

| 模型 | 规模 / 结构 | vLLM | SGLang | 备注 |
|---|---|---|---|---|
| **DeepSeek-V4 / V4 Pro** | MoE + MLA + DSA | 一等，**DSpark 投机原生** | 一等，cookbook 官方基准 | 8×B300 约 250 tok/s，DSpark 比 MTP 高 12–42% |
| **GLM-5.2** | MoE + DSA | 生产方案：NVFP4+MTP+P/D | **深度调优标杆** | Blackwell 上 500+ tok/s/user；PCP 让 prefill 20.1k→27.3k |
| **Qwen3 / Qwen3.5 / Qwen3-VL** | 稠密 + MoE + VLM | 一等（含 Transformers 后端 M-RoPE） | 一等 | 注意 GPTQ + 投机的组合坑（#48816） |
| **Step 3.7 Flash** | 196B MoE / 激活 11B / 256K | **预建镜像** `vllm/vllm-openai:stepfun37` | dev 镜像 `lmsysorg/sglang:dev-step-3.7-flash` | FP8 / BF16 / NVFP4 + MTP / EAGLE |
| **Step-3.5-Flash** | 196B MoE + 3:1 滑窗 | **官方 recipe 部署指南** | 支持 | Int4 权重 vLLM 暂不支持 |
| **Step3-VL-10B** | 10B 端侧 VLM | nightly ≥ 0.14.0rc2 | latest main + cookbook | 单张 RTX 4090 可跑，AIME2025 94.43% |
| **腾讯 Hy3** | 295B MoE | 即日支持 | 即日支持 | 国产模型 day-one 适配案例 |
| **Llama / Mistral / Gemma 系** | 稠密 | 全覆盖 | 全覆盖 | Gemma4 注意 NVFP4 需 0.25.1+ |
| **Cosmos3 Edge 等视频模型** | 多模态 | 支持（#49190 修复） | — | vLLM 多模态覆盖更广 |
| **HF 上刚发布的新架构** | 任意 | **Day-0 全速**（Transformers parity） | 需等原生实现 | vLLM 明显优势 |

<div class="keybox">
<strong>最重要的一条差异：</strong>vLLM 的 <code>Transformers backend parity</code> 意味着<strong>只要 HF 上有实现，新模型当天就能全速服务</strong>，不用等框架写原生 kernel。追新模型的团队，这一条往往就决定了选型。
</div>

## 2. 硬件支持矩阵

| 硬件 | vLLM | SGLang | 备注 |
|---|---|---|---|
| **NVIDIA Hopper (H100/H200)** | ✅ 成熟 | ✅ 成熟 | FP8 原生 |
| **NVIDIA Blackwell (B200/B300/GB200)** | ✅ FA4 + SM100 FP8 KV | ✅ NVFP4 深度优化 | SGLang 在 NVFP4 上更激进 |
| **消费级 Blackwell (SM120)** | ✅ | ✅ #30272 DeepSeek-V4 mxfp4 MoE + TP2 | 单卡/双卡玩家路径 |
| **AMD ROCm** | ✅ AITER v0.1.16.post5 | ✅ MXFP4 (#28291) | vLLM 覆盖更全 |
| **Intel XPU** | ✅ DeepSeek-V4 `fuse_index_q` SYCL | 有限 | **vLLM 独有优势** |
| **TPU / CPU** | ✅ | 有限 | vLLM 广度领先 |

**结论**：硬件广度上 **vLLM 显著领先**；但在 Blackwell + NVFP4 这个最前沿的组合上，**SGLang 的调优更深**。

## 3. 选型决策树

<div class="fig">
<svg viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <rect x="240" y="10" width="200" height="34" rx="17" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="340" y="32" font-size="12" font-weight="700" fill="#374151" text-anchor="middle">选 vLLM 还是 SGLang？</text>

  <path d="M340 44 L340 62" stroke="#9ca3af" stroke-width="1.5"/>
  <rect x="216" y="62" width="248" height="34" rx="6" fill="#fffbeb" stroke="#fcd34d"/>
  <text x="340" y="84" font-size="11" fill="#92400e" text-anchor="middle">请求之间共享长前缀吗？（system/多轮/评测）</text>

  <path d="M240 96 L120 122" stroke="#10b981" stroke-width="1.5"/>
  <text x="150" y="112" font-size="10" fill="#047857">是，共享率高</text>
  <path d="M440 96 L560 122" stroke="#6b7280" stroke-width="1.5"/>
  <text x="500" y="112" font-size="10" fill="#6b7280">否 / 不确定</text>

  <rect x="20" y="124" width="200" height="34" rx="6" fill="#ecfdf5" stroke="#6ee7b7"/>
  <text x="120" y="146" font-size="11" font-weight="700" fill="#047857" text-anchor="middle">优先 SGLang</text>

  <rect x="460" y="124" width="200" height="34" rx="6" fill="#fffbeb" stroke="#fcd34d"/>
  <text x="560" y="146" font-size="11" fill="#92400e" text-anchor="middle">要跑刚发布的新模型吗？</text>

  <path d="M120 158 L120 178" stroke="#10b981" stroke-width="1.5"/>
  <rect x="20" y="180" width="200" height="76" rx="6" fill="#f0fdf4" stroke="#86efac"/>
  <text x="34" y="198" font-size="10" fill="#065f46">· Agent / 多轮对话</text>
  <text x="34" y="214" font-size="10" fill="#065f46">· 批量评测 / 思维树</text>
  <text x="34" y="230" font-size="10" fill="#065f46">· 严格结构化输出</text>
  <text x="34" y="246" font-size="10" fill="#065f46">· 大规模 EP · NVFP4 极致调优</text>

  <path d="M510 158 L440 182" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="452" y="176" font-size="10" fill="#1d4ed8">是</text>
  <path d="M610 158 L610 182" stroke="#6b7280" stroke-width="1.5"/>
  <text x="618" y="176" font-size="10" fill="#6b7280">否</text>

  <rect x="330" y="184" width="180" height="34" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="420" y="206" font-size="11" font-weight="700" fill="#1d4ed8" text-anchor="middle">选 vLLM（Day-0 全速）</text>

  <rect x="520" y="184" width="140" height="34" rx="6" fill="#fffbeb" stroke="#fcd34d"/>
  <text x="590" y="206" font-size="10" fill="#92400e" text-anchor="middle">非 NVIDIA 硬件？</text>

  <path d="M560 218 L500 242" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="506" y="236" font-size="10" fill="#1d4ed8">是</text>
  <path d="M630 218 L630 242" stroke="#6b7280" stroke-width="1.5"/>
  <text x="638" y="236" font-size="10" fill="#6b7280">否</text>

  <rect x="380" y="244" width="180" height="30" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="470" y="264" font-size="11" font-weight="700" fill="#1d4ed8" text-anchor="middle">选 vLLM（ROCm/XPU/TPU）</text>

  <rect x="570" y="244" width="90" height="30" rx="6" fill="#f9fafb" stroke="#d1d5db"/>
  <text x="615" y="264" font-size="10" fill="#374151" text-anchor="middle">两家都行</text>

  <rect x="20" y="290" width="640" height="96" rx="8" fill="#faf5ff" stroke="#c4b5fd"/>
  <text x="340" y="310" font-size="12" font-weight="700" fill="#6d28d9" text-anchor="middle">实践建议：不是单选题</text>
  <text x="36" y="332" font-size="11" fill="#5b21b6">· 两家都提供 OpenAI 兼容 API → 切换成本低，值得<tspan font-weight="700">用自己的真实流量各压测一轮</tspan>再定</text>
  <text x="36" y="352" font-size="11" fill="#5b21b6">· 大规模场景可以<tspan font-weight="700">混部</tspan>：Agent 主链路走 SGLang（吃前缀复用），长尾/新模型走 vLLM（吃覆盖广度）</text>
  <text x="36" y="372" font-size="11" fill="#5b21b6">· 压测必须看 <tspan font-weight="700">TTFT / P99 TPOT / 吞吐</tspan> 三个数字，只看总吞吐会选错</text>
</svg>
</div>

## 4. 一句话选型口诀

<div class="keybox">
<strong>重共享前缀 → SGLang；要模型 / 硬件广度 → vLLM。</strong>
</div>

展开一点：

**倾向 SGLang 的信号**

- 主力场景是 Agent、多轮对话、ReAct 循环（system prompt + 工具定义每步重发）
- 批量评测、思维树、自洽性采样（同一前缀大量分叉）
- 严格 JSON Schema / 正则约束输出，且占比高
- 跑 DeepSeek / GLM 这类 MoE + MLA/DSA 模型，想要 NVFP4 极致吞吐
- 需要大规模专家并行（EP）部署

**倾向 vLLM 的信号**

- 模型种类杂，且经常要上刚发布的新架构
- 硬件不是纯 NVIDIA（ROCm / Intel XPU / TPU）
- 量化格式需求杂（AWQ / GPTQ / FP8 / NVFP4 / MXFP4 混用）
- 要接入现成生态：KV Connector、Rust router、Dynamo、llm-d
- 团队更看重"版本稳、文档全、坑有人踩过"

## 5. 典型模型的部署要点

### 5.1 Step 3.7 Flash（196B MoE / 激活 11B / 256K）

- **vLLM**：用官方预建镜像 `vllm/vllm-openai:stepfun37` 最稳，支持 MTP 投机 + NVFP4 4 卡部署
- **SGLang**：`lmsysorg/sglang:dev-step-3.7-flash` + EAGLE
- 两框架适配都已"一等成熟"，但**深度调优与公开 benchmark 的成熟度仍落后 GLM / DeepSeek**
- 模型侧要求 `transformers ≥ 5.0`（自研建模，走 `trust_remote_code`）

### 5.2 Step-3.5-Flash：为什么又快又省

这个模型是"三手段叠加"的好例子：

1. **3:1 混合滑窗注意力**：大部分层用滑窗，把主体开销从 O(n²) 降到 O(n·w)，少量全局层负责长程信息传播；
2. **MTP 自带草稿头**：一步出约 4 个 token，等于自带投机解码；
3. **稀疏 MoE**：196B 总参，每 token 只激活 11B——**显存吃总参，算力吃激活**。

<div class="warnbox">
部署注意：vLLM 上 <strong>MoE / MTP 相关优化并非默认全开</strong>，需要按官方 recipe 显式配置；<strong>Int4 权重 vLLM 暂不支持</strong>。
</div>

### 5.3 GLM-5.2：当前调优最深的组合

生产方案是三件套叠加：**NVFP4 量化 + MTP 投机 + P/D 分离**。

- `IndexerCache` 提升 MTP 接受率
- **PCP（上下文并行）把 prefill 吞吐从 20.1k 提到 27.3k**
- SGLang 侧：IndexShare MTP（草稿复用 top-k，长上下文降本 1.9×）+ TopK-V2（Lightning-TopK，优化 80k 级输入）
- 结果：**Blackwell 上 500+ tok/s/user**
- 建议锁 SGLang **v0.5.15.post1** 或更高

### 5.4 Step3-VL-10B：端侧多模态的落地点

- 10B VLM（PE-lang 1.8B + Qwen3-8B），**单张 RTX 4090 可跑**（BF16 / FP8）
- AIME2025 94.43%
- 需要 vLLM nightly ≥ 0.14.0rc2 或 SGLang latest main（有官方 cookbook）
- 依赖 nightly / main，**生产必须锁版本**
- 这是目前国产端侧多模态推理最值得跟的落地点

## 6. 部署实践检查清单

**上线前**

- [ ] 用**真实流量分布**压测，而不是固定长度的合成请求
- [ ] 三个指标都测：TTFT、P99 TPOT、总吞吐
- [ ] 量化模型跑**输出正确性回归**（NVFP4 乱码是静默的）
- [ ] 确认前缀共享率——决定 RadixAttention 能不能给你带来收益
- [ ] 锁定 release tag，不要跟 main

**调参顺序建议**

1. 先把 `--max-model-len` 设成真实需要的值（直接决定 KV 预算）
2. 开前缀缓存 / RadixAttention（几乎无副作用）
3. 调 `--max-num-seqs` 与 `gpu-memory-utilization` 找吞吐拐点
4. 再开投机解码（MTP / EAGLE），观察接受率是否 > 60%
5. 最后才考虑量化和 PD 分离（收益大但复杂度也大）

**什么时候不需要 PD 分离**

- 单机 8 卡以内、中小模型 → 混合部署更简单
- prompt 普遍很短 → prefill 本来就不构成干扰
- 没有 NVLink / RDMA 高速互联 → KV 传输会吃掉收益

## 7. 系列总结

六篇走下来，这条主线大概是这样：

| 篇 | 主题 | 一句话 |
|---|---|---|
| 一 | 为什么需要引擎 | 朴素推理浪费在"显存空洞、槽位空转、GPU 空等" |
| 二 | vLLM 原理 | 分页 KV + 连续批处理起家，靠 MRv2 消灭同步、靠广度立身 |
| 三 | SGLang 原理 | 从"LLM 应用是有结构的程序"出发，靠前缀复用与前沿吞吐优化立身 |
| 四 | 共同前沿 | 同步停顿、投机解码、PD 分离、低比特量化四条战线互相叠加 |
| 五 | 版本演进 | 2026 年 7 月两家几乎同步完成架构换代，大版本后必有修复潮 |
| 六 | 模型与选型 | 重共享前缀选 SGLang，要广度选 vLLM，最好用真实流量各压一轮 |

<div class="keybox">
最后一个观察：这两个引擎的竞争<strong>对使用者是纯粹的好事</strong>。一家做出 MRv2，另一家很快就有 Spec V2；一家上 NVFP4，另一家马上跟 NVFP4_AWQ。它们都提供 OpenAI 兼容接口，<strong>切换成本很低</strong>——所以不要把选型当成一次性的终身决定，定期用自己的真实流量重新压一轮就好。
</div>

> 相关阅读：投机解码的算法原理见[推测解码手记](/LLM-blog/blog/ep1-speculative-decoding)；模型侧的 MoE、注意力与多模态结构见[多模态解码手记](/LLM-blog/blog/mm5-efficiency-frontier)。
