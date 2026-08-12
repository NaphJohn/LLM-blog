---
title: vLLM & SGLang 社区跟踪 · 2026-08-07：KV 缓存复用进入「组合正确性」阶段（HiCache / Mooncake 原理与架构）
description: LLM Infra Daily 第 7 期。窗口 08-05~08-07：vLLM / SGLang 各 100+ commits；主线是「前缀缓存 × 分层存储 × 投机解码 × PD 分离」正交组合时的静默正确性缺陷。深度拆解 HiCache 分层缓存与 Mooncake 分布式 KV 存储的原理、演进与架构。
pubDate: 2026-08-12
series: 社区跟踪手记
lang: zh
altLang: en
altHref: /en/blog/tr2-vllm-sglang-20260807
layout: ../../layouts/BlogPost.astro
---

## ★ 今日最值得关注

**SGLang #30393「HiCache 支持 packed / sidecar 草稿缓存」——它不是性能优化，而是一次隐性正确性缺陷的披露。** 凡是同时开了分层 KV 缓存（HiCache L2/L3）与投机解码（MTP / EAGLE / DSpark）的生产环境，此前都可能在「缓存命中率完全正常」的情况下，因草稿侧状态未被恢复而白白损失一截接受长度——指标不报警、日志不报错、只是吞吐上不去。今天就能做的事：对比升级 main 前后的 `accept_length`。

次选关注：**SGLang #28836（torch 2.13 大升级，提前排期）** 与 **vLLM #49206（PRIORITY 调度静默丢请求，正确性级修复）**。

## 一、版本基线（三方均无新 tag）

| 对象 | 最新稳定版 | 发布时间 | 上一版 |
|------|-----------|----------|--------|
| vLLM | v0.26.0 | 2026-07-27 | v0.25.1（07-14） |
| SGLang | v0.5.16 | 2026-07-25 | v0.5.15.post1（07-14） |
| 阶跃 Step | Step-3.7-Flash / 3.5-Flash | 仓库推送停 06-01 / 04-03 | — |

本期动态全部来自 main 分支：vLLM 100+ commits、SGLang 100+ commits（48h 跑满单页上限）。

## 二、本期主线：KV 缓存复用进入「组合正确性」阶段

过去一年，前缀缓存 / KV 卸载的竞赛主题是**命中率**——能省多少重算。本期两家在同一窗口暴露并修掉了同一类问题：**缓存复用与其他特性正交组合时会「静默出错」**。

- SGLang #30393：HiCache 只恢复目标模型 KV、不恢复草稿侧状态时，前缀会「报告命中」但投机接受率下降。
- SGLang #30545：PD 分离的 staging buffer 与 radix cache 组合时，命中会让首个分片越过网格槽位、把后续所有 chunk 索引全部错位。
- vLLM #50507：补上块内尾部前缀的细粒度复用（Mamba 混合模型场景）；#48069 / #44956：给 Mooncake 加租户与分组语义。

> **一句话**：缓存不再是「显存里一块 buffer」，而是一套要和投机解码、PD 分离、多租户同时正确共存的**存储系统**。

## 三、🔬 原理深度：HiCache 与 Mooncake 是什么、怎么演进

### 3.1 HiCache —— SGLang 的分层 KV 缓存

**定位**：把 KV cache 从「显存内一块 buffer」扩展成**跨层级存储**——L1=GPU 显存、L2=主机内存、L3=远端存储（如对象存储 / 分布式 KV 池）。命中层级越低，省的重算越多；跨节点复用前缀时，L2/L3 让多副本部署不必各自重算长系统提示词。

**核心结构**：前缀匹配在 **radix tree** 上按 token 序列做——判定依据是「目标模型的 KV 是否可用」。缓存以「主机池（host pool）」为单位在 L2/L3 间整体搬运。

**本期关键演进（#30393）**：投机解码引入了**第二套状态机**，HiCache 此前只搬运目标池，导致草稿侧状态缺失。本 PR 给出两条集成路径：

| 路径 | 适用拓扑 | 主机池布局 |
|------|----------|-----------|
| **Packed（打包）** | 标准 NextN MTP/EAGLE：DeepSeek-V3.2/V4、GLM-5.x、MiMo-V2.5；含 DeepSeek-V4 DSpark | 草稿 KV / indexer / SWA 缓冲追加到目标主机池尾部层，同槽位、同一次 HiCache 操作搬运 |
| **Sidecar（挎斗）** | 独立 EAGLE/EAGLE3、DFlash、非 DeepSeek-V4 的 DSpark | 仅为非空草稿状态建独立 DRAFT / DRAFT_INDEXER / DRAFT_SWA 池，索引由目标 KV/SWA 派生，挂到同一次 L2/L3 操作 |

> 约束：草稿缓存的索引空间必须能从目标缓存索引空间**派生**，否则一次搬运无法对齐两边。Packed 用于「草稿层就是目标模型多出来的几层」；Sidecar 用于结构上独立的草稿模型（层数/维度对不上）。只为非空草稿状态建 sidecar——不开投机时零开销。

### 3.2 Mooncake —— 分布式 KV 存储（vLLM 侧主推）

**定位**：Mooncake 是 vLLM 的 **KV 连接器（KV connector）** 生态里的分布式 KV 存储标准件，把 KV cache 托管到独立集群，让多个 vLLM 部署共享一套前缀缓存，实现 PD 分离与跨部署复用。

**本期关键演进（#48069 / #44956）**：补齐**多租户与分组语义**。

- **#48069 租户命名空间**：从 Mooncake JSON 配置读取 `tenant_id` 并透传给 `MooncakeDistributedStore.setup()`，不同 vLLM 部署可用独立命名空间（缺省归一为 `default` 且不传参，向后兼容；老版 Mooncake 配了非默认租户会明确报 `RuntimeError`）。
- **#44956 分组语义（enable_group_semantics）**：让属于同一条逻辑 KV 条目的多个物理对象带共享 `group id`，Mooncake 侧可做**分组感知的元数据路由、租约刷新与淘汰**，显著减少「同一条缓存被拆散淘汰」的碎片化。

**部署影响**：共用一套 Mooncake 集群跑多套 vLLM（多业务线 / 多环境）终于有官方隔离手段，此前只能靠 key 前缀土办法。#51067 进一步把 Docker 改用 Mooncake **官方 wheel** 而非自建构建——Mooncake 已从「某个连接器实现」变成两大框架共同依赖的标准件，这是生态收敛信号。

### 3.3 为什么「分层缓存 × 投机解码」会静默掉性能

一次「前缀命中」在指标上是**布尔**的。但在投机解码系统里，命中其实有两种质量：

- **完整命中**：目标 KV + 草稿状态都在。
- **残缺命中**：只有目标 KV。

残缺命中不会报错、不会降命中率指标，只会让**接受长度悄悄变短**，最后表现为「缓存命中率很高，但吞吐没涨」。

**投机解码到底缓存了什么**（第二套状态机）：

- 草稿模型 KV —— EAGLE/EAGLE3/DFlash 是独立小模型，有自己的层、自己的 KV；
- DSA indexer 状态 —— DeepSeek 稀疏注意力的索引结构，决定 Top-K 选哪些 token；
- SWA 状态 —— 滑动窗口注意力的窗口内缓存。

这些在 SGLang 里放在与目标 KV **分离的设备池**。HiCache 做 L2/L3 分层时若只搬目标池，草稿池要么没备份、要么恢复回来时 slot 映射对不上。前缀匹配按目标 KV 判定「这段算过」→ 跳过 prefill 直接 decode → 草稿模型拿空/错位的草稿 KV 提议 → 大量被拒 → `accept_length` 从 4~5 掉到 1~2，`cache_hit_rate` 漂亮但 TPS 原地踏步。**没有任何日志告诉你哪里错了。**

**同一天、同一类 bug 的另两个版本**：

- #30545（PD 分离 × radix cache）：staging 按 `chunk_idx = start_page // full_chunk_pages` 定位分片，前提是均匀网格；radix 一命中，首次发送撑过一格，之后每个 chunk 索引全错。修法：prefill 侧向下取整到网格、decode 侧完全由到达驱动 scatter。
- #50507（Mamba 混合模型）：物理块很大、卸载只能整块存取，块尾算好的 token 白丢（900 token 提示词丢 112 个）。假设「块是复用最小单位」被「块变很大」打破。

> **可迁移的判断**：特性矩阵已大到无法穷举组合测试（前缀缓存 × 分层存储 × 投机解码 × PD 分离 × 混合架构 × 异构 TP）。这类 bug 共同形态是「A 模块的隐含假设被 B 模块打破，且系统没有断言去检查」。对使用方：**不要假设两个特性都标了 stable 一起开就没事**；每引入一个新特性，都把 `accept_length`、`cached_tokens`、`TTFT` 做一次前后对比，而不只是看有没有报错。

## 四、🟢 vLLM 要点（main 08-05 ~ 08-07）

- **KV/PD #48069 / #44956**：Mooncake 多租户 + 分组语义（见 3.2）。
- **性能 #50507 / #50992**：KV 卸载支持「块内尾部」前缀细粒度复用；Attention–Mamba 混合模型（Qwen3.6）按 `prefix_match_unit` 恢复尾部片段、对 Attention KV 与 Mamba 循环状态分别选对源块、copy-on-write 续写。实测 Qwen3.6-27B / 900 token 提示词，命中 784 → 896 token（多救回 112 ≈ 14%）。#50992 消除 ARC 批量淘汰的二次方复杂度。
- **多模态 #50390**：EPD 去掉 decode 侧重复图像预处理并把预处理搬上 GPU（256²→2048² 加速 2.3~8.6×）。关键：encoder 实例不分配 KV cache，故前端不与语言模型抢资源——正是「把 E 拆出来」后才敢做的优化。
- **MTP/SP 内核提速 #50904（2.0×）/ #51070（1.5~3×）/ #50230**：#50904 修 MTP 走投机路径时 `set_skip_topk(True)` 未真正跳过冗余重算（只改 7 行）；#51070 把 Kimi-K3 SP 的多次 all-gather 合并为最终一次；#50230 启用 PDL 减少 kernel 启动串行等待。均只在 main，需等 v0.26.1 或自行 cherry-pick。
- **服务化 #51089 / #49206 / #50289**：HTTP header 解析请求优先级；#49206 修复 PRIORITY 调度下请求被**静默跳过**的严重缺陷（正确性级修复，建议尽快评估）。
- **模型/硬件 #51045 / #49453 / #47106**：Ling 3.0 Flash 全栈支持（BF16 + MTP + parser 一次性合入主线）；CPU MLA 后端让 DeepSeek-V2/V3 纯 CPU 跑；NVFP4 CuTeDSL MoE 支持 `swiglu-oai` / `relu²` 激活（正是 08-01 点名的 Step-3.5 NVFP4 内核缺口）。

## 五、🟠 SGLang 要点（main 08-05 ~ 08-07）

- **重大依赖变更 #28836**：CUDA PyTorch 栈整体升 torch 2.11→2.13、triton 3.6→3.7.1 等（34 文件）。下一个稳定 tag 大概率带 torch 2.13。**有自研 Triton 算子 / 魔改 kernel 的团队，建议现在就基于 main 做兼容性预演。**
- **KV×投机 #30393**：HiCache 支持 packed/sidecar 草稿缓存（见 3.1）。同时开 HiCache + 投机解码的生产环境，此前存在隐性接受率损失。
- **PD 分离 #30545**：Disagg staging buffer 终于能和 radix cache 共存（见 3.3）。
- **多模态 #32365（35 文件）**：rust-server 原生多模态端到端跑通 Qwen VL，`SGLANG_RUST_SERVER=1` 即端到端、请求路径无 Python mm_processor、无回退。架构关键决定：**MM 输出特征绝不经过 ingress**——大缓冲（features/grids/hashes/M-RoPE）停在按 `rid` 索引的 sidecar，ingress 只传 `MmEncoded{rid, input_ids}`，Python 调度器 `take_mm(rid)` 零拷贝取走。
- **扩散 #33823 等 15+ 提交**：SGLang-Diffusion 形成独立产品线（FLUX.2 残差门控 −1.2%、FLUX.1 −1.1% 无损、ERNIE-Image e2e 15.63→15.00s 等，均 bit-exact）。SGLang 正从「LLM 推理框架」变成「生成式负载统一服务层」。
- **默认值变更 #33618**：MoE deferred finalize 默认开启（升级后行为会变，值得回归）；#33459 DFlash 支持 logprobs；#33138 cache_aware router 引入随机平局打破。

## 六、🇨🇳 阶跃星辰 Step：MTP 主线仍卡在 open PR

**vLLM #49642（仍 open，+18/−5，两周未合）**：Step3p5AMultiTokenPredictor 在 `range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)` 上构造草稿块，即 MTP 层 `layer_idx == num_hidden_layers`，比基础解码层多一位；而 `step3p5.py` 若干按层查表的配置（layer_types / rope_theta / use_rope_layers / partial_rotary_factors / swiglu_limits）都是长度 `num_hidden_layers` 的列表且无越界检查 → 在 Step-3.7-Flash 上加 `--speculative-config '{"method":"mtp"}'` 直接 `IndexError` 起不来。修法：对每处按 `layer_idx` 查表加边界判断，越界草稿层退回标准默认。

**为什么重要**：Step-3.7-Flash 三大卖点是「混合滑窗注意力 + MTP 自带草稿 + 稀疏 MoE（196B 总参 / 11B 激活）」，MTP 恰恰是官方宣传吞吐的关键，却在 vLLM 主线上开不起来。同一窗口内 vLLM 给蚂蚁 Ling 3.0 Flash 一次性合入「BF16 + MTP + parser」全套。**差距不在模型能力，在上游维护投入。**

**行动建议**：① 现在跑 Step-3.7-Flash 先别开 MTP，或自行 cherry-pick #49642；② 要 MTP 收益短期用 SGLang 路线（`--reasoning-parser step3p5 --tool-call-parser step3p5` + NVFP4）更稳。

## 七、⚖️ 横向对比

| 维度 | vLLM | SGLang |
|------|------|--------|
| KV 复用主攻 | 外部存储生态：Mooncake 租户/分组语义/官方 wheel、块内尾部细粒度复用 | 内部层级耦合：HiCache 打通投机草稿状态 + 可观测性增强 |
| 多模态提速 | 拆物理拓扑：E/P/D 分离消灭重复变换、预处理上 GPU（1.9~8.6×） | 换执行语言：整条视觉管线纯 Rust、worker pool + rid sidecar 零拷贝 |
| 对 Python 态度 | 保留回退与 transformers 兼容路径，广度优先 | 不支持即启动失败、不留回退，确定性优先 |
| 依赖节奏 | FlashAttention 转 torch stable ABI，渐进解耦 | torch 2.11→2.13 一次性大升，激进跟进 |
| 产品边界 | 聚焦 LLM/VLM 服务，前端 Rust 化 + 调度治理 | 向扩散模型大幅扩张（FLUX/ERNIE-Image/Ideogram/Z-Image + dp-size） |
| Step 适配 | 官方 recipe 完备，但 MTP 开不起来（#49642 open） | NVFP4 + trtllm_mha + 双 parser 可用，社区活跃度停 07-06 |

**一句话对照**：本期两家解决同一个问题的两半——「请求进 GPU 之前和缓存出显存之后的那段路」。vLLM 手法是重新切分物理资源（encoder 拆独立节点、预处理上 GPU、KV 托给外部多租户存储）；SGLang 手法是重写执行载体（管线搬进 Rust、草稿状态纳入分层缓存、PD 传输对齐网格）。选型：要广度、快速接新模型选 vLLM；要尾延迟确定性、且扛得住 torch 2.13 大升级节奏，选 SGLang。

> 数据来源：GitHub API 直查 vLLM / SGLang releases、main commits、指定 PR（#30393 / #30545 / #28836 / #48069 / #44956 / #50507 / #49206 / #51045 / #49642 等）。性能数字与增删行数均取自官方仓库与 PR 描述原文；判断与选型建议为分析观点，请结合自身负载实测。
