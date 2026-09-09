---
title: 前沿架构解码手记（八）：Qwen3.8 三版本同框——2.4T 旗舰、Flash 服务版与 Flash-Next 架构预览
description: 把 Qwen3.8 家族三条产品线一次讲清：Flash-Next 是基于下一代 Qwen4 架构的先导预览（125B MoE + 51B N-gram、每 token 仅激活 6B、3×GDN+1×QSA 交替、Muon 优化器、训练成本约为 Qwen3.7-Plus 的 1/9）；Flash 是同架构的服务调优版（权重未开放，$0.16/$0.47 每百万 token）；2.4T-A95B 是当前代旗舰（512 专家 10+1、每 token 激活 95B、1M 上下文、$2/$6 无长提示附加费）。附三版本对照表与结构拆解图。
pubDate: 2026-09-09
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa8-qwen38-three-tiers
layout: ../../layouts/BlogPost.astro
---

## 0. 结论先行

Qwen3.8 家族不是"三个大小的模型"，而是**三条产品线各干一件事**：

| 版本 | 本质 | 一句话 |
|------|------|--------|
| **Flash-Next** | 下一代 Qwen4 架构的**先导预览**（2026-08-26 开源） | 用 6B 激活试探新架构的方向 |
| **Flash** | 同架构的**服务调优版**（权重未开放） | 承载 API 定价的生产模型 |
| **2.4T-A95B / Max** | **当前代旗舰**（开放权重 8/12 发布） | Qwen-Max 级能力首次开放权重 |

一句话记忆：**Next 是"告诉你下一代长什么样"，Flash 是"你现在能便宜用到的"，2.4T 是"当前最强且你能自己下载的"**。

## 1. 一张表看懂三条产品线

| 维度 | Flash-Next | Flash | 2.4T-A95B（Max） |
|------|-----------|-------|------------------|
| **定位** | 架构预览（Qwen4Exp） | 服务调优生产版 | 当前代旗舰 |
| **总参数** | 125B MoE + 51B N-gram（约 180B 存储） | 同 Flash-Next 架构 | **2.4T** 稀疏 MoE |
| **每 token 激活** | **约 6B（<5%）** | 同左 | **约 95B（约 4%）** |
| **注意力** | **QSA 稀疏注意力**（与 GDN 3:1 交替） | 同左 | 门控注意力（GDN 配套） |
| **专家** | 512 专家 MoE | 同左 | 512 专家（**10 路由 + 1 共享**） |
| **上下文** | 原生 262,144 → YaRN 至 1M | 同左 | 1M |
| **模态** | 文本 / 图像 / 视频 | 同左 | 多模态 |
| **权重** | ✅ 开放（qwen-community-1.0，BF16 + FP8） | ❌ 未开放 | ✅ 开放（2.4T-A95B，自定义许可证） |
| **API 定价** | 输入 1 元 / 输出 3 元 每百万 token | **$0.16 / $0.47** 每百万 token | **$2 / $6**，无长提示附加费 |
| **独立评测** | 多数基准超 DeepSeek-V4-Flash | — | Artificial Analysis 智能指数 **58**、编码 **71.8** |

> 训练成本：Flash-Next 约为 **Qwen3.7-Plus 的 1/9**——这是"小激活 + 新架构"路线最直接的证据。

## 2. 七件共用组件：源码层面是对齐的

把三个版本的仓库源码并排看，**七类组件的命名与组织是一致的**：GDN、QSA、MoE、RMSNorm、Residual、KV Cache、MTP。差别不在"有没有"，而在"参数与组合方式"：

<div class="fig">
<svg viewBox="0 0 660 300" role="img" aria-label="Flash-Next 与 2.4T 旗舰的结构对比">
  <title>图 1：Flash-Next（Qwen4 预览）与 2.4T-A95B（当前代旗舰）的结构对比</title>
  <text x="20" y="26" font-size="13" fill="#6b7280">Flash-Next / Flash — Qwen4 预览架构（每 token 激活约 6B）</text>
  <rect x="20" y="40" width="300" height="34" rx="6" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="170" y="62" font-size="12" fill="#534AB7" text-anchor="middle">N-gram Embedding 51B（驻留主机内存，非显存）</text>
  <rect x="20" y="81" width="300" height="34" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="170" y="103" font-size="12" fill="#185FA5" text-anchor="middle">GDN ×3 / 组（Gated DeltaNet 线性注意力）</text>
  <rect x="20" y="122" width="300" height="34" rx="6" fill="#f3e8ff" stroke="#7f77dd"/>
  <text x="170" y="144" font-size="12" fill="#534AB7" text-anchor="middle">QSA 稀疏注意力 ×1 / 组（3:1 与 GDN 交替）</text>
  <rect x="20" y="163" width="300" height="34" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="170" y="185" font-size="12" fill="#185FA5" text-anchor="middle">MoE 512 专家 → 每 token 约 6B 激活</text>
  <rect x="20" y="204" width="300" height="34" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="170" y="226" font-size="12" fill="#185FA5" text-anchor="middle">MTP 多词元预测头</text>
  <text x="360" y="26" font-size="13" fill="#6b7280">2.4T-A95B / Max — 当前代旗舰（每 token 激活约 95B）</text>
  <rect x="360" y="52" width="280" height="44" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="500" y="72" font-size="12" fill="#185FA5" text-anchor="middle">GDN（Gated DeltaNet）</text>
  <text x="500" y="88" font-size="11" fill="#6b7280" text-anchor="middle">线性注意力抓局部与全局状态</text>
  <rect x="360" y="108" width="280" height="44" rx="6" fill="#faeeda" stroke="#BA7517"/>
  <text x="500" y="128" font-size="12" fill="#854F0B" text-anchor="middle">门控注意力（当前代主力注意力）</text>
  <text x="500" y="144" font-size="11" fill="#6b7280" text-anchor="middle">非 QSA：仍是这一代已验证的路线</text>
  <rect x="360" y="164" width="280" height="44" rx="6" fill="#faeeda" stroke="#BA7517"/>
  <text x="500" y="184" font-size="12" fill="#854F0B" text-anchor="middle">MoE 512 专家（10 路由 + 1 共享）</text>
  <text x="500" y="200" font-size="11" fill="#6b7280" text-anchor="middle">每 token 约 95B 激活（约 4%）</text>
  <rect x="360" y="220" width="280" height="44" rx="6" fill="#e6f1fb" stroke="#378ADD"/>
  <text x="500" y="240" font-size="12" fill="#185FA5" text-anchor="middle">1M 上下文 · 多模态 · $2/$6 统一定价</text>
  <text x="500" y="256" font-size="11" fill="#6b7280" text-anchor="middle">不收长提示附加费</text>
  <text x="20" y="288" font-size="11" fill="#6b7280">蓝 = 两代共用组件 ｜ 紫 = Flash-Next 新增/替换（Qwen4 预览） ｜ 橙 = 旗舰当前代路线</text>
</svg>
<figcaption>图 1：两代架构共用 GDN / MoE / MTP 等骨架；Flash-Next 用 QSA 稀疏注意力与 51B 主存 N-gram 表换 6B 激活，旗舰走门控注意力 + 95B 激活的当前代路线。</figcaption>
</div>

**两代真正的分水岭在注意力**：旗舰（2.4T）用的是当前一代已验证的**门控注意力**；Flash-Next 换成了 **QSA 稀疏注意力**，并以 3:1 的比例与 GDN 交替。这就是"预览"的含义——**Qwen4 大概率会沿 QSA 这条线走**。

## 3. Flash-Next：用 51B 主存换 6B 激活

fa7 已拆过它的权重组织，这里只讲**为什么这个设计成立**：

- **125B 主模型 + 51B N-gram Embedding**，磁盘约 180B 参数（BF16 约 360GB）
- 关键取舍：**51B 的 N-gram 表不进显存，驻留主机内存**——用 PCIe 换 HBM
- 每 token 只激活约 **6B**（稀疏度约 95%），换来的是推理成本的数量级下降
- **Muon 优化器**训练，成本约为 Qwen3.7-Plus 的 **1/9**，但编程与办公任务反而更强
- 上下文原生 262,144，YaRN 扩展到 1M

**这套设计的隐含前提**：你得能接受"模型的一部分常驻主存"。对单卡用户不友好，对有 Host 内存的服务部署是划算买卖——这也解释了为什么它的 API 价格能做到输入 1 元 / 输出 3 元每百万 token，比 DeepSeek-V4-Flash 还低约三分之一。

## 4. Flash：同架构的服务调优版

Flash 的定位最容易误解——**它不是"小一号的 Flash-Next"，而是同一架构为生产环境调优的版本**：

- 权重**未开放**（这是与 Flash-Next 唯一但关键的区别）
- API 定价 **$0.16 / 百万输入**、**$0.47 / 百万输出**
- 承载 QwenCloud API 的生产流量

**为什么要有两个名字**：预览版（Next）承担"公开架构方向、收集社区反馈"的任务；服务版（Flash）承担"稳定供给"的任务。架构相同但服务等级、量化方案、上线节奏都可以分开管理。这也是为什么名字里带 "Next" 的模型**不该直接上生产**——官方文档也明确它是 architecture preview，工具链支持可能滞后。

## 5. 2.4T-A95B：Qwen-Max 级能力首次开放权重

这一版的历史意义大于参数意义：**Max 档一直是闭源托管的最强选项，这是第一次以开放权重放出**。

- **2.4T 总参数稀疏 MoE**，每 token 激活约 **95B**（约 4%，和 Flash-Next 的稀疏比例接近，但绝对量大 15 倍以上）
- 512 专家中 **10 个路由专家 + 1 个共享专家**——与 Flash-Next 的纯 512 专家结构不同
- API 侧 **1M 上下文**、多模态、**$2/$6 统一定价且不收长提示附加费**（长上下文场景这个定价策略很关键）
- 开放权重版 **Qwen3.8-2.4T-A95B**（8/12 发布）与 **FP8 变体**，自定义许可证（非 Apache）
- 独立评测：Artificial Analysis **智能指数 58、编码指数 71.8**，是 Qwen3.8 系列最强

## 6. 推理控制：两个被低估的参数

Qwen3.8 引入的两个 API 参数，对工程的影响比跑分更大：

| 参数 | 作用 | 工程价值 |
|------|------|----------|
| **`reasoning_effort`** | 按请求调节推理深度 | 分类/短文本不必深思考，省延迟与成本；多步调试才给深推理。**不用再为"便宜的活"和"难的活"配两个模型** |
| **`preserve_thinking`** | 跨轮保留思考上下文 | 迭代会话中模型不再每轮重新推导一遍相同结论——省 token，也避免二次推导结果漂移。长 Agent 会话更稳也更便宜 |

两者指向同一个设计目标：**为多步任务优化，而不是为单次惊艳的回答优化**。

## 7. 选型判断

| 你的场景 | 选 |
|----------|-----|
| 自己部署、想试下一代架构、能接受 360GB 权重 + 主存 N-gram 表 | **Flash-Next**（BF16 或 FP8） |
| 追求最低 API 成本、日常编程与办公任务 | **Flash**（$0.16/$0.47） |
| 需要当前最强能力 + 1M 上下文 + 多模态，且要自己部署 | **2.4T-A95B**（显存预算要按 95B 激活算，不是 2.4T） |
| 本地硬件有限、想跑得动 | **27B**（稠密，55.6GB BF16，社区下载量最高） |

**一个提醒**：2.4T 的显存估算别按 2.4T 算——按激活 95B 算 KV Cache 与激活值（见 sys9 的公式），权重才按总参数摊。

## 8. 与系列文章的衔接

- **fa6（Gated DeltaNet）**：本文三版本共用的线性注意力层，原理在那篇。
- **fa7（双 checkpoint 对比）**：Flash-Next 与 27B 的权重组织、config 差异与存储拆解，本文不重复。
- **sys9 / sys10**：算这三个模型的 KV Cache 与部署成本，用那两篇的公式与工具。
- **ag1**：`preserve_thinking` 正是为 Agentic 多轮负载设计的——KV Cache 从显存问题变成调度问题的又一例证。
