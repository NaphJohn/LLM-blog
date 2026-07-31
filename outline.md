# 推测解码技术连载 · 博客大纲（Spec-Decode Blog — Outline）

> 状态：**大纲评审稿**（未建仓库）。待用户确认后初始化 `hangkaiwang.github.io`（Astro + 双语）。
> 决策记录：仓库=`hangkaiwang.github.io` · 框架=Astro · 语言=双语并行（每主题 1 篇中文 + 1 篇英文，内容一致）· 节奏=先大纲、后建站。

---

## 0. 项目元信息

| 项 | 值 |
|---|---|
| 仓库 | `hangkaiwang.github.io`（GitHub Pages，免费） |
| 框架 | Astro + Content Collections + i18n 路由 |
| 语言策略 | 双语并行：**每个主题 = 1 篇中文 + 1 篇英文，内容一致** |
| 默认路由 | 建议 `/` = 英文，`/zh/` = 中文（国际读者为主；可调） |
| 部署 | GitHub Actions → Pages，`git push` 即上线（无需服务器/CDN 费用） |
| 可复用资产 | `dflash_explainer.html`（CPU 领先 GPU 时序 SVG）、DFlash vs DSpark 对比文、DSpark 邮件素材 |

---

## 1. 站点架构（简版）

- `src/content/blog/zh/<slug>.md` 与 `src/content/blog/en/<slug>.md` 成对存放，同一 `slug` 中英对照。
- 列表页 `/blog` 与 `/zh/blog` 各自按语言聚合；每篇底部提供「中文 / English」切换链接（指向同 slug 的另一语言版本）。
- 交互组件用 Astro `.astro` / MDX 封装：
  - 「草稿→验证」循环动画（Ep1）
  - CPU 领先 GPU 时序对比 SVG（复用 `dflash_explainer.html`，Ep3）
  - DFlash vs DSpark 可切换对比表 / 并排架构图（Ep5）
- 部署：仓库根 `.github/workflows/deploy.yml` 跑 `astro build` → 推 `gh-pages` → Pages 启用。

---

## 2. 系列总览（Series Map）

循序渐进 7 篇（第 7 篇可选），每篇中英各一版：

| # | 中文标题 | English Title | 主题 |
|---|---|---|---|
| 1 | 推测解码：让大模型"一次生成多个 token"的无损加速 | Speculative Decoding: Lossless Multi-Token Generation | 基础概念 |
| 2 | 为什么传统草稿模型只能加速 2–3 倍？ | Why Autoregressive Drafters Cap at 2–3× | 瓶颈分析 |
| 3 | DFlash：用块扩散 + 特征注入革掉草稿的串行 | DFlash: Block-Diffusion Drafting Meets Target Conditioning | DFlash 深度解析 |
| 4 | DSpark：半自回归 + 置信度调度，让验证长度自适应 | DSpark: Semi-Autoregressive Drafting with Confidence-Aware Scheduling | DSpark 深度解析 |
| 5 | 同台对比：DFlash 与 DSpark 到底差在哪 | Head-to-Head: Where DFlash and DSpark Actually Differ | 对比篇 |
| 6 | 在 OpenInfer Qwen3-4B 上跑通 DFlash 与 DSpark | Benchmarking DFlash & DSpark on OpenInfer Qwen3-4B | 实测篇 |
| 7（可选） | 动手训练一个 DFlash drafter | Training Your Own DFlash Drafter | 动手实践 |

---

## 3. 逐篇大纲

### Episode 1 — 推测解码是什么
**中文**：推测解码：让大模型"一次生成多个 token"的无损加速
**English**：Speculative Decoding: Lossless Multi-Token Generation
- 1.1 动机：自回归生成的串行瓶颈（每 token 一次大模型前向，贵且慢）
- 1.2 核心思想：Draft + Verify 两阶段（小模型并行起草 → 大模型并行验证）
- 1.3 为什么无损？拒绝采样（rejection sampling）保分布
- 1.4 加速从哪来？接受长度 α 的直觉与加速比公式
- 1.5 最小直觉例子：草稿 4 个、接受 3 个
- 1.6 小结 + 下篇预告
- 🔧 交互：草稿→验证循环动画小部件

### Episode 2 — 瓶颈：自回归草稿模型的天花板
**中文**：为什么传统草稿模型只能加速 2–3 倍？
**English**：Why Autoregressive Drafters Cap at 2–3×
- 2.1 回顾：加速 ≈ α / (1 + β·k)
- 2.2 EAGLE-3 为何卡在 2–3×
- 2.3 草稿成本随投机 token 数线性增长（drafting cost ∝ k）
- 2.4 内存墙与串行化（draft 与 verify 之间的 gap）
- 2.5 破局思路：并行 / 非自回归草稿 → 引出 DFlash / DSpark
- 2.6 预告

### Episode 3 — DFlash 深度解析
**中文**：DFlash：用块扩散 + 特征注入革掉草稿的串行
**English**：DFlash: Block-Diffusion Drafting Meets Target Conditioning
- 3.1 来源与定位（Z Lab + SGLang + Modal，2026-06-15 博客）
- 3.2 Block Diffusion Drafter：一次前向并行草拟一整块 masked future tokens
- 3.3 Target Hidden-State Conditioning + KV Injection（跨层注入 target 特征，维持高接受率）
- 3.4 Spec V2 引擎 + Overlap Scheduler：消除 host-device 同步空转（+33%）
- 3.5 实测数据：Qwen3-8B 最高 6×（比 EAGLE-3 快 ~2.5×）、Blackwell 15×
- 3.6 生态：SGLang 主支持、vLLM PR #16818、Qwen3 系列多档 drafter
- 🔧 交互：CPU 领先 GPU 时序对比 SVG（复用 `dflash_explainer.html`）

### Episode 4 — DSpark 深度解析
**中文**：DSpark：半自回归 + 置信度调度，让验证长度自适应
**English**：DSpark: Semi-Autoregressive Drafting with Confidence-Aware Scheduling
- 4.1 来源与定位（DeepSeek + 北大，2026-06-27 开源，MIT，deepseek-ai/DeepSpec）
- 4.2 半自回归 + 马尔可夫头：块内顺序依赖怎么建
- 4.3 置信度调度器：高置信多验证 / 低置信少验证
- 4.4 零质量损失 + 大模型并行验证
- 4.5 实测数据：V4 加速 57–85%、吞吐 +400%
- 4.6 生态：vLLM PR #46995（复用稀疏 MLA 非因果索引、DSparkSpeculator 继承 DFlash、Triton 非因果 SWA 内核），同时支持 SGLang / OpenInfer

### Episode 5 — 对比篇：DFlash vs DSpark
**中文**：同台对比：DFlash 与 DSpark 到底差在哪
**English**：Head-to-Head: Where DFlash and DSpark Actually Differ
- 5.1 共性：都是投机解码、都零质量损失
- 5.2 维度对比表（草稿范式 / 顺序建模 / 验证调度 / 执行优化 / 效果 / 生态）
- 5.3 关键区别一句话：DFlash 革「草稿并行方式 + 执行引擎」；DSpark 革「块内顺序建模 + 验证调度」
- 5.4 为什么正交可叠加（OpenInfer 能同时挂两套路径的原因）
- 5.5 发布提醒：点明是「独立双后端」还是「drafter 互换 + 共享执行引擎」
- 🔧 交互：可切换对比表 / 并排架构图

### Episode 6 — 实测篇：OpenInfer Qwen3-4B 双后端
**中文**：在 OpenInfer Qwen3-4B 上跑通 DFlash 与 DSpark
**English**：Benchmarking DFlash & DSpark on OpenInfer Qwen3-4B
- 6.1 测试环境（硬件 / 基线 = 原生自回归）
- 6.2 评测方法（接受长度 / tokens·s⁻¹ / 吞吐 / 质量一致性）
- 6.3 DFlash 路径结果与调参
- 6.4 DSpark 路径结果与调参
- 6.5 组合策略（若叠加）
- 6.6 结论 + 复现命令

### Episode 7（可选）— 动手训练一个 DFlash drafter
- 7.1 数据准备
- 7.2 drafter 结构
- 7.3 训练目标
- 7.4 接入 OpenInfer / SGLang

---

## 4. 发布节奏建议

- **起步**：先发 Ep1（基础，必读门槛低）+ Ep5（对比篇，资产现成、最能吸睛），攒初始流量。
- **节奏**：建议每周 1 篇（中英同步发），或双周 1 篇，保持连续更新对 SEO 友好。
- **工作流**：你每写好一篇草稿（或让我据大纲扩写），我负责 build + deploy，你只需 `git push` 审阅后的版本。

## 5. 待你拍板的小项

1. **默认语言**：建议英文默认（`/`） + `/zh` 中文；若你更想服务中文读者可反过来。
2. **站点标题 / 品牌名**：例如 "SpecDecode Notes" 或 "推测解码手记"？影响域名子路径与导航。
3. **Ep6 数据可公开性**：OpenInfer Qwen3-4B 的实测数字是否可对外发布？决定 Ep6 现在能否动笔。
4. **开建触发**：大纲确认后，是否立即初始化仓库 + 先把 Ep1/Ep5 推上去（用现成资产）？

---

*生成时间：2026-07-31 · 依据前序对话中已核实的 DFlash（Z Lab+SGLang+Modal 2026-06-15）、DSpark（DeepSeek+北大 2026-06-27）、OpenInfer Qwen3-4B 双后端支持等素材整理。*
