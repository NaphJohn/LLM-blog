# LLM-blog · 全站系列地图与写作 SOP

> 仓库：`NaphJohn/LLM-blog`（GitHub Pages 项目页）· 站点：https://naphjohn.github.io/LLM-blog/
> 框架：Astro（`base: '/LLM-blog/'`，默认语言 zh 无前缀，英文在 `/en/` 下）· 双语：每篇中文 + 英文各一版
> 本文件替代 2026-08 初版《推测解码技术连载 · 博客大纲》（那版只覆盖 ep 系列，已过期）。

---

## 1. 目录与文件约定

```text
src/pages/blog/<slug>.md        中文正文   layout: ../../layouts/BlogPost.astro
src/pages/en/blog/<slug>.md     英文正文   layout: ../../../layouts/BlogPost.astro   ← 多一层 ../
src/pages/index.astro          中文首页：分类概览 cats 数组 + 手写系列数组 + 各系列 <h2>/<p class="series-tag">/<ul class="post-list">
src/pages/en/index.astro       英文首页：同上（数组名 sysn / agn / rln 等，避开 JS 全局名）
src/pages/panorama.astro       中文端到端推理全景页（7 阶段 × 6 层 × 4 条主线）
src/pages/en/panorama.astro    英文全景页
src/components/SiteHeader.astro 顶部导航：含「全景 / Panorama」全站入口
dist/                          构建产物，不提交
```

**每篇文章 frontmatter 模板**

```yaml
---
title: 系列名（N）：标题
description: 一句话摘要（会进 meta description，写满一行）
pubDate: YYYY-MM-DD
series: 系列名
lang: zh          # en 版写 en
altLang: en       # en 版写 zh
altHref: /en/blog/<slug>
layout: ../../layouts/BlogPost.astro
---
```

**英文 frontmatter 三条规定（踩过坑）**
1. `title` / `description` 含 `": "` 必须整体用**双引号**包裹；
2. 双引号内**禁止出现单引号**（如 `RT-1's` 会破坏 YAML）→ 改写为 `the RT-1 route` 或去掉撇号；
3. 英文版 layout 路径比中文版**多一层 `../`**。

---

## 2. 系列地图（13 个系列，截至 2026-10-27）

站点已按 **5 大类** 重组（首页有「按分类概览」卡片 + 正文分类分隔标题，另有端到端推理全景页做交叉导航）。下表按分类分组。

| 分类 | 前缀 | 系列名 | EN | 篇数 | slug 区间 | 锚点 | 定位 |
|---|---|---|---|---|---|---|---|
| **一 · 模型与解码算法** | `fa` | 前沿架构解码手记 | Frontier Architecture Decoding Notes | 10 | fa1–fa10 | `#s-fa` | Kimi K3 / MiniMax M3 / DeepSeek V4 / Mamba 混合状态 + PD 分离 / Gated DeltaNet / Qwen3.8 双 checkpoint / 三版本同框 / **残差连接演进** / **MiniMax M3.1 Flash** |
| | `ep` | 推测解码手记 | Speculative Decoding Notes | 8 | ep1–ep8 | `#s-spec` | 本博客起点：draft/verify → DFlash / DSpark / EAGLE-3 / MTP → DFlash 2 |
| | `op` | 算子讲解手记 | Operator Notes | 3 | op1–op3 | `#s-op` | MLA 算子逐行伪代码、**模型量化谱系总表（qt 系列导读入口）**、**EAGLE QBSA** |
| | `qt` | 量化手记 | Quantization Notes | 7 | qt1–qt7 | `#s-qt` | **新增（2026-10-27）**：量化基本语言 → RTN/GPTQ（OBS 推导）→ SmoothQuant/AWQ → QuaRot/SpinQuant/DuQuant（W4A4）→ KV Cache 量化与 FP4 → **QAT（STE/PACT/LSQ）** → **量化的三个边界（投机解码/MLA/端侧）** |
| **二 · 推理系统与工程** | `sys` | 推理系统基础设施手记 | Inference Systems Infrastructure Notes | 12 | sys1–sys12 | `#s-sys` | NUMA/PCIe/NIC → Ring Attention → 注意力改造 → 蒸馏 → KV Cache 全景与工程实现 → DCP / PD 分离 → HiCache L2 昆仑芯调优 → **Mooncake** |
| | `fw` | vLLM 与 SGLang 框架解码手记 | vLLM & SGLang Serving Notes | 10 | fw1–fw10 | `#s-fw` | 推理引擎原理、前沿对抗、版本演进、选型、Wan2.2 / vLLM-Omni、**DeepInfer 生产级推理栈** |
| **三 · 多模态与具身** | `vla` | VLA 解码手记 | VLA Notes | 9 | vla1–vla9 | `#s-vla` | 「动手」能力：动作生成 → π 系列 → 国内玩家 → 世界模型 → RT-1 → V-JEPA 2 → Dreamer V3 → Octo |
| | `mm` | 多模态解码手记 | Multimodal Decoding Notes | 5 | mm1–mm5 | `#s-mm` | 对齐范式 → VLM → 生成模型 → VLA/世界模型 → 高效系统 |
| | `pp` | 论文科普手记 | Paper Primer Notes | 2 | pp1–pp2 | `#s-pp` | Transformer 精读、Diffusion Policy 精读 |
| **四 · 训练与智能体** | `rl` | 强化学习训练手记 | RL Training Notes | 1 | rl1– | `#s-rl` | 训练侧：分布式 RL 训练 / PPO·GRPO / RLHF 基础设施 |
| | `ag` | 智能体手记 | Agent Notes | 1 | ag1– | `#s-ag` | Agent 能力栈、上下文工程、Agentic 负载下的推理重构 |
| **五 · 时效跟踪** | `aihot` | 每日AI热点 | Daily AI Hotspot Notes | 43 | aihot-YYYYMMDD | `#s-aihot` | 自动聚合（`import.meta.glob`），无需在首页手写注册 |
| | `tr` | 社区跟踪手记 | Community Tracker Notes | 3 | tr1–tr3 | `#s-tr` | vLLM/SGLang 上游 commit 的时间切片（**停在 tr3 / 08-11**；此后跟踪并入 `aihot`，注意避免同期内容双挂） |

**合计 114 篇**，其中 71 篇为原创双语长文（114 − 43 篇 aihot）；分类篇数：模型与解码算法 28 / 推理系统与工程 22 / 多模态与具身 16 / 训练与智能体 2 / 时效跟踪 46。

**首页系列顺序（当前，按分类分组）**：`fa → ep → op → qt` ｜ `sys → fw` ｜ `vla → mm → pp` ｜ `rl → ag` ｜ `aihot → tr`

**`posts.js` 的 `SERIES_ORDER` 必须同步**：`SERIES_ORDER.zh` 里 `'量化手记'` 放在 `'算子讲解手记'` 之后，`SERIES_ORDER.en` 里 `'Quantization Notes'` 放在 `'Operator Notes'` / `'Operator Explainers'` 之后。这是「上一篇/下一篇」能跨进新系列的唯一开关，漏掉只会表现为「链到 op3 就断了」，不会报错。

**分类分隔标题锚点**：`#c-model` / `#c-sys` / `#c-multi` / `#c-train` / `#c-daily`（CSS 已设 `:root { scroll-padding-top: 76px; }` 抵消 sticky header 遮挡）

**端到端推理全景页**：`src/pages/panorama.astro`（中文）与 `src/pages/en/panorama.astro`（英文）。横向 7 阶段（入口与协议 / 调度与批处理 / 前缀复用与缓存 / Prefill 与并行切分 / 解码加速 / 采样与约束 / 返回与流式）× 纵向 6 层（L5 应用与协议 → L0 并行与算子），另给 4 条阅读主线（搞懂原理 / 调优线上服务 / 选型判断 / 国产硬件落地）。已从 `SiteHeader.astro` 给全站每页加了「全景 / Panorama」入口。

---

## 3. 推荐学习路径（从零读起）

> 想先看全局，直接去 **端到端推理全景页**（`/panorama/`、`/en/panorama/`）：横轴是请求生命周期，纵轴是技术分层，一张图定位站内对应文章。下面是按问题出发的四条路径。

```text
① 想懂「推理为什么慢」
   pp1 Transformer → fw1 为什么需要推理引擎 → sys9/sys10 KV Cache 全景与工程 → fw2/fw3 vLLM / SGLang
   → fw10 DeepInfer（看生产级栈怎么把上面这些拼起来）

② 想懂「改模型能省什么」
   sys3 MSA/CSA/HCA 三种注意力改造 → fa1 前沿总览 → fa4 DeepSeek V4 → fa7/fa8 Qwen3.8 → fa9 残差连接演进

③ 想懂「机器人怎么动」
   mm2 ViT→CLIP→LLaVA → vla1 什么是 VLA → vla2 动作生成发动机（Diffusion Policy / Flow Matching）
   → pp2 Diffusion Policy 精读 → vla3 π 系列 → vla9 Octo（通用策略 + 可微调）

④ 想懂「模型怎么被练出来」
   sys2 Ring Attention 长序列训练 → sys4 知识蒸馏 → rl1 分布式 RL 训练（PPO / GRPO / RLHF）

⑤ 想懂「线上服务怎么调优」
   fw10 DeepInfer 四层架构 → sys8 PD 分离实例配比 → sys12 Mooncake → fw7 CUDA Graph → ep 系列选解码加速
   （全景页「调优线上服务」主线就是这条顺序）

⑥ 想懂「量化到底怎么做」（2026-10 新增，qt 系列）
   qt1 量化基本语言（一个反量化公式 + 粒度/对称性/校准三旋钮 + 四路线地图）
   → qt2 RTN vs GPTQ（W4A16 主线：OBS 推导 → 逐列补偿 → Cholesky / 惰性批量更新）
   → qt3 SmoothQuant vs AWQ（W8A8 与 W4A16 两条路：搬走 outlier vs 放大保护显著通道）
   → qt4 QuaRot / SpinQuant / DuQuant（W4A4 唯一出路：R1–R4 四个旋转位置 + 三代方案）
   → qt5 KV Cache 量化与 FP8/FP4（正交维度：KIVI、residual window、NVFP4 微缩放、2026 选型表）
   → qt6 QAT 与量化感知训练（唯一「权重会动」的一篇：STE 三个故障 → PACT / LSQ 两种修法 → 七步流程）
   → qt7 量化的三个边界（投机解码下草稿树要变小、MLA 下 KIVI 失效、端侧格式决定 NPU 通路）
   横向索引看 op2 的「量化方法谱系总表」（按「改什么」分四条路线 + 上限位宽 + 可叠加性）
```

---

## 4. 新增一篇文章的 SOP（照做即可，别漏步）

1. **写中文版** `src/pages/blog/<slug>.md`（按第 1 节 frontmatter 模板）。
2. **写英文版** `src/pages/en/blog/<slug>.md`（title/description 双引号规则、layout 多一层 `../`）。
3. **首页注册（两处都要）**：
   - `src/pages/index.astro` → 找到对应 `const <系列> = [...]`，追加一条 `{ href: base + 'blog/<slug>', title: '（N）标题', date: 'YYYY-MM-DD' }`；
   - `src/pages/en/index.astro` → 同一系列数组（注意 `sysn` / `agn` / `rln` 这类改名）追加英文条目。
4. **同步系列描述**：更新该系列 `<p class="series-tag">` 的**篇数**（`共 N 篇` / `Series · N episodes`）与主线段落，把新篇补进"循序渐进"链条里。
5. **若新增系列**：两个 index 都要加 `const` 数组 + `<h2>`/`series-tag`/`<ul>` 区块，并更新 `<p class="lead">` 的"N 大系列"计数与列举。
6. **同步分类概览**（2026-10 起，新增一篇的必做步骤）：
   - 两个 index 的 `const cats = [...]` 里对应系列的 `count` 是由数组 `length` 自动算的，一般无需改；但若是**新增系列**，要把它加进对应的分类 `series` 数组（含 `anchor`），并确认 `totalAll` / `totalBi` 自动更新；
   - 正文里该系列 `<h2>` 的 `id="s-xxx"` 必须与 `cats` 里的 `anchor` 一致，否则卡片点过去不跳；
   - 若新篇落进全景页的某个阶段，顺手在 `panorama.astro` 对应 `stages[].posts` 里补一条。
7. **构建**：
   ```bash
   cd spec-decode-blog && export PATH="/Users/hangkai/.workbuddy/binaries/node/versions/22.22.2-3/bin:$PATH" && export CODEBUDDY_SAFE_DELETE_ENABLED=0 && npm run build
   ```
   （不设该变量会在清理阶段被 `checkBulkDeleteGuard` 拦下。）构建后确认新 slug 的 `dist/blog/...` 与 `dist/en/blog/...` 均已生成。
8. **提交推送**：
   ```bash
   git add -A && git commit -m "<前缀>: <标题> 中英双版 + 首页注册"
   GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=20" git push origin main
   git ls-remote origin main   # 比对 HEAD 确认推送成功
   ```

---

## 5. 待补清单（下一批候选）

**量化手记（qt8+，2026-10-27 起）**
- 混合精度量化：哪些层该保 FP8/FP16 的自动搜索（敏感度分析 vs 搜索）
- 量化 + 稀疏叠加：昇腾 EagleQBSA 之后还有哪些组合（与 op3 联动）
- 连续 batch 下 per-token 激活 scale 的重算开销实测
- 量化模型的蒸馏补偿：用 FP16 教师修 4 bit 学生的 logits 差距

**VLA 系列（vla10+）**
- π0.5 / π0.7 的层级推理（高层语言子目标 + 低层动作）
- GRPO / RLAIF 做 VLA 偏好对齐的完整案例（与 rl 系列联动）

**强化学习训练手记（rl2+）**
- GRPO 深入：去掉 Critic 之后优势估计的方差控制
- Rollout 引擎对比：vLLM vs SGLang 作为 RL rollout 的取舍
- 异步 RL（AReaL 路线）与 replay buffer 设计
- Agentic RL：把工具调用纳入 rollout 的工程问题

**前沿架构（fa9+）**
- 长上下文推理的稀疏注意力量化对比

**系统（sys12+）**
- PD 分离在生产环境的实例配比实测
- DeepInfer 未开源三块（完整 KV 管理 / 3FS、PD 分离、副本负载均衡）的可推断实现

**框架（fw11+）**
- DeepInfer 在国产硬件上的落地实测（注意 `DEEPINFER_BATCH_INVARIANT=1` 仅 NVIDIA 支持）
- 与 vLLM v1 / SGLang 的同场景吞吐对照

---

## 6. 已知的写作口径

- 每篇结尾放「系列导航」，用绝对路径链到相关篇目（中文版链中文，英文版链英文）。
- 图表用**手写 SVG**（`<div class="fig">` + `<svg viewBox="0 0 680 H">` + `<p class="cap">`），不使用外部图片。
- 公式放代码块（`text` / `python`），避免在 markdown 里写裸露的 `$` 数学。
- 代码块给"能跑的骨架"，不给伪代码占位符。
- 投资相关内容只做产业映射，篇末必带免责声明。
