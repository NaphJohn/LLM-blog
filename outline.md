# LLM-blog · 全站系列地图与写作 SOP

> 仓库：`NaphJohn/LLM-blog`（GitHub Pages 项目页）· 站点：https://naphjohn.github.io/LLM-blog/
> 框架：Astro（`base: '/LLM-blog/'`，默认语言 zh 无前缀，英文在 `/en/` 下）· 双语：每篇中文 + 英文各一版
> 本文件替代 2026-08 初版《推测解码技术连载 · 博客大纲》（那版只覆盖 ep 系列，已过期）。

---

## 1. 目录与文件约定

```text
src/pages/blog/<slug>.md        中文正文   layout: ../../layouts/BlogPost.astro
src/pages/en/blog/<slug>.md     英文正文   layout: ../../../layouts/BlogPost.astro   ← 多一层 ../
src/pages/index.astro          中文首页：手写系列数组 + 各系列 <h2>/<p class="series-tag">/<ul class="post-list">
src/pages/en/index.astro       英文首页：同上（数组名 sysn / agn / rln 等，避开 JS 全局名）
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

## 2. 系列地图（12 个系列，截至 2026-09-11）

| 前缀 | 系列名 | EN | 篇数 | slug 区间 | 定位 |
|---|---|---|---|---|---|
| `ep` | 推测解码手记 | Speculative Decoding Notes | 8 | ep1–ep8 | 本博客起点：draft/verify → DFlash / DSpark / EAGLE-3 / MTP → DFlash 2 |
| `mm` | 多模态解码手记 | Multimodal Decoding Notes | 5 | mm1–mm5 | 对齐范式 → VLM → 生成模型 → VLA/世界模型 → 高效系统 |
| `fw` | vLLM 与 SGLang 框架解码手记 | vLLM & SGLang Serving Notes | 9 | fw1–fw9 | 推理引擎原理、前沿对抗、版本演进、选型 |
| `vla` | VLA 解码手记 | VLA Notes | 9 | vla1–vla9 | 「动手」能力：动作生成 → π 系列 → 国内玩家 → 世界模型 → RT-1 → V-JEPA 2 → Dreamer V3 → **Octo** |
| `fa` | 前沿架构解码手记 | Frontier Architecture Decoding Notes | 8 | fa1–fa8 | Kimi K3 / MiniMax M3 / DeepSeek V4 / Qwen3.8 双 checkpoint / 三版本同框 |
| `op` | 算子讲解手记 | Operator Notes | 2 | op1–op2 | MLA 算子逐行伪代码、模型量化部署谱系 |
| `pp` | 论文科普手记 | Paper Primer Notes | 2 | pp1–pp2 | Transformer 精读、Diffusion Policy 精读 |
| `tr` | 社区跟踪手记 | Community Tracker Notes | 3 | tr1–tr3 | vLLM/SGLang 上游 commit 的时间切片 |
| `sys` | 推理系统基础设施手记 | Inference Systems Infrastructure Notes | 10 | sys1–sys10 | NUMA/PCIe/NIC → Ring Attention → 注意力改造 → 蒸馏 → KV Cache 全景与工程实现 |
| `rl` | 强化学习训练手记 | RL Training Notes | 1 | rl1– | **训练侧**：分布式 RL 训练 / PPO·GRPO / RLHF 基础设施 |
| `ag` | 智能体手记 | Agent Notes | 1 | ag1– | Agent 能力栈、上下文工程、Agentic 负载下的推理重构 |
| `aihot` | 每日AI热点 | Daily AI Hotspot Notes | 18+ | aihot-YYYYMMDD | 自动聚合（`import.meta.glob`），无需在首页手写注册 |

**首页系列顺序（当前）**：`ep → mm → fw → sys → ag → vla → fa → op → pp → tr → rl → aihot`

---

## 3. 推荐学习路径（从零读起）

```text
① 想懂「推理为什么慢」
   pp1 Transformer → fw1 为什么需要推理引擎 → sys9/sys10 KV Cache 全景与工程 → fw2/fw3 vLLM / SGLang

② 想懂「改模型能省什么」
   sys3 MSA/CSA/HCA 三种注意力改造 → fa1 前沿总览 → fa4 DeepSeek V4 → fa7/fa8 Qwen3.8

③ 想懂「机器人怎么动」
   mm2 ViT→CLIP→LLaVA → vla1 什么是 VLA → vla2 动作生成发动机（Diffusion Policy / Flow Matching）
   → pp2 Diffusion Policy 精读 → vla3 π 系列 → vla9 Octo（通用策略 + 可微调）

④ 想懂「模型怎么被练出来」
   sys2 Ring Attention 长序列训练 → sys4 知识蒸馏 → rl1 分布式 RL 训练（PPO / GRPO / RLHF）
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
6. **构建**：
   ```bash
   cd spec-decode-blog && export CODEBUDDY_SAFE_DELETE_ENABLED=0 && npm run build
   ```
   （不设该变量会在清理阶段被 `checkBulkDeleteGuard` 拦下。）构建后确认新 slug 的 `dist/blog/...` 与 `dist/en/blog/...` 均已生成。
7. **提交推送**：
   ```bash
   git add -A && git commit -m "<前缀>: <标题> 中英双版 + 首页注册"
   GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=20" git push origin main
   git ls-remote origin main   # 比对 HEAD 确认推送成功
   ```

---

## 5. 待补清单（下一批候选）

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

**系统（sys11+）**
- PD 分离在生产环境的实例配比实测

---

## 6. 已知的写作口径

- 每篇结尾放「系列导航」，用绝对路径链到相关篇目（中文版链中文，英文版链英文）。
- 图表用**手写 SVG**（`<div class="fig">` + `<svg viewBox="0 0 680 H">` + `<p class="cap">`），不使用外部图片。
- 公式放代码块（`text` / `python`），避免在 markdown 里写裸露的 `$` 数学。
- 代码块给"能跑的骨架"，不给伪代码占位符。
- 投资相关内容只做产业映射，篇末必带免责声明。
