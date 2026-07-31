---
title: 推测解码：让大模型“一次生成多个 token”的无损加速
description: 一文讲清推测解码的两阶段原理、为何无损、加速从何而来，以及接受长度 α 的直觉。
pubDate: 2026-07-31
series: 推测解码手记
lang: zh
altLang: en
altHref: /en/blog/ep1-speculative-decoding
layout: ../../layouts/BlogPost.astro
---

## 1. 动机：自回归生成的串行瓶颈

今天的大语言模型（LLM）基本都是**自回归（autoregressive）**生成：每产出一个 token，就要把包括这个新 token 在内的全部上下文喂回模型，跑一次完整前向。换句话说，生成是**严格串行**的——第 t+1 个 token 必须等第 t 个算完。

这意味着两件事：

- **延迟高**：每个 token 都要等一次大模型前向（大模型很贵）。
- **算力浪费**：一次前向本身能并行处理很多 token，但自回归把生成“锁”成了一步一步的串行循环。

有没有可能让一次大模型前向**顺带多产出几个 token**？这正是推测解码（Speculative Decoding, SD）要解决的问题。

## 2. 核心思想：Draft + Verify 两阶段

SD 引入一个**草稿模型（draft / small model）**：它比目标大模型小得多、便宜得多，但“大致同任务”。生成不再是“大模型一步步走”，而是两轮协作：

1. **起草（Draft）**：用小模型一次性并行草拟出接下来的一整块候选 token（比如 4 个）。
2. **验证（Verify）**：把这块候选**整体**交给大模型做**一次并行**前向，大模型对每个候选独立给出“接受 / 拒绝”的判断（基于拒绝采样）。

被接受的候选直接采用；第一个被拒绝的位置，大模型用自己的预测覆盖，并从那里继续。关键在于：**无论草案多长，大模型只跑了一次前向**——所以如果草案大部分被接受，我们就“用一次前向换来了多个 token”。

> 直觉：让便宜的小模型先“猜”一串，贵的大模型只“看一眼”就盖章。猜得越准，加速越大。

<div class="sd-anim" id="sdAnim">
  <div class="sd-row"><span class="sd-role">草稿模型（小）</span><div class="sd-tokens" id="draftTokens"></div></div>
  <div class="sd-row"><span class="sd-role">目标模型（大）</span><div class="sd-tokens" id="targetTokens"></div></div>
  <div class="sd-status" id="sdStatus">点击「开始」观察 Draft → Verify 循环</div>
  <button class="sd-btn" id="sdBtn">开始 / 暂停</button>
</div>

<script>
  (function () {
    const draft = document.getElementById('draftTokens');
    const target = document.getElementById('targetTokens');
    const status = document.getElementById('sdStatus');
    const btn = document.getElementById('sdBtn');
    if (!draft) return;
    const N = 4, accept = 3;
    const tokens = [];
    for (let i = 0; i < N; i++) {
      const d = document.createElement('span'); d.className = 'sd-tok'; d.textContent = '?'; draft.appendChild(d); tokens.push(d);
      const t = document.createElement('span'); t.className = 'sd-tok'; t.textContent = '·'; target.appendChild(t);
    }
    let running = false, timer = null;
    function frame() {
      status.textContent = '① 草稿模型并行起草 ' + N + ' 个候选 token…';
      tokens.forEach((t, i) => { t.className = 'sd-tok sd-drafting'; t.textContent = 't' + (i + 1); });
      timer = setTimeout(() => {
        status.textContent = '② 目标模型一次并行验证…';
        tokens.forEach((t, i) => { t.className = i < accept ? 'sd-tok sd-ok' : 'sd-tok sd-rej'; });
        tokens[accept].textContent = '✗→大树';
        timer = setTimeout(() => {
          status.textContent = '结果：接受 ' + accept + ' 个，第 ' + (accept + 1) + ' 位由大模型覆盖 → 1 次前向产出 ' + (accept + 1) + ' 个 token';
          timer = setTimeout(() => { if (running) frame(); }, 1800);
        }, 1000);
      }, 1000);
    }
    btn.addEventListener('click', () => {
      running = !running;
      if (running) { frame(); } else { clearTimeout(timer); status.textContent = '已暂停（点击继续）'; }
    });
  })();
</script>

## 3. 为什么无损？拒绝采样

这是 SD 最迷人的性质：**输出分布与目标大模型逐 token 完全一致，零质量损失**。

秘密在拒绝采样（rejection sampling）。大模型在验证时并不是“全接受”或“全拒绝”，而是按概率**逐 token 重新抽样**：

- 若草稿 token 与大树在该位置的概率分布一致，高概率接受；
- 若不一致，则以大树分布为基准做修正抽样，保证最终采样的边际分布 = 大树分布。

所以 SD 不改变模型“会说出什么”，只改变“说出得多快”。这一点对生产部署极其重要——你可以用 SD 提速，而不必担心回答质量下降。

## 4. 加速从哪来？接受长度 α

设一次大模型前向耗时 ≈ 1 单位，起草 k 个 token 的小模型耗时 ≈ β·k（β≪1）。若平均有 α 个候选被接受（**接受长度 acceptance length**），则每产出 α 个 token 的总成本 ≈ 1 + β·k。

加速比 ≈ **α / (1 + β·k)**。

可见：

- 加速**正比于接受长度 α**（草案越准，α 越大）；
- 起草成本随 k 线性增长，所以 k 不是越大越好，存在一个甜点；
- 传统自回归草稿模型（如 EAGLE-3）的 α 常被卡在 2–3，这就是下一篇要聊的“瓶颈”。

## 5. 最小直觉例子

假设起草 4 个 token，大模型一次验证后接受了前 3 个、在第 4 个位置拒绝并用自己的预测替换：

- 串行基线：产出 4 个 token 需要 **4 次**大模型前向。
- SD：产出 4 个 token 只用了 **1 次**大模型前向（+ 几乎可忽略的小模型成本）。

在这一步里，我们相当于把“4 次”压成了“1 次”——这就是 SD 加速的来源。

## 6. 小结与下篇预告

- SD = 小模型起草 + 大模型并行验证，靠拒绝采样**保分布、零质量损失**。
- 加速来自**接受长度 α**：草案越准，省得越多。
- 但 α 有天花板：传统自回归草稿模型为何卡在 2–3×？下一篇《为什么传统草稿模型只能加速 2–3 倍？》我们拆解瓶颈，并引出 **DFlash、DSpark** 这两条破局路线。
