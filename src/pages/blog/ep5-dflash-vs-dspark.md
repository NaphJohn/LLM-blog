---
title: 同台对比：DFlash 与 DSpark 到底差在哪
description: 从草稿范式、顺序建模、验证调度到执行优化，逐维度拆解 DFlash 与 DSpark 的本质区别，以及为什么二者正交可叠加。
pubDate: 2026-07-31
series: 推测解码手记
lang: zh
altLang: en
altHref: /en/blog/ep5-dflash-vs-dspark
layout: ../../layouts/BlogPost.astro
---

## 1. 共性：都是投机解码

DFlash 与 DSpark 都建立在推测解码（Speculative Decoding, SD）范式上：小草稿模型并行出块、目标大模型并行验证、靠拒绝采样保分布（**零质量损失**）。二者都不是“新模型”，而是**解码 / 服务层的加速方案**。

下文逐维度拆开看它们的不同。

## 2. 维度对比

<table class="cmp-table" id="cmpTable">
<thead><tr><th>维度</th><th>DFlash</th><th>DSpark</th></tr></thead>
<tbody>
<tr class="diff"><td>来源</td><td>Z Lab + SGLang + Modal（2026-06-15 博客）</td><td>DeepSeek + 北大（2026-06-27 开源，MIT，deepseek-ai/DeepSpec）</td></tr>
<tr class="diff"><td>草稿范式</td><td>Block Diffusion：一次前向并行预测整块 masked future tokens</td><td>Semi-Autoregressive + Markov head：块级并行出块，马尔可夫头注入块内顺序依赖</td></tr>
<tr class="diff"><td>条件化 / 顺序建模</td><td>Target hidden-state conditioning + KV injection（跨层注入 target 特征，维持高接受率）</td><td>马尔可夫头注入块内顺序依赖</td></tr>
<tr class="diff"><td>验证调度</td><td>固定 block size（默认 16）标准并行验证</td><td>置信度调度器动态调验证长度（高置信多验证、低置信少验证）</td></tr>
<tr class="diff"><td>执行层优化</td><td>Spec V2 引擎 + overlap scheduler：消除 host-device 同步空转（再 +33%）</td><td>侧重算法层，执行依赖宿主引擎（vLLM / SGLang）</td></tr>
<tr class="diff"><td>公开效果</td><td>Qwen3-8B 最高 6×（比 EAGLE-3 快 ~2.5×）；Blackwell 上 15×</td><td>V4 加速 57–85%、吞吐 +400%</td></tr>
<tr class="diff"><td>生态状态</td><td>SGLang 主支持、vLLM PR #16818 in progress，含 Qwen3 系列多档 drafter</td><td>vLLM PR #46995 合入中，同时支持 SGLang / OpenInfer</td></tr>
</tbody>
</table>

<button class="cmp-btn" id="cmpBtn">高亮核心差异</button>

<script>
  (function () {
    const btn = document.getElementById('cmpBtn');
    const t = document.getElementById('cmpTable');
    if (!btn) return;
    btn.addEventListener('click', () => {
      t.classList.toggle('diff-on');
      btn.textContent = t.classList.contains('diff-on') ? '取消高亮' : '高亮核心差异';
    });
  })();
</script>

## 3. 关键区别一句话

- **DFlash** 革的是「草稿的并行方式（自回归 → 块扩散）」+「执行引擎（消除 host-device 同步空转的 overlap scheduler）」。
- **DSpark** 革的是「草稿块内的顺序建模（半自回归 + 马尔可夫头）」+「验证长度的动态调度（置信度调度器）」。

## 4. 为什么正交可叠加

DFlash 改的是“草稿怎么生成”和“怎么执行”，DSpark 改的是“块内顺序怎么建”和“验证多少 token”。它们作用在**不同层面**，因此正交：OpenInfer 能在 Qwen3-4B 上**同时挂两套路径**——一端是 DFlash 的 block-diffusion drafter + KV injection + overlap scheduler，另一端是 DSpark 的 semi-AR drafter + Markov head + confidence scheduler。

## 5. 发布提醒（重要）

对外写博客 / 论文时，建议**点明你们是“独立双后端”还是“drafter 互换 + 共享执行引擎”**。读者很容易误以为“两套 drafter 是并联运行”，其实更可能是共享同一套 verify 流水线、只是 draft 阶段互换。具体以 OpenInfer 代码 / 官方博客为准。

## 6. 下一篇

实测篇（在 OpenInfer Qwen3-4B 上跑通 DFlash 与 DSpark）将给出评测方法与数据——**仅发布已验证的数字**，若某些数据尚待复核则先不发布。
