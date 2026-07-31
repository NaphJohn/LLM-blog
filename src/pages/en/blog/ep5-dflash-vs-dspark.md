---
title: "Head-to-Head: Where DFlash and DSpark Actually Differ"
description: A dimension-by-dimension breakdown of DFlash vs DSpark — draft paradigm, ordering, verification scheduling, execution — and why the two are orthogonal and stackable.
pubDate: 2026-07-31
series: Speculative Decoding Notes
lang: en
altLang: zh
altHref: /blog/ep5-dflash-vs-dspark
layout: ../../../layouts/BlogPost.astro
---

## 1. Common ground: both are speculative decoding

DFlash and DSpark both build on the speculative decoding (SD) paradigm: a small draft model emits blocks in parallel, the target large model verifies in parallel, and rejection sampling preserves the distribution (**zero quality loss**). Neither is a "new model" — both are **decoding / serving-layer acceleration** schemes.

Below we pull them apart dimension by dimension.

## 2. Dimension-by-dimension

<table class="cmp-table" id="cmpTable">
<thead><tr><th>Dimension</th><th>DFlash</th><th>DSpark</th></tr></thead>
<tbody>
<tr class="diff"><td>Origin</td><td>Z Lab + SGLang + Modal (2026-06-15 blog)</td><td>DeepSeek + Peking University (2026-06-27, open source, MIT, deepseek-ai/DeepSpec)</td></tr>
<tr class="diff"><td>Draft paradigm</td><td>Block Diffusion: one forward pass predicts a whole block of masked future tokens in parallel</td><td>Semi-Autoregressive + Markov head: block-level parallel emission, Markov head injects intra-block sequential dependency</td></tr>
<tr class="diff"><td>Conditioning / ordering</td><td>Target hidden-state conditioning + KV injection (injects target features across layers, keeps acceptance high)</td><td>Markov head injects intra-block sequential dependency</td></tr>
<tr class="diff"><td>Verify scheduling</td><td>Fixed block size (default 16), standard parallel verification</td><td>Confidence scheduler dynamically adjusts verify length (verify more when confident, less when not)</td></tr>
<tr class="diff"><td>Execution optimization</td><td>Spec V2 engine + overlap scheduler: eliminates host-device sync idle (+33% more)</td><td>Algorithm-layer focus; execution relies on the host engine (vLLM / SGLang)</td></tr>
<tr class="diff"><td>Published results</td><td>Qwen3-8B up to 6× (≈2.5× faster than EAGLE-3); 15× on Blackwell</td><td>V4 speedup 57–85%, throughput +400%</td></tr>
<tr class="diff"><td>Ecosystem</td><td>SGLang primary; vLLM PR #16818 in progress; multiple Qwen3 drafter tiers</td><td>vLLM PR #46995 merging; also supports SGLang / OpenInfer</td></tr>
</tbody>
</table>

<button class="cmp-btn" id="cmpBtn">Highlight key differences</button>

<script>
  (function () {
    const btn = document.getElementById('cmpBtn');
    const t = document.getElementById('cmpTable');
    if (!btn) return;
    btn.addEventListener('click', () => {
      t.classList.toggle('diff-on');
      btn.textContent = t.classList.contains('diff-on') ? 'Remove highlight' : 'Highlight key differences';
    });
  })();
</script>

## 3. The one-line difference

- **DFlash** reinvents *how the draft is generated* (autoregressive → block diffusion) + *the execution engine* (overlap scheduler that removes host-device sync idle).
- **DSpark** reinvents *intra-block ordering* (semi-autoregressive + Markov head) + *dynamic verification scheduling* (confidence scheduler).

## 4. Why they are orthogonal and stackable

DFlash changes "how the draft is produced" and "how it is executed"; DSpark changes "how intra-block order is built" and "how many tokens to verify". They operate at **different layers**, hence orthogonal: OpenInfer can mount **both paths on Qwen3-4B simultaneously** — one side is DFlash's block-diffusion drafter + KV injection + overlap scheduler, the other is DSpark's semi-AR drafter + Markov head + confidence scheduler.

## 5. Publishing note (important)

When writing the blog post / paper, we recommend **stating clearly whether you use "independent dual backends" or "interchangeable drafters + shared execution engine"**. Readers easily assume "the two drafters run in parallel", whereas in practice it is more likely a shared verify pipeline with only the draft stage swapped. Definitive wording should follow the OpenInfer code / official blog.

## 6. What's next

The benchmark post (running DFlash & DSpark on OpenInfer Qwen3-4B) will give the methodology and numbers — **only verified numbers will be published**; if some data is still under review, it will be held back.
