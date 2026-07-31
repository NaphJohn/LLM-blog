---
title: "Speculative Decoding: Lossless Multi-Token Generation"
description: A clear walkthrough of the two-stage speculative decoding mechanism, why it is lossless, and where the speedup comes from (acceptance length α).
pubDate: 2026-07-31
series: Speculative Decoding Notes
lang: en
altLang: zh
altHref: /blog/ep1-speculative-decoding
layout: ../../../layouts/BlogPost.astro
---

## 1. Motivation: the serial bottleneck of autoregressive generation

Today's large language models (LLMs) are almost all **autoregressive**: to produce each new token, the model feeds the entire context — including that new token — back through a full forward pass. In other words, generation is **strictly serial** — token t+1 cannot start until token t finishes.

This implies two things:

- **High latency**: every token waits for one expensive large-model forward pass.
- **Wasted compute**: a single forward pass can process many tokens in parallel, yet autoregressive decoding locks generation into a step-by-step loop.

Could one large-model forward pass **emit several tokens at once**? That is exactly what speculative decoding (SD) addresses.

## 2. Core idea: Draft + Verify, two stages

SD introduces a **draft (small) model**: much smaller and cheaper than the target large model, but trained for roughly the same task. Generation is no longer "the large model walks step by step"; it is a two-stage collaboration:

1. **Draft**: the small model parallelizes a whole block of candidate tokens at once (say, 4 tokens).
2. **Verify**: the candidate block is handed to the large model for **a single parallel** forward pass; the large model independently decides "accept / reject" for each candidate (via rejection sampling).

Accepted candidates are kept; at the first rejected position, the large model overrides with its own prediction and continues from there. The key point: **no matter how long the draft, the large model ran only one forward pass** — so if most candidates are accepted, we traded one forward pass for multiple tokens.

> Intuition: let the cheap small model "guess" a sequence first; the expensive large model just "glances once" and stamps it. The more accurate the guess, the bigger the speedup.

<div class="sd-anim" id="sdAnim">
  <div class="sd-row"><span class="sd-role">Draft (small)</span><div class="sd-tokens" id="draftTokens"></div></div>
  <div class="sd-row"><span class="sd-role">Target (large)</span><div class="sd-tokens" id="targetTokens"></div></div>
  <div class="sd-status" id="sdStatus">Click "Start" to watch the Draft → Verify loop</div>
  <button class="sd-btn" id="sdBtn">Start / Pause</button>
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
      status.textContent = '① Draft model parallelizes ' + N + ' candidate tokens…';
      tokens.forEach((t, i) => { t.className = 'sd-tok sd-drafting'; t.textContent = 't' + (i + 1); });
      timer = setTimeout(() => {
        status.textContent = '② Target model verifies in one parallel pass…';
        tokens.forEach((t, i) => { t.className = i < accept ? 'sd-tok sd-ok' : 'sd-tok sd-rej'; });
        tokens[accept].textContent = '✗→large';
        timer = setTimeout(() => {
          status.textContent = 'Result: ' + accept + ' accepted, position ' + (accept + 1) + ' overridden by the large model → 1 forward pass yields ' + (accept + 1) + ' tokens';
          timer = setTimeout(() => { if (running) frame(); }, 1800);
        }, 1000);
      }, 1000);
    }
    btn.addEventListener('click', () => {
      running = !running;
      if (running) { frame(); } else { clearTimeout(timer); status.textContent = 'Paused (click to resume)'; }
    });
  })();
</script>

## 3. Why lossless? Rejection sampling

This is SD's most appealing property: **the output distribution matches the target large model token-by-token, with zero quality loss**.

The secret is rejection sampling. During verification the large model does not simply "accept all or reject all"; it re-samples **per token by probability**:

- if a draft token is consistent with the large model's distribution at that position, it is accepted with high probability;
- if not, it is corrected by re-sampling against the large model's distribution, guaranteeing the marginal distribution of the final sample equals the large model's.

So SD does not change *what* the model says, only *how fast* it says it. This is crucial for production: you can speed up with SD without worrying about answer quality dropping.

## 4. Where does the speedup come from? Acceptance length α

Let one large-model forward pass cost ≈ 1 unit, and drafting k tokens cost ≈ β·k (β≪1). If on average α candidates are accepted (**acceptance length α**), the total cost to produce α tokens ≈ 1 + β·k.

Speedup ≈ **α / (1 + β·k)**.

So:

- speedup is **proportional to acceptance length α** (the more accurate the draft, the larger α);
- drafting cost grows linearly with k, so bigger k is not always better — there is a sweet spot;
- classical autoregressive drafters (e.g., EAGLE-3) are often capped at α ≈ 2–3, which is the "bottleneck" of the next post.

## 5. A minimal intuition example

Suppose we draft 4 tokens; the large model accepts the first 3 in one verification and rejects the 4th, overriding it with its own prediction:

- Serial baseline: producing 4 tokens needs **4** large-model forward passes.
- SD: producing 4 tokens used only **1** large-model forward pass (+ negligible small-model cost).

In that step we compressed "4 passes" into "1" — that is where SD's speedup comes from.

## 6. Summary and what's next

- SD = small-model drafting + large-model parallel verification, preserving the distribution via rejection sampling (**lossless**).
- The speedup comes from **acceptance length α**: the more accurate the draft, the more you save.
- But α has a ceiling: why are classical autoregressive drafters capped at 2–3×? The next post, *Why Autoregressive Drafters Cap at 2–3×*, breaks down the bottleneck and introduces **DFlash** and **DSpark** as two ways out.
