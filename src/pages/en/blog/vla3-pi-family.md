---
title: 'VLA Decoding Notes (3): The π Family — Physical Intelligence’s VLA Lineage'
description: 'From π0 to π0.7, how Physical Intelligence pushed VLA toward "better than specialist policies". The core is a structural timeline from π0 to π0.5 (continuous head → discrete tokens → two heads with hierarchy), plus a component-by-component comparison, then the backbone upgrade, cross-embodiment, RL specialization, and π0.7 memory and compositional generalization.'
pubDate: 2026-08-04
series: VLA Decoding Notes
lang: en
altLang: zh
altHref: /blog/vla3-pi-family
layout: ../../../layouts/BlogPost.astro
---

## 1. π0: A "Universal Interface" for VLA (2024-10)

Physical Intelligence's **π0** set the baseline:

- **Backbone**: PaliGemma (~3B VLM) + a separate **action expert** using **Flow Matching** for continuous actions at **50Hz**.
- **Data**: trained on OXE + proprietary demos, covering **7 robot platforms, 68 tasks**.
- **Unified interface**: language/image in, continuous action out. Proof that "one general VLA can work across embodiments".

## 2. π0-Fast: Discrete Speedup

Adds **action discretization + autoregressive** on top of π0 for inference speed — an engineering trade-off between the continuous vs discrete routes, for latency-sensitive settings.

## 3. π0.5: Cross-Embodiment + Task Decomposition (2025-04)

- **Cross-embodiment co-training**: multiple different robots share one "general brain", learning hardware-agnostic manipulation common sense.
- **Task decomposition**: a high level splits complex instructions into sub-goals; low-level policies execute them in turn — making long-horizon tasks tractable.

## 4. The Structural Timeline: What Actually Changed from π0 to π0.5

The first three sections describe capability gains. This one describes parts being swapped. The place where the π family genuinely rewires **structure** is the π0 → π0.5 stretch, and the **parameter count barely moves** — what changes is the order of three decisions:

1. **Action representation**: π0 uses continuous vectors through flow matching; π0-Fast switches to discrete tokens through autoregression.
2. **Head count**: π0 and π0-Fast each have one head; π0.5 splits one head into two (a FAST discrete head plus a flow head).
3. **Gradient routing**: after π0.5, two heads trained jointly started fighting each other, so Knowledge Insulation arrived — stop-grad cuts the expert-to-backbone gradient without touching a single module.

<div class="fig">
<svg id="fig-pi-tl" viewBox="0 0 680 768" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" aria-labelledby="pi-tl-t pi-tl-d">
<title id="pi-tl-t">Structural timeline from π0 to π0.5</title>
<desc id="pi-tl-d">A vertical timeline of six nodes: π0 with a single continuous head, π0-Fast with discrete tokens, π0.5 with two heads plus hierarchy, Knowledge Insulation gradient blocking, π0.6 with scaling and reinforcement learning, π0.7 with memory and multimodal prompts.</desc>
<style>
#fig-pi-tl{--t1:#2C2C2A;--t2:#5F5E5A;--ln:#DDD9CE;
--b-p:#E6F1FB;--b-s:#185FA5;--b-t:#0C447C;
--t-p:#E1F5EE;--t-s:#1D9E75;--t-t:#04342C;
--g-p:#F1EFE8;--g-s:#CDC9BC;--g-t:#444441;}
:root[data-theme="dark"] #fig-pi-tl{--t1:#F1EFE8;--t2:#D3D1C7;--ln:#3A4046;
--b-p:#042C53;--b-s:#85B7EB;--b-t:#B5D4F4;
--t-p:#04342C;--t-s:#5DCAA5;--t-t:#9FE1CB;
--g-p:#2C2C2A;--g-s:#5F5E5A;--g-t:#D3D1C7;}
#fig-pi-tl text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
#fig-pi-tl .h1{font-size:14px;font-weight:500;fill:var(--t1)}
#fig-pi-tl .sub{font-size:11.5px;fill:var(--t2)}
#fig-pi-tl .dt{font-size:11px;fill:var(--t2);text-anchor:end;dominant-baseline:central}
#fig-pi-tl .tt{font-size:12.5px;font-weight:500;dominant-baseline:central}
#fig-pi-tl .bd{font-size:11.5px;fill:var(--t2);dominant-baseline:central}
#fig-pi-tl .c-b{fill:var(--b-p);stroke:var(--b-s);stroke-width:.5}
#fig-pi-tl .c-t{fill:var(--t-p);stroke:var(--t-s);stroke-width:.5}
#fig-pi-tl .c-g{fill:var(--g-p);stroke:var(--g-s);stroke-width:.5}
#fig-pi-tl .k-b{fill:var(--b-t)}#fig-pi-tl .k-t{fill:var(--t-t)}#fig-pi-tl .k-g{fill:var(--g-t)}
#fig-pi-tl .lk{stroke:var(--ln);stroke-width:1;fill:none}
</style>
<text class="h1" x="40" y="30">π0 → π0.5: a structural timeline</text>
<text class="sub" x="40" y="50">Parameter count barely moves. What changes is representation, then head count, then gradient routing.</text>
<line x1="118" y1="66" x2="118" y2="746" stroke="var(--ln)" stroke-width="1.5"/>

<text class="dt" x="104" y="90">2024-10</text>
<circle cx="118" cy="90" r="5" fill="#185FA5"/>
<line class="lk" x1="123" y1="90" x2="147" y2="90"/>
<rect class="c-b" x="150" y="72" width="490" height="108" rx="8"/>
<text class="tt k-b" x="166" y="96">π0 · one continuous head: VLA gets a universal interface</text>
<text class="bd" x="166" y="120">PaliGemma 3B + 300M action expert, one shared attention stack</text>
<text class="bd" x="166" y="137">Flow Matching: 50 steps × 14 dims = 1 second at 50 Hz</text>
<text class="bd" x="166" y="154">7 embodiments, 68 tasks; language and image in, action out</text>
<text class="bd" x="166" y="171">Structure: single head, dense action vector, no semantic layer</text>

<text class="dt" x="104" y="214">2025-01</text>
<circle cx="118" cy="214" r="5" fill="#185FA5"/>
<line class="lk" x1="123" y1="214" x2="147" y2="214"/>
<rect class="c-b" x="150" y="196" width="490" height="92" rx="8"/>
<text class="tt k-b" x="166" y="220">π0-Fast · swap actions for discrete tokens</text>
<text class="bd" x="166" y="244">FAST tokenizer turns an action chunk into tokens, learned autoregressively</text>
<text class="bd" x="166" y="261">Gains web-data mixing and faster inference; pays with serial decoding</text>
<text class="bd" x="166" y="278">Structure: the action head degrades into a vocabulary</text>

<text class="dt" x="104" y="320">2025-04</text>
<circle cx="118" cy="320" r="6" fill="#1D9E75"/>
<line class="lk" x1="123" y1="320" x2="147" y2="320"/>
<rect class="c-t" x="150" y="302" width="490" height="126" rx="8"/>
<text class="tt k-t" x="166" y="326">π0.5 · two action heads + hierarchy (the inflection point)</text>
<text class="bd" x="166" y="350">One head becomes two: FAST discrete head plus flow continuous head</text>
<text class="bd" x="166" y="367">Loss = CE(text &amp; FAST) + α·FM; pretrain α=0, post-train α=10</text>
<text class="bd" x="166" y="384">At α=0 it degrades to a plain VLM, eating six data types on one objective</text>
<text class="bd" x="166" y="401">Hierarchy: emit a subtask, then generate a 50-step chunk from it</text>
<text class="bd" x="166" y="418">Structure: an extra text exit; two action reps kept apart by masks</text>

<text class="dt" x="104" y="462">2025-05</text>
<circle cx="118" cy="462" r="5" fill="#1D9E75"/>
<line class="lk" x1="123" y1="462" x2="147" y2="462"/>
<rect class="c-t" x="150" y="444" width="490" height="92" rx="8"/>
<text class="tt k-t" x="166" y="468">Knowledge Insulation · change the gradient, not the module</text>
<text class="bd" x="166" y="492">Stop-grad: the action expert no longer pushes gradients into the VLM</text>
<text class="bd" x="166" y="509">Backbone learns actions from FAST tokens; ~7.5× faster convergence</text>
<text class="bd" x="166" y="526">Structure: gradient paths and masks change; topology does not</text>

<text class="dt" x="104" y="570">2025-11</text>
<circle cx="118" cy="570" r="5" fill="#B4B2A9"/>
<line class="lk" x1="123" y1="570" x2="147" y2="570"/>
<rect class="c-g" x="150" y="552" width="490" height="76" rx="8"/>
<text class="tt k-g" x="166" y="576">π0.6* · scaling + RECAP reinforcement learning</text>
<text class="bd" x="166" y="600">Gemma3 4B + 860M expert; offline RL specialization on the trained VLA</text>
<text class="bd" x="166" y="617">Structure: two heads stay; scale and post-training recipe change</text>

<text class="dt" x="104" y="660">2026-04</text>
<circle cx="118" cy="660" r="5" fill="#B4B2A9"/>
<line class="lk" x1="123" y1="660" x2="147" y2="660"/>
<rect class="c-g" x="150" y="642" width="490" height="94" rx="8"/>
<text class="tt k-g" x="166" y="666">π0.7 · memory + multimodal prompts</text>
<text class="bd" x="166" y="690">MEM memory: pulls context from past experience, steadier on long tasks</text>
<text class="bd" x="166" y="707">Image / language / demo prompts; compositional generalization emerges</text>
<text class="bd" x="166" y="724">Structure: a memory and prompt channel sits above the two heads</text>
</svg>
<p class="fig-cap">Fig 1: The structural timeline of the π family. Blue boxes are hard structural changes, green boxes are changes to gradients and training recipes, gray-green boxes are scale and memory layered on top.</p>
</div>

Three ways to read that figure:

- **Blue → green → gray-green** is a curve of increasingly softer changes: swap the representation (hard structural change), then alter loss and gradients (soft recipe change), then layer on scale and memory (capability stacking).
- **π0-Fast is not an upgraded π0; it is a side branch.** It demotes the action head into a vocabulary to buy speed. π0.5 then invites the continuous head back, and the two coexist — which is exactly what makes π0.5 stand on both routes at once.
- **The watershed in π0.5 is α.** The training loss is `CE(text & FAST) + α·FM`: at pretrain time α=0, and the whole model degrades into a plain autoregressive VLM, so web QA, captions, subtask annotations and FAST action tokens can all be eaten by one next-token objective. At post-train time α=10, and the continuous head wakes up to learn fine-grained actions. **One model, two training rhythms, each taking what it needs.**

<div class="fig">
<svg id="fig-pi-cmp" viewBox="0 0 680 482" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" aria-labelledby="pi-cmp-t pi-cmp-d">
<title id="pi-cmp-t">π0 versus π0.5, component by component</title>
<desc id="pi-cmp-d">Six rows comparing input, backbone, intermediate layer, action generation, output and training objective. π0.5 adds a FAST discrete head and a subtask text exit, and conditions the action expert on the subtask.</desc>
<style>
#fig-pi-cmp{--t1:#2C2C2A;--t2:#5F5E5A;--ln:#DDD9CE;
--b-p:#E6F1FB;--b-s:#185FA5;--b-t:#0C447C;
--t-p:#E1F5EE;--t-s:#1D9E75;--t-t:#04342C;
--g-p:#F1EFE8;--g-s:#CDC9BC;--g-t:#444441;}
:root[data-theme="dark"] #fig-pi-cmp{--t1:#F1EFE8;--t2:#D3D1C7;--ln:#3A4046;
--b-p:#042C53;--b-s:#85B7EB;--b-t:#B5D4F4;
--t-p:#04342C;--t-s:#5DCAA5;--t-t:#9FE1CB;
--g-p:#2C2C2A;--g-s:#5F5E5A;--g-t:#D3D1C7;}
#fig-pi-cmp text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
#fig-pi-cmp .h1{font-size:14px;font-weight:500;fill:var(--t1)}
#fig-pi-cmp .sub{font-size:11.5px;fill:var(--t2)}
#fig-pi-cmp .rl{font-size:11px;fill:var(--t2);dominant-baseline:central}
#fig-pi-cmp .tt{font-size:12.5px;font-weight:500;text-anchor:middle}
#fig-pi-cmp .bd{font-size:11.5px;fill:var(--t2);dominant-baseline:central}
#fig-pi-cmp .c-b{fill:var(--b-p);stroke:var(--b-s);stroke-width:.5}
#fig-pi-cmp .c-t{fill:var(--t-p);stroke:var(--t-s);stroke-width:.5}
#fig-pi-cmp .c-g{fill:var(--g-p);stroke:var(--g-s);stroke-width:.5}
#fig-pi-cmp .k-b{fill:var(--b-t)}#fig-pi-cmp .k-t{fill:var(--t-t)}#fig-pi-cmp .k-g{fill:var(--g-t)}
</style>
<text class="h1" x="40" y="30">π0 versus π0.5, component by component</text>
<text class="sub" x="40" y="50">Same backbone, two extra exits; the action head goes from independent to subtask-conditioned</text>
<text class="tt k-b" x="248" y="82">π0 · 2024-10</text>
<text class="tt k-t" x="510" y="82">π0.5 · 2025-04</text>
<line x1="40" y1="96" x2="640" y2="96" stroke="var(--ln)" stroke-width="1"/>

<text class="rl" x="40" y="125">Input</text>
<rect class="c-b" x="128" y="102" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="125">images + language</text>
<rect class="c-b" x="380" y="102" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="125">images + language + state as text</text>

<text class="rl" x="40" y="181">Backbone</text>
<rect class="c-b" x="128" y="158" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="181">PaliGemma 3B</text>
<rect class="c-b" x="380" y="158" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="181">same family, 2B language trunk</text>

<text class="rl" x="40" y="237">Middle</text>
<rect class="c-g" x="128" y="214" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="237">no semantic layer</text>
<rect class="c-t" x="380" y="214" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="237">FAST head + subtask text exit</text>

<text class="rl" x="40" y="293">Action</text>
<rect class="c-b" x="128" y="270" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="293">300M expert, flow matching</text>
<rect class="c-t" x="380" y="270" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="293">same expert, conditioned on subtask</text>

<text class="rl" x="40" y="349">Output</text>
<rect class="c-b" x="128" y="326" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="349">chunk of 50 × 14 at 50 Hz</text>
<rect class="c-b" x="380" y="326" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="349">same chunk, one semantic hop earlier</text>

<text class="rl" x="40" y="405">Loss</text>
<rect class="c-g" x="128" y="382" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="405">pure flow matching</text>
<rect class="c-g" x="380" y="382" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="405">CE plus α·FM, α from 0 to 10</text>

<text class="bd" x="40" y="443">What differs is the middle layer and the α in the loss.</text>
<text class="bd" x="40" y="460">The middle layer sets long-horizon reach; α sets whether training looks like a VLM or a policy.</text>
</svg>
<p class="fig-cap">Fig 2: π0 and π0.5 side by side. π0.5 splits one continuous exit into three: a discrete head, a subtask, and a continuous head.</p>
</div>

In one line: **π0 sees and acts immediately; π0.5 sees, says what it is about to do, then acts.** That extra sentence in the middle is the whole secret of long-horizon tasks — it slices a 15-minute cleanup into checkable sub-goals, so a failure can retreat to the sub-goal level instead of starting over.

## 5. π0.6*: RL Specialization (RECAP)

Uses **reinforcement learning** to specialize the trained VLA (RECAP-style), pushing specific-task success rates further. The typical second half of "general pre-train + specialist post-train".

## 6. π0.7: Memory + Compositional Generalization (2026-04)

π0.7 is the current apex:

- **Backbone upgrade**: Gemma3 4B + 860M action expert.
- **MEM memory**: learns from past experience, carries contextual memory for steadier long multi-step tasks.
- **Multimodal prompts**: image / language / demonstration mixed prompts — "do it like the demo".
- **Emergent compositional generalization**: generalizes to unseen instruction combinations — a signal of moving toward "truly general".
- **Result**: beats **specialist policies** on several benchmarks, not just matching them.

## 7. The Through-Line (One Line)

> VLM backbone upgrade → cross-embodiment reuse → RL specialization → memory + compositional generalization. **General VLA is moving from "can work" to "stronger than specialists".**

## 8. Bridge

The π family is the overseas benchmark. Next: domestic players — Ant Lingbo's LingBot-VLA 2.0 "one brain, many machines", Xiaomi's open real-time VLA, Tencent HyVLA's FlowPRO — plus a clarification of a common mix-up.
