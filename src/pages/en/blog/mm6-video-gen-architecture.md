---
title: 'Multimodal Decoding Notes (6): The Architectural Fork in Video Generation — MAGI-2 Unified Single Stream vs Seedance 2.5 Dual Branch'
description: 'Continuing from Part 3 on diffusion fundamentals, this chapter fills in how a video-generation model is actually wired. Using Sand.ai MAGI-2 (480B total / 28B active Multi-Head Latent MoE on a single unified token stream) and ByteDance Seedance 2.5 (dual-branch visual and audio DiT with native 4K progressive training) as specimens, it dissects three watersheds: modality fusion, MoE scaling, and the high-resolution path, then explains where the 0.5 CNY per second price comes from.'
pubDate: 2026-11-25
series: Multimodal Decoding Notes
lang: en
altLang: zh
altHref: /blog/mm6-video-gen-architecture
layout: ../../../layouts/BlogPost.astro
---

## 0. Why a Separate Chapter

Part 3 covered the **fundamentals** of diffusion — noise and denoising, latent space, the LDM template. But between those fundamentals and "shoot a 30-second clip with sound" there sits an entire layer of **architectural engineering**: how attention works over space and time, how audio enters the model, how parameters are scaled, and where 4K comes from.

November 2026 handed us two excellent specimens. Sand.ai released **MAGI-2** on November 25, and ByteDance released **Seedance 2.5** at the Volcano Engine FORCE conference in June. The two overlap heavily on surface specs — duration, resolution, synchronized audio-visual output — yet underneath they take **two nearly opposite paths**. This chapter takes those paths apart.

> **Credibility note (important)**: For MAGI-2, "single stream + Multi-Head Latent MoE + 480B/28B + 4T tokens" is Sand.ai official language and can be treated as fact. Seedance 2.5 **has not published a complete technical report**; ByteDance disclosed only partial details at the FORCE conference and in developer documentation. The "dual-branch DiT / 50 references / progressive 4K" picture comes from official disclosure layered with third-party teardown — **high-confidence inference**, not official confirmation. Each claim below is tagged by source strength, which is the habit any serious technical tracker should keep.

## 1. Watershed One: How Modalities Enter the Model

This is the most fundamental difference between the two paths. Nearly every other difference is downstream of it.

<div class="fig">
  <svg viewBox="0 0 680 210" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="165" y="22" text-anchor="middle" font-size="12" fill="#534AB7" font-weight="600">MAGI-2 · single stream</text>
    <text x="505" y="22" text-anchor="middle" font-size="12" fill="#0F6E56" font-weight="600">Seedance 2.5 · dual branch</text>
    <line x1="340" y1="32" x2="340" y2="196" stroke="#D3D1C7" stroke-width="1" stroke-dasharray="3 4"/>

    <rect x="30" y="36" width="88" height="30" rx="6" fill="#EEEDFE" stroke="#534AB7"/>
    <text x="74" y="55" text-anchor="middle" font-size="11" fill="#3C3489">text tokens</text>
    <rect x="122" y="36" width="88" height="30" rx="6" fill="#EEEDFE" stroke="#534AB7"/>
    <text x="166" y="55" text-anchor="middle" font-size="11" fill="#3C3489">video tokens</text>
    <rect x="214" y="36" width="88" height="30" rx="6" fill="#EEEDFE" stroke="#534AB7"/>
    <text x="258" y="55" text-anchor="middle" font-size="11" fill="#3C3489">audio tokens</text>

    <path d="M74 66 L74 84" fill="none" stroke="#888780" stroke-width="1.2"/>
    <path d="M166 66 L166 84" fill="none" stroke="#888780" stroke-width="1.2"/>
    <path d="M258 66 L258 84" fill="none" stroke="#888780" stroke-width="1.2"/>

    <rect x="30" y="84" width="272" height="34" rx="8" fill="#CECBF6" stroke="#534AB7"/>
    <text x="166" y="105" text-anchor="middle" font-size="12" fill="#26215C">concatenated into one sequence</text>

    <path d="M166 118 L166 136" fill="none" stroke="#888780" stroke-width="1.2"/>
    <rect x="30" y="136" width="272" height="44" rx="8" fill="#EEEDFE" stroke="#534AB7"/>
    <text x="166" y="154" text-anchor="middle" font-size="12" fill="#3C3489">one backbone, one forward pass</text>
    <text x="166" y="171" text-anchor="middle" font-size="11" fill="#534AB7">audio and image couple in shared attention</text>

    <rect x="370" y="36" width="130" height="30" rx="6" fill="#E1F5EE" stroke="#0F6E56"/>
    <text x="435" y="55" text-anchor="middle" font-size="11" fill="#085041">text / visual tokens</text>
    <rect x="510" y="36" width="130" height="30" rx="6" fill="#E1F5EE" stroke="#0F6E56"/>
    <text x="575" y="55" text-anchor="middle" font-size="11" fill="#085041">audio tokens</text>

    <path d="M435 66 L435 84" fill="none" stroke="#888780" stroke-width="1.2"/>
    <path d="M575 66 L575 84" fill="none" stroke="#888780" stroke-width="1.2"/>

    <rect x="370" y="84" width="130" height="44" rx="8" fill="#9FE1CB" stroke="#0F6E56"/>
    <text x="435" y="102" text-anchor="middle" font-size="12" fill="#04342C">visual DiT</text>
    <text x="435" y="119" text-anchor="middle" font-size="11" fill="#0F6E56">spatiotemporal attention</text>

    <rect x="510" y="84" width="130" height="44" rx="8" fill="#9FE1CB" stroke="#0F6E56"/>
    <text x="575" y="102" text-anchor="middle" font-size="12" fill="#04342C">audio DiT</text>
    <text x="575" y="119" text-anchor="middle" font-size="11" fill="#0F6E56">own denoising trajectory</text>

    <rect x="370" y="140" width="270" height="24" rx="6" fill="#0F6E56"/>
    <text x="505" y="156" text-anchor="middle" font-size="11" fill="#E1F5EE">cross-layer cross-attention for alignment</text>

    <text x="166" y="196" text-anchor="middle" font-size="11" fill="#5F5E5A">simple, naturally synced, audio is the weak spot</text>
    <text x="505" y="182" text-anchor="middle" font-size="11" fill="#5F5E5A">audio controllable, but two backbones are costly</text>
  </svg>
</div>

**MAGI-2 goes single stream**: text, video, and audio tokens are all concatenated into **one continuous token sequence**, processed by a single Transformer backbone in a single forward pass. The upside is simplicity — no extra cross-modal alignment modules, and audio and picture are synchronized by construction inside the same attention window, so lip movement and impact sounds need no post-hoc alignment. The cost is that audio competes for the same token budget as vision and **gets too little capacity**. Public evaluation of MAGI-2 says exactly this: picture quality around Seedance 2.3 level, strong camera work and physics, **audio somewhat weak**. That is not a tuning issue but a built-in architectural bias.

**Seedance 2.5 goes dual branch**: one visual stream, one audio stream, each with its own backbone and denoising trajectory, communicating only through **cross-layer cross-attention** at the transformer base attention layer. The upside is independent capacity and optimization objectives for audio, which is why it achieves phoneme-level lip sync, 48kHz stereo, and timbre locking from a reference clip. The cost is two backbones, making training and scaling more expensive.

An intuition: single stream is like **one performer who speaks and acts at the same time** — sync is free, but attention is split. Dual branch is like **a voice actor and an on-screen actor sharing the stage** — each is specialized, and they are aligned by a conductor (cross-attention).

## 2. Inside MAGI-2

### 2.1 Multi-Head Latent MoE: Rewriting MoE for Video Training

The real innovation is not "it uses MoE" but **why conventional MoE works poorly on video, and how it was changed**.

The pain point of conventional MoE is brutally amplified at video scale. At hundreds of billions of parameters, the token stream mixes video, audio, and reference material, so expert routing is **inherently unbalanced**. Cross-GPU All-to-All traffic swings wildly: some GPUs saturate, others idle, and utilization collapses under communication.

The fix in Multi-Head Latent MoE is to **decouple communication from routing and fix the order**:

```
Conventional MoE:  route first → All-to-All by routing result → traffic jitters with routing
MH-Latent MoE:     split latent space into K independent subspaces
                   → complete cross-GPU exchange first, in a fixed order (constant, pre-schedulable traffic)
                   → then perform expert routing locally
```

The key move is **deferring the uncertain routing decision until after communication**. The communication pattern becomes statically schedulable, eliminating jitter. Official measurement: at equal compute in 480B pretraining, about **22% faster** than LatentMoE.

Parameter accounting: 480B total, roughly **28B active per token**, a sparsity ratio near 17:1. The previous MAGI-2 Preview was 114B / 6B active — total parameters grew 4.2x while active grew 4.7x, essentially proportional scaling, indicating **scaling came from adding experts rather than adding depth**.

### 2.2 System Layer: Four In-House Components

A good architecture is worthless if 480B cannot be deployed. Sand.ai shipped four pieces:

| Component | What it solves |
|---|---|
| **Head Parallel** | Splits attention heads across devices to avoid activation memory peaks |
| **MagiFSDP** | Fully sharded data parallelism with activation recomputation tuned for long video sequences |
| **MagiMoE** | Expert-parallel communication scheduling paired with the fixed exchange order of MH-Latent MoE |
| **MagiMuon** | A Muon optimizer stabilized for 480B scale |

One deployment detail worth noting: this stack has been **adapted to domestic Chinese silicon including Hygon and Moore Threads**. For a model promising open weights, that means domestic teams can realistically self-host.

### 2.3 The 1080p Is Not Natively Rendered

This is the point most easily glossed over by marketing. MAGI-2 does **not** run its first denoising pass on a full 1080p pixel grid. The pipeline denoises in a lower-resolution latent space first, then uses a **dedicated latent spatiotemporal upscaler** to reach 1080p.

It also applies a clever patch: **the original reference images are re-injected during upscaling**. What upscaling destroys first is small faces and on-screen text; re-injecting references pins those high-frequency details back in place.

The implication: **compute goes into getting motion and composition right, not into computing every pixel from scratch**. This is one structural reason the price reaches 0.5 CNY per second. Compared with the native 4K of Seedance 2.5, these are two entirely different accounting models.

### 2.4 Data and Output Specs

- Pretraining on **4T tokens** (about one billion samples), roughly 3x the previous generation, annotated in three weeks
- Deliberate data mix: preserving real-world distributions such as **multi-person interaction, complex motion, frequent shot cuts, on-screen text, and complex sound fields**
- Output: 4–30 seconds, 1080p, 25fps, native synchronized audio and video, six aspect ratios
- Reference capacity: up to 20 items per request (10 images, 5 videos, 5 audio clips), each assignable to a specific role
- Prompts: up to 20,000 characters, with timeline control accurate to about 0.1 seconds

The "preserve hard samples" point deserves its own sentence. Most teams filter blurry, shaky, frequently-cut footage out of pretraining to get a cleaner loss. Sand.ai does the opposite — **because real-world video actually looks like that**. Filter it out and the learned distribution is narrower than reality, which post-training cannot repair. This is a concrete case of "data sets the ceiling; post-training only sets the floor."

## 3. Seedance 2.5, Taken Apart

### 3.1 Dual-Branch DiT and Two Kinds of Attention

<div class="fig">
  <svg viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <rect x="30" y="20" width="620" height="52" rx="10" fill="#E1F5EE" stroke="#0F6E56"/>
    <text x="340" y="42" text-anchor="middle" font-size="12" fill="#04342C" font-weight="600">50 multimodal reference inputs</text>
    <text x="340" y="60" text-anchor="middle" font-size="11" fill="#0F6E56">injected per semantic channel, not averaged into one muddy mix</text>

    <path d="M340 72 L340 88" fill="none" stroke="#888780" stroke-width="1.2"/>
    <rect x="30" y="88" width="620" height="34" rx="8" fill="#9FE1CB" stroke="#0F6E56"/>
    <text x="340" y="109" text-anchor="middle" font-size="12" fill="#04342C">decoupled spatial-ID attention cross-layer · ~30 faces + 10 motion + 10 timbre</text>

    <path d="M200 122 L200 138" fill="none" stroke="#888780" stroke-width="1.2"/>
    <path d="M480 122 L480 138" fill="none" stroke="#888780" stroke-width="1.2"/>

    <rect x="30" y="138" width="340" height="44" rx="8" fill="#E1F5EE" stroke="#0F6E56"/>
    <text x="200" y="155" text-anchor="middle" font-size="12" fill="#04342C">hierarchical temporal attention: short / mid / long</text>
    <text x="200" y="172" text-anchor="middle" font-size="11" fill="#0F6E56">suppresses flicker and morphing over 30 seconds</text>

    <rect x="390" y="138" width="260" height="44" rx="8" fill="#E1F5EE" stroke="#0F6E56"/>
    <text x="520" y="155" text-anchor="middle" font-size="12" fill="#04342C">decoupled spatiotemporal attention</text>
    <text x="520" y="172" text-anchor="middle" font-size="11" fill="#0F6E56">temporal and spatial computed separately</text>
  </svg>
</div>

Three components with distinct jobs:

- **Decoupled spatiotemporal attention**: temporal and spatial cross-attention are computed separately. The motivation is practical — what breaks first over 30 continuous seconds is kinematic consistency (where things should move) and lighting consistency (where light comes from). Those two are modeled at different scales and interfere with each other when crammed into one attention.
- **Hierarchical temporal attention**: short range for frame-to-frame continuity, mid range for within-shot motion, long range for narrative structure across the clip. The point of layering is to **avoid attention dilution when a single-scale temporal attention spans 30 seconds**.
- **Decoupled spatial-ID attention cross-layer**: this layer specifically fights feature muddiness across 50 references. Earlier approaches (IP-Adapter, ControlNet) **average** multiple reference features into a muddy composite. Here the references are split into discrete semantic channels — about 30 for faces and physique, 10 for motion clips, 10 for timbre. In effect, **classify before injecting**, so conditions of different natures do not contaminate each other.

### 3.2 Native 4K and Progressive Resolution Training

Seedance 2.5 uses **progressive resolution training** from 480p to 1080p to 4K, with final output as native 4K at 10-bit color depth. This mirrors MAGI-2:

| | MAGI-2 | Seedance 2.5 |
|---|---|---|
| Source of high resolution | latent spatiotemporal upscaler plus reference re-injection | resolution increased during training itself |
| Strength | cheap, low cost, high frequencies patched by re-injection | genuine high-frequency detail, no upscaling artifacts, Rec.2020 grading headroom |
| Weakness | extreme detail (hair, text) depends on upscaler quality | very high training cost, steep compute at the 4K tier |

Native 4K costs an order of magnitude more at the same spec, essentially the price of **running the first denoising pass on a high-dimensional grid**.

### 3.3 The Data Flywheel

This is the layer of Seedance that is hardest to replicate. Training data comes from opt-in licensed TikTok and Douyin video plus synthetic data, estimated at **over 100 million hours**. Against models like Sora trained mainly on licensed stock footage and public web video, the gap is not only volume but **breadth of distribution** — real users walking, dancing, fighting, embracing in uncontrolled environments, a diversity no stock library provides.

ByteDance being both model developer and platform operator is a genuine moat for video generation: data scale, real motion distribution, and camera-language diversity all flow from the same source.

## 4. Head-to-Head Across Six Dimensions

| Dimension | MAGI-2 | Seedance 2.5 | Nature of the fork |
|---|---|---|---|
| **Modality fusion** | single unified token stream | visual and audio branches plus cross-layer alignment | shared capacity vs specialized capacity |
| **Backbone** | Multi-Head Latent MoE, 480B / 28B active | undisclosed (dual-branch DiT) | sparse scaling vs data scaling |
| **Scaling mechanism** | add experts; decoupling communication from routing buys 22% faster training | add data (100M+ hours) plus progressive resolution | engineering efficiency vs data volume |
| **High-resolution path** | latent upscaling with reference re-injection | native 4K progressive training | save compute vs preserve detail |
| **Reference capacity** | 20 items, each role-assignable | 50 items, split by semantic channel | flexibility vs resistance to blurring |
| **Signature control** | 20K-character prompts, 0.1s timeline precision, language swap preserving framing and lip sync | 3D white-box previz, local mask repaint, frame-level timestamp keyframes | long-script orchestration vs industrial pipeline integration |
| **Distribution** | open-weights promise (currently third-party hosted) | closed, via Volcano Engine / Jimeng / Dreamina | opposite ecosystem strategies |

One cross-check worth recording: on the ArtificialAnalysis video leaderboard, **MAGI-2 already surpasses Seedance 2.5 on image-to-video (I2V)**, yet its fine-grained picture quality is rated only "around Seedance 2.3," with weak audio.

That seemingly contradictory pair of data points says one thing: **sparse MoE scaling buys stability in complex narrative, camera scheduling, and instruction following — not pixel-level fidelity**. Fidelity — fabric texture, water caustics, hair edges — only comes from data density and native high resolution. The first is "thinking correctly," the second is "seeing clearly," two distinct capability axes.

## 5. Why Two Paths: The Compute Economics

Compress all of the above into a sentence:

> **MAGI-2 is solving "how to make a usable 30 seconds with less compute." Seedance 2.5 is solving "how to make a 30 seconds that can enter a studio with enough compute and data."**

MAGI-2 at 0.5 CNY per second (1080p list price) is not subsidized. It is three accounts stacked:

1. **Sparse MoE activation**: 480B total runs only 28B, so per-token compute is low
2. **Training efficiency**: decoupled communication and routing saves 22% training time at 480B scale
3. **Upscaled 1080p**: the first denoising pass does not run on a full pixel grid

All three are **structural savings**, not a price war. Seedance 2.5 is the mirror image: native 4K, dual branch, 50 references, 100M hours of data — every item adds cost. It is expensive for a reason, and only **commercial-delivery customers** will pay that reason.

This also explains why both paths converged on "30 seconds plus synchronized audio" at the same moment: **30 seconds is the shared length of Douyin short video, the standard ad spot, and Instagram Story — the smallest narrative unit holding a complete arc**. It is a target set by demand, not a technical self-selection.

## 6. Where Each One Is Weak

No cheerleading; both limitations are clear.

**MAGI-2**
- **Audio is a structural weakness.** In a single-stream architecture audio gets limited capacity. Tuning cannot fix it; only architecture changes can.
- **The open-source promise is unfulfilled.** Weights were not released at the November 25 launch; only third-party hosting (Higgsfield) is available. Promised and released are two different things — track the release date and license.
- **Upscaling carries detail risk.** Small faces, text, and fine textures can distort during upscaling. Reference re-injection mitigates but does not eliminate this.

**Seedance 2.5**
- **No complete technical report.** All "dual branch / 50 references / progressive 4K" details may be revised by a formal paper, so leave margin in technical judgments.
- **Closed ecosystem.** Tied to Volcano Engine / Jimeng / Dreamina, with no self-hosting and no fine-tuning — a hard constraint for workflows needing private deployment.
- **Complex physical fluids still hallucinate.** Public comparisons list this explicitly as a weakness, paid for in retries.
- **Expensive.** The compute draw at native 4K restricts it to scenarios with clear delivery budgets.

## 7. Practical Takeaways

**Technical side, three transferable lessons:**

1. **The bottleneck in MoE is communication, not parameter count.** The reordering of "exchange first in fixed order, route locally after" bought 22% faster training — a general pattern worth trying for any large-scale MoE, language models included. See `sys13-moe-expert-parallel-comms` for the communication analysis.
2. **Do not over-filter hard samples in pretraining.** Keeping multi-person interaction, complex motion, and frequent cuts is the same principle as "do not narrow the real distribution with quality filters" in multimodal and embodied training. Post-training fixes the floor, not the ceiling.
3. **For multi-reference control, the answer is channel separation, not more capacity.** Seedance splitting 50 references into discrete semantic channels is the correct anti-blurring approach. Any multi-condition control work (ControlNet-class) can copy this directly.

**Investment side, three things to watch:**

- **The pace of delivering the open-source promise matters more than the launch.** If MAGI-2 truly releases 480B weights, the open-source ceiling for video generation rises to closed-source frontier level — **valuation pressure on video-generation startups**, and **demand tailwind for compute and inference infrastructure** (sparse or not, 480B needs real VRAM).
- **Pricing structure determines market stratification.** 0.5 CNY per second versus a tenfold price implies a clean two-tier market: **cheap open-source for bulk content production, expensive closed-source for commercial delivery**. The middle tier suffers most.
- **The data flywheel moat belongs only to platform operators.** That 100M-hour real video distribution is unreplicable for any pure model team. When evaluating video-generation companies, "does it own a data source" matters more than "how big is the model" — the same logic applies to embodied AI, where the real-robot-data versus simulation-data debate is the same question.

## 8. Series Map

```
(1) Foundations & alignment   → multimodality = extending the generative space beyond text
(2) VLM evolution             → ViT / CLIP / LLaVA: the standard see-and-say paradigm
(3) Generative models         → diffusion / GAN: the fundamentals of drawing
(4) VLA & world models        → RT-2 → pi0.7 → HiF-VLA: acting, and thinking before acting
(5) Efficiency & frontier     → attention / inference optimization / on-device / RL: the substrate
(6) Video-gen architecture    → single stream vs dual branch: how a 30s clip with sound is built
```

Part 6 is the link connecting the fundamentals of Part 3 to **contemporary industrial video models**. Natural continuations: the dual-expert MoE architecture of Wan2.2 (deployment already covered in `fw9-wan22-vllm-omni`, architecture still open), the physical-simulation path of Kling 3.0, and the bridge from video generation to world models — which is exactly the extension of Part 4 into embodied intelligence.
