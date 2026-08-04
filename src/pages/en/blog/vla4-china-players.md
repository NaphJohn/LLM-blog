---
title: 'VLA Decoding Notes (4): Domestic Players — Ant Lingbo, Xiaomi, Tencent'
description: 'A panorama of domestic VLA deployment: Ant Lingbo LingBot-VLA 2.0 "one brain, many machines" cross-embodiment; Xiaomi Xiaomi-Robotics-0 open real-time VLA; Tencent HyVLA FlowPRO. Plus a clarification — "Hy3" is Tencent, not Xiaomi.'
pubDate: 2026-08-04
series: VLA Decoding Notes
lang: en
altLang: zh
altHref: /blog/vla4-china-players
layout: ../../../layouts/BlogPost.astro
---

## 0. Clarify a Mix-Up First

The "**Xiaomi also has Hy3**" you mentioned needs correcting: **Hy3 / HyVLA belongs to Tencent's Hunyuan embodied team, not Xiaomi.** Xiaomi's VLA model is **Xiaomi-Robotics-0**. We cover all three separately below.

## 1. Ant Lingbo: LingBot-VLA 2.0 "One Brain, Many Machines"

- **Background**: Ant Lingbo Tech, a wholly-owned Ant Group subsidiary, founded 2025-03 in Shanghai Pudong, focused on the robot "general brain". Started its first independent funding round in 2026-08, targeting ~¥1.5B.
- **Flagship LingBot-VLA 2.0** (released 2026-07): "**one brain, many machines**" — pre-trained to adapt to **17 brands (Unitree, AgiBot, Galbot, Leju, etc.), 20+ robot configurations**, showing cross-embodiment reuse and generalization.
- **LingBot model matrix**: VLA 2.0 (base / cross-embodiment), VA 2.0 (world-action model), Vision (spatial-native visual base model), World 2.0 (infinite-duration real-time interactive world model), Video (MoE open video generation).
- **Industry take (CEO Zhu Xing)**: in past years "cerebellum" and hardware advanced fast, but **the brain is the bottleneck**; future "AI will redefine hardware".

## 2. Xiaomi: Xiaomi-Robotics-0 (Open Real-Time VLA)

- **Released**: open-sourced 2026-02, 4.7B params, announced by Lei Jun, weights/HF/code all public.
- **Architecture**: MoT — VLM brain + 16-layer **DiT cerebellum**, KV Cache loosely coupled (see Part 2).
- **Performance**: **80ms latency, 30Hz, real-time on RTX 4090**, SOTA on LIBERO / CALVIN / SimplerEnv.
- **Highlight**: solves "action discontinuity" via async inference + Action Proposal + Clean Action Prefix + Λ-shape attention.
- **Significance**: a big lab open-sourcing a consumer-grade real-time VLA lowers the industry barrier and feeds a factory data flywheel (90.2% → 98% frame-by-frame optimization).

## 3. Tencent: HyVLA and FlowPRO

- **FlowPRO**: uses **flow-matching regression loss as an implicit reward proxy**, no extra Critic network; four long-horizon bimanual tasks hit **94%–99%** success, beating traditional DAgger's 83%–93%.
- **Data**: direct optical motion-capture, sub-millimeter precision.
- **Positioning**: Tencent Hunyuan (Hy family) methodology output for embodied AI — emphasizing that post-training innovation affects deployment more than parameter scale.

## 4. Landscape

| Player | Model | Trait | Posture |
|---|---|---|---|
| Ant Lingbo | LingBot-VLA 2.0 | one brain many machines | funding expand |
| Xiaomi | Xiaomi-Robotics-0 | open / real-time / 4090 | open + factory |
| Tencent | HyVLA (FlowPRO) | implicit-reward post-train | methodology |
| AgiBot/Unitree/Galbot/Leju | own embodiment+policy | strong hardware/cerebellum | ecosystem fed in |

> Common trend: the "**compete on brains**" stage — whoever first has a complete model system + real-scene data loop runs out a platform-level company. Ant Lingbo already plugs 17 brands into one brain, an upstream "brain supplier" logic over cerebellum/hardware vendors (Unitree, etc.).

## 5. Bridge

Domestic players push VLA toward "cross-embodiment + open-source + engineering loop". Next: VLA's next stop — world models, Sim-to-Real, data flywheels, and the scaling bottleneck.
