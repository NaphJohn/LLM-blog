---
title: 'VLA Notes (6): RT-1''s Discrete-Action Route — TokenLearner, Causal Mask, and Cross-Entropy'
description: 'RT-1 was the first system to frame robot control as language generation. This post breaks down how it turns images into tokens with EfficientNet, compresses 6 frames of 486 tokens down to 48 with TokenLearner, and decodes actions with a decoder-only Transformer using causal masks and cross-entropy loss — including the common variants of both.'
pubDate: 2026-08-07
series: VLA Notes
lang: en
altLang: zh
altHref: /blog/vla6-rt1-discrete-action
layout: ../../../layouts/BlogPost.astro
---

## 1. Why RT-1 Chose Discrete Action Tokens

In Part 1 of this series we said there are two ways to represent robot actions:

- **Continuous generation**: directly regress joint angles and velocities for smooth trajectories — the territory of π0 and Xiaomi Robotics today.
- **Discrete tokenization**: quantize continuous actions into discrete symbols and generate them autoregressively, like an LLM writing a sentence.

**RT-1 (Google Robotics, 2022)** took the second path. The core idea is simple: if LLMs can learn language structure from trillions of text tokens, why not treat actions as tokens and reuse the whole language-model training stack — autoregression, Transformers, cross-entropy?

The downside is obvious: quantizing continuous motion throws away precision and can make trajectories jittery. But RT-1 proved the **engineering simplicity** and **scalability** of the route, paving the way for RT-2 and OpenVLA.

> This post answers the two questions that most often confuse beginners:
> 1. **How do 6 frames become 486 visual tokens, and how does TokenLearner squeeze them to 48?**
> 2. **In the decoder-only action decoder, what exactly are "standard causal mask + categorical cross-entropy loss", and what variants exist?**

---

## 2. Vision Side: EfficientNet + TokenLearner

### 2.1 From Image to Tokens: 81 per Frame

RT-1's input is **six consecutive robot-camera images**.

Why six? At a control rate of **3 Hz**, six frames cover roughly the **last two seconds** of observation. The model needs this short-term memory to judge where objects are moving, where the gripper is, and how the previous action is unfolding.

Each 300×300 RGB image first passes through an **EfficientNet-B3** visual encoder:

- The final feature map is **9×9×512**.
- Flattening the 9×9 spatial grid gives **81 visual tokens**, each a 512-d vector.

So for one frame:

```text
300×300 image  →  EfficientNet-B3  →  9×9×512  →  81 tokens (512-d)
```

### 2.2 Six Frames Explode the Token Count

Six frames, 81 tokens each, gives:

```text
6 × 81 = 486 visual tokens
```

That is before language instruction tokens and special tokens. Four hundred and eighty-six visual tokens is too expensive for 3 Hz real-time inference — Transformer cost roughly scales with the square of sequence length.

### 2.3 TokenLearner: 81 Tokens → 8 Tokens

RT-1's solution is **TokenLearner (Google, 2021)**. Instead of uniformly downsampling the feature map, it lets the model **decide which spatial locations matter**.

How it works (element-wise attention):

1. For every spatial position in the 9×9 feature map, compute an **importance score**.
2. The score is element-wise: it looks not only at spatial location but also at which of the 512 channel responses are strong.
3. Use these scores to **adaptively aggregate** the 81 tokens into **8 representative tokens**.

Intuitive analogy:

> TokenLearner is like a photographer who doesn't shrink the whole photo uniformly. Instead, it automatically zooms in on the few regions most relevant to the manipulation task and blurs the background.

Result:

```text
6 frames × 81 tokens  →  TokenLearner  →  6 frames × 8 tokens = 48 visual tokens
```

In the RT-1 paper this raises inference speed by about **2.4×**.

<div class="fig">
  <svg viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="340" y="22" text-anchor="middle" font-size="15" font-weight="bold" fill="#1a1a1a">RT-1 Vision Compression: From 486 to 48 Tokens</text>

    <!-- 6 frames -->
    <g transform="translate(30,50)">
      <text x="0" y="0" font-size="12" fill="#555">Input: 6 history frames (3 Hz ≈ 2 seconds)</text>
      <rect x="0" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
      <rect x="47" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
      <rect x="94" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
      <rect x="141" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
      <rect x="188" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
      <rect x="235" y="12" width="42" height="42" fill="#E8F0FE" stroke="#4285F4" rx="5"/>
    </g>

    <!-- EfficientNet -->
    <rect x="30" y="125" width="130" height="55" rx="8" fill="#E6F4EA" stroke="#34A853"/>
    <text x="95" y="150" text-anchor="middle" font-size="12" fill="#1a1a1a">EfficientNet-B3</text>
    <text x="95" y="168" text-anchor="middle" font-size="10" fill="#555">300×300 → 9×9×512</text>

    <!-- arrow -->
    <line x1="95" y1="102" x2="95" y2="123" stroke="#888" stroke-width="2" marker-end="url(#rtaen)"/>

    <!-- per-frame tokens -->
    <g transform="translate(180,125)">
      <text x="0" y="18" font-size="11" fill="#555">81 tokens/frame</text>
      <text x="0" y="36" font-size="11" fill="#555">6 frames = 486 tokens</text>
      <text x="0" y="54" font-size="11" fill="#EA4335">Too expensive for 3Hz</text>
    </g>

    <!-- TokenLearner -->
    <rect x="330" y="125" width="130" height="55" rx="8" fill="#FEF7E0" stroke="#FBBC04"/>
    <text x="395" y="150" text-anchor="middle" font-size="12" fill="#1a1a1a">TokenLearner</text>
    <text x="395" y="168" text-anchor="middle" font-size="10" fill="#555">Element-wise attn → 8/frame</text>

    <line x1="270" y1="152" x2="328" y2="152" stroke="#888" stroke-width="2" marker-end="url(#rtaen)"/>

    <!-- compressed -->
    <g transform="translate(480,125)">
      <text x="0" y="18" font-size="11" fill="#555">Compressed to 48 tokens</text>
      <text x="0" y="36" font-size="11" fill="#555">+ language + &lt;BOS&gt;</text>
      <text x="0" y="54" font-size="11" fill="#34A853">Inference speed ↑ 2.4×</text>
    </g>

    <line x1="460" y1="152" x2="478" y2="152" stroke="#888" stroke-width="2" marker-end="url(#rtaen)"/>

    <!-- equation bar -->
    <rect x="30" y="220" width="620" height="45" rx="6" fill="#f8f9fa" stroke="#dadce0"/>
    <text x="340" y="248" text-anchor="middle" font-size="14" fill="#1a1a1a" font-family="monospace">300×300 × 6 frames  →  9×9×512 × 6 = 486 tokens  →  TokenLearner  →  8×6 = 48 tokens</text>

    <text x="340" y="310" text-anchor="middle" font-size="12" fill="#1a73e8">6 frames = last ~2 seconds; TokenLearner adaptively keeps task-relevant regions.</text>

    <defs>
      <marker id="rtaen" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="#888"/>
      </marker>
    </defs>
  </svg>
  <p class="fig-cap">Figure: RT-1 first turns each frame into 81 tokens with EfficientNet, then compresses each frame to 8 tokens with TokenLearner.</p>
</div>

---

## 3. Action Side: Decoder-Only Transformer

After visual compression, the 48 visual tokens are concatenated with **language instruction tokens** and a special **`<BOS>`** start token, then fed into a decoder-only Transformer.

### 3.1 What Goes In and What Comes Out

**Input sequence** (left to right):

```text
<BOS>  [language instruction tokens]  [visual tokens × 48]
```

**Output sequence** (generated autoregressively):

```text
[mode token]  [arm action tokens ...]  [base action tokens ...]
```

RT-1 discretizes actions into three token types:

| Type | Meaning | Examples |
|---|---|---|
| **Mode** | End-effector open/close / motion mode | open / close / move |
| **Arm** | Discretized 6–7 DOF joint angles | each dimension binned |
| **Base** | Mobile base velocity / heading | forward / back / turn |

The model generates a string of action tokens, which are then decoded back into the robot's actual control command.

### 3.2 Standard Causal Mask: Look Left Only

The signature of a decoder-only Transformer is the **causal mask**:

> Position t can only attend to positions ≤ t; it cannot peek at the future.

This preserves autoregressive validity: when predicting the t-th action token, the model may only use positions 1 … t-1 plus the input-side history.

<div class="fig">
  <svg viewBox="0 0 680 160" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="340" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a1a1a">Standard Causal Mask: Position t Can Only See ≤ t</text>

    <g transform="translate(220,40)">
      <text x="-20" y="15" font-size="12" fill="#555">query</text>
      <text x="100" y="-8" text-anchor="middle" font-size="12" fill="#555">key</text>
      <rect x="0" y="0" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="24" y="0" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="48" y="0" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="72" y="0" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="96" y="0" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="120" y="0" width="24" height="24" fill="#dadce0" stroke="#fff"/>
      <rect x="0" y="24" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="24" y="24" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="48" y="24" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="72" y="24" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="96" y="24" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="120" y="24" width="24" height="24" fill="#dadce0" stroke="#fff"/>
      <rect x="0" y="48" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="24" y="48" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="48" y="48" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="72" y="48" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="96" y="48" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="120" y="48" width="24" height="24" fill="#dadce0" stroke="#fff"/>
      <rect x="0" y="72" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="24" y="72" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="48" y="72" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="72" y="72" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="96" y="72" width="24" height="24" fill="#dadce0" stroke="#fff"/><rect x="120" y="72" width="24" height="24" fill="#dadce0" stroke="#fff"/>
      <rect x="0" y="96" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="24" y="96" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="48" y="96" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="72" y="96" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="96" y="96" width="24" height="24" fill="#34A853" stroke="#fff"/><rect x="120" y="96" width="24" height="24" fill="#dadce0" stroke="#fff"/>

      <text x="12" y="-15" text-anchor="middle" font-size="10" fill="#555">1</text>
      <text x="36" y="-15" text-anchor="middle" font-size="10" fill="#555">2</text>
      <text x="60" y="-15" text-anchor="middle" font-size="10" fill="#555">3</text>
      <text x="84" y="-15" text-anchor="middle" font-size="10" fill="#555">4</text>
      <text x="108" y="-15" text-anchor="middle" font-size="10" fill="#555">5</text>
      <text x="132" y="-15" text-anchor="middle" font-size="10" fill="#555">6</text>

      <text x="-12" y="16" text-anchor="end" font-size="10" fill="#555">1</text>
      <text x="-12" y="40" text-anchor="end" font-size="10" fill="#555">2</text>
      <text x="-12" y="64" text-anchor="end" font-size="10" fill="#555">3</text>
      <text x="-12" y="88" text-anchor="end" font-size="10" fill="#555">4</text>
      <text x="-12" y="112" text-anchor="end" font-size="10" fill="#555">5</text>
    </g>

    <g transform="translate(450,60)">
      <rect x="0" y="0" width="16" height="16" fill="#34A853"/>
      <text x="24" y="13" font-size="12" fill="#555">Visible</text>
      <rect x="0" y="28" width="16" height="16" fill="#dadce0"/>
      <text x="24" y="41" font-size="12" fill="#555">Masked</text>
    </g>

    <text x="340" y="150" text-anchor="middle" font-size="12" fill="#1a73e8">Lower-triangle green: predicting token t can only use positions 1…t.</text>
  </svg>
  <p class="fig-cap">Figure: A standard causal mask is a lower-triangle matrix; it prevents the model from peeking at future tokens.</p>
</div>

---

## 4. Common Forms of Causal Mask

In practice, "look left only" comes in several flavors.

### 4.1 Strict Causal Mask

The most common. Position t can only attend to positions ≤ t. Both training and inference use this lower-triangle pattern.

### 4.2 Mask with a Start Token

If the sequence begins with a special `<BOS>` or `<PAD>`, you sometimes want **every position to attend to that start token** (it carries no future information, just global context), while keeping everything else strictly causal. In matrix form, the first column is all green and the rest is lower-triangular.

### 4.3 Block Causal Mask

Actions are often generated in groups: first a "mode", then a group of "arm joints", then "base". A block causal mask allows:

- Strict causality inside each block;
- Current block to attend to all previous blocks;
- No peeking at later blocks.

This fits RT-1-style multi-field action tokens naturally: base tokens can see mode and arm tokens, but arm tokens cannot see base tokens.

### 4.4 KV-Cache Mask

At inference time, Keys and Values of already-generated tokens are cached. The mask then looks like:

- Historical tokens (cached): marked with `*`, not recomputed;
- Current new token: attends only to cache + its own row;
- Future tokens: still masked.

<div class="fig">
  <svg viewBox="0 0 680 210" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="340" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a1a1a">Four Common Causal-Mask Patterns</text>

    <g transform="translate(30,40)">
      <text x="60" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">1. Strict Causal</text>
      <g transform="translate(20,10)">
        <rect x="0" y="0" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="28" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="42" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="14" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="14" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="42" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="28" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="28" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="28" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="42" y="28" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="28" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="42" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="56" y="42" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="42" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="56" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/>
      </g>
      <text x="60" y="90" text-anchor="middle" font-size="10" fill="#555">Most common</text>
    </g>

    <g transform="translate(180,40)">
      <text x="60" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">2. With &lt;BOS&gt;</text>
      <g transform="translate(20,10)">
        <rect x="0" y="0" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="28" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="42" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="14" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="14" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="42" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="28" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="28" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="28" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="42" y="28" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="28" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="42" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="56" y="42" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="42" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="56" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/>
      </g>
      <text x="60" y="90" text-anchor="middle" font-size="10" fill="#555">Col 1 all visible</text>
    </g>

    <g transform="translate(330,40)">
      <text x="60" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">3. Block Causal</text>
      <g transform="translate(20,10)">
        <rect x="0" y="0" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="28" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="42" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="14" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="14" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="42" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="28" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="28" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="28" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="42" y="28" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="28" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="42" y="42" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="56" y="42" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="14" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="28" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="42" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/><rect x="56" y="56" width="14" height="14" fill="#34A853" stroke="#fff"/>
      </g>
      <text x="60" y="90" text-anchor="middle" font-size="10" fill="#555">Across-block visible</text>
    </g>

    <g transform="translate(480,40)">
      <text x="60" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">4. KV-Cache</text>
      <g transform="translate(20,10)">
        <rect x="0" y="0" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="14" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="28" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="42" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="0" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="14" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="14" y="14" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="28" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="42" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="14" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="28" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="14" y="28" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="28" y="28" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="42" y="28" width="14" height="14" fill="#dadce0" stroke="#fff"/><rect x="56" y="28" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="42" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="14" y="42" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="28" y="42" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="42" y="42" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="56" y="42" width="14" height="14" fill="#dadce0" stroke="#fff"/>
        <rect x="0" y="56" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="14" y="56" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="28" y="56" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="42" y="56" width="14" height="14" fill="#FBBC04" stroke="#fff"/><rect x="56" y="56" width="14" height="14" fill="#FBBC04" stroke="#fff"/>
      </g>
      <text x="60" y="90" text-anchor="middle" font-size="10" fill="#555">Yellow = cached</text>
    </g>

    <text x="340" y="195" text-anchor="middle" font-size="12" fill="#1a73e8">Strict causal is default; BOS/block suit multi-field actions; KV-cache is an inference optimization.</text>
  </svg>
  <p class="fig-cap">Figure: Four causal-mask variants. Green = visible, gray = masked, yellow = cached.</p>
</div>

---

## 5. Categorical Cross-Entropy: Standard and Variants

Because action tokens are discrete categories, the training objective is **categorical cross-entropy**:

```text
L = - Σ  y_t,v · log( p_hat_t,v )
```

Here `y_t` is the one-hot ground-truth action token and `p_hat_t` is the predicted probability distribution. Intuitively, the loss only cares about whether the probability mass on the true category is high.

### 5.1 Standard Cross-Entropy

The ground-truth label is one-hot: 1 at the true position and 0 elsewhere. Loss = -log(probability at the true position). This is the RT-1 baseline.

### 5.2 Label Smoothing

Standard cross-entropy can push the model to be over-confident and overfit. Label smoothing changes the one-hot target to:

```text
true position = 1 - ε
all others    = ε / (V - 1)
```

The model no longer has to push the true class probability all the way to 1; it is allowed to leave a small probability for every other class. This usually improves generalization.

### 5.3 Weighted Cross-Entropy

Different action classes can be imbalanced — for example, "base motion" tokens may appear far less often than "arm" tokens. Weighted cross-entropy multiplies the loss of rare classes by a larger weight:

```text
L_t = - w_y_t · log( p_hat_t, y_t )
```

This is very practical in robotics data, preventing the model from learning only the most common "safe" actions.

<div class="fig">
  <svg viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">
    <text x="340" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a1a1a">Three Cross-Entropy Variants</text>

    <g transform="translate(40,45)">
      <text x="80" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">1. Standard</text>
      <g transform="translate(0,15)">
        <rect x="0" y="40" width="20" height="10" fill="#4285F4"/>
        <rect x="25" y="0" width="20" height="50" fill="#EA4335"/>
        <rect x="50" y="45" width="20" height="5" fill="#4285F4"/>
        <rect x="75" y="42" width="20" height="8" fill="#4285F4"/>
        <rect x="100" y="44" width="20" height="6" fill="#4285F4"/>
        <rect x="125" y="41" width="20" height="9" fill="#4285F4"/>
        <line x1="0" y1="50" x2="150" y2="50" stroke="#333" stroke-width="1"/>
        <line x1="0" y1="50" x2="0" y2="0" stroke="#333" stroke-width="1"/>
      </g>
      <text x="80" y="95" text-anchor="middle" font-size="10" fill="#555">True class probability = 1</text>
    </g>

    <g transform="translate(250,45)">
      <text x="80" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">2. Label Smoothing</text>
      <g transform="translate(0,15)">
        <rect x="0" y="35" width="20" height="15" fill="#4285F4"/>
        <rect x="25" y="5" width="20" height="45" fill="#EA4335"/>
        <rect x="50" y="38" width="20" height="12" fill="#4285F4"/>
        <rect x="75" y="36" width="20" height="14" fill="#4285F4"/>
        <rect x="100" y="37" width="20" height="13" fill="#4285F4"/>
        <rect x="125" y="35" width="20" height="15" fill="#4285F4"/>
        <line x1="0" y1="50" x2="150" y2="50" stroke="#333" stroke-width="1"/>
        <line x1="0" y1="50" x2="0" y2="0" stroke="#333" stroke-width="1"/>
      </g>
      <text x="80" y="95" text-anchor="middle" font-size="10" fill="#555">True = 1-ε, others split ε</text>
    </g>

    <g transform="translate(460,45)">
      <text x="80" y="0" text-anchor="middle" font-size="11" fill="#1a1a1a" font-weight="bold">3. Weighted</text>
      <g transform="translate(0,15)">
        <rect x="0" y="42" width="20" height="8" fill="#4285F4"/>
        <rect x="25" y="2" width="20" height="48" fill="#EA4335"/>
        <rect x="50" y="10" width="20" height="40" fill="#FBBC04"/>
        <rect x="75" y="8" width="20" height="42" fill="#FBBC04"/>
        <rect x="100" y="12" width="20" height="38" fill="#FBBC04"/>
        <rect x="125" y="9" width="20" height="41" fill="#FBBC04"/>
        <line x1="0" y1="50" x2="150" y2="50" stroke="#333" stroke-width="1"/>
        <line x1="0" y1="50" x2="0" y2="0" stroke="#333" stroke-width="1"/>
      </g>
      <text x="80" y="95" text-anchor="middle" font-size="10" fill="#555">Classes use weights w</text>
    </g>

    <rect x="30" y="155" width="620" height="45" rx="6" fill="#f8f9fa" stroke="#dadce0"/>
    <text x="340" y="173" text-anchor="middle" font-size="13" fill="#1a1a1a">Standard: L = -log p̂ₜ,ᵧₜ</text>
    <text x="340" y="190" text-anchor="middle" font-size="13" fill="#1a1a1a">Weighted: L = -wᵧₜ · log p̂ₜ,ᵧₜ ｜ Smoothed: true = 1-ε, others = ε/(V-1)</text>
  </svg>
  <p class="fig-cap">Figure: Three cross-entropy variants. Standard is hardest; label smoothing prevents overfitting; weighted handles class imbalance.</p>
</div>

---

## 6. From RT-1 to RT-2 to Native VLA

RT-1's importance is not "state-of-the-art performance" but **proof that VLA can be engineered**:

- Turn images into tokens;
- Turn actions into tokens;
- Train a decoder-only Transformer with causal masks and cross-entropy end-to-end.

RT-2 later scaled this by attaching a vision-language model (VLM) directly to action outputs, letting robots leverage internet-scale vision-language knowledge. Then π0, OpenVLA, and Xiaomi pushed toward **native VLA + continuous generation**, improving precision and smoothness.

> So RT-1 is not the destination; it is the starting point of "reframing control as language." Understanding TokenLearner, causal masks, and cross-entropy means understanding the shared DNA of nearly every VLA that followed.

> Investment angle: the embodied-AI model stack is converging quickly. The discrete route (RT-1/RT-2/OpenVLA) and the continuous route (π0/Xiaomi) are not mutually exclusive — they will coexist across different latency/accuracy/cost scenarios. Watch teams that can run both routes and have a real data flywheel.
