---
title: "MTP Heads and Confidence Heads: Linking Three Self-Drafting Routes into One Chain"
description: Starting from the multi-token prediction head, this post connects the three self-drafting routes EAGLE-3, DFlash, and DSpark, and dissects DSpark's parallel backbone plus Markov Head plus Confidence Head and its prefix survival probability scheduling.
pubDate: 2026-08-19
series: Speculative Decoding Notes
lang: en
altLang: zh
altHref: /blog/ep7-mtp-dspark
layout: ../../../layouts/BlogPost.astro
---

> The main thread in one line: **MTP teaches the model during training to predict several tokens in one forward pass; EAGLE-3 guesses autoregressively with a specially trained Draft Head; DFlash guesses a whole block in parallel in one shot; DSpark = DFlash parallel backbone + lightweight Markov Head + Confidence Head + load-aware scheduling.** All three are "self-drafting, no separate small LM," they just differ in draft topology.

First, correcting the most easily confused point: **DSpark is not MTP, and it is not simply "one draft model generating multiple tokens autoregressively."** The drafting core of DSpark is "DFlash's block-parallel backbone + a lightweight Markov Head," plus a Confidence Head — that is, "parallel generation + lightweight sequential dependency + confidence scheduling." Both the official paper and the vLLM Speculators documentation describe it this way. The rest of this post takes the whole chain apart.

## 1. What Exactly Is an MTP Head

MTP = **Multi-Token Prediction**. An ordinary LM Head only produces the next token:

```text
A B C D -> Transformer -> LM Head -> P(E | ABCD) -> E
```

MTP attaches several extra prediction heads / MTP modules to the original model, so one forward yields a string of future tokens:

```text
A B C D -> Transformer
              |- LM Head      -> E
              |- MTP-1 Head   -> F
              |- MTP-2 Head   -> G
              |- MTP-3 Head   -> H
```

So one forward gives `E F G H`. The core idea is: **explicitly train the model during training to predict multiple future tokens** (the vLLM Speculators definition of MTP is also "finetune the model's native multi-token prediction head").

<div class="fig">
<svg viewBox="0 0 680 330" width="100%" role="img" font-family="-apple-system,PingFang SC,Microsoft YaHei,sans-serif">
  <title>MTP Head structure (DeepSeek-V3 multi-token prediction module)</title>
  <desc>Main Transformer hidden state plus the token embedding shifted one step forward, passed through an MTP Transformer Block, RMSNorm, and a shared LM Head to predict one extra future token; D modules can be stacked for multi-token supervision.</desc>
  <defs>
    <marker id="arrowEn1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="20" y="22" font-size="15" font-weight="500" fill="#173404">MTP Head structure (DeepSeek-V3 multi-token prediction module)</text>

  <rect x="24" y="54" width="150" height="58" rx="8" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="99" y="80" font-size="14" font-weight="500" fill="#27500A" text-anchor="middle">Main Transformer</text>
  <text x="99" y="98" font-size="12" fill="#3B6D11" text-anchor="middle">shared vocab / LM Head</text>

  <line x1="174" y1="83" x2="256" y2="107" stroke="#185FA5" stroke-width="1.5" marker-end="url(#arrowEn1)"/>
  <text x="190" y="78" font-size="12" fill="#0C447C">h_t</text>
  <text x="190" y="94" font-size="11" fill="#5F5E5A">hidden at pos t</text>

  <rect x="96" y="150" width="92" height="44" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="142" y="172" font-size="13" font-weight="500" fill="#633806" text-anchor="middle">Emb(t+1)</text>
  <text x="142" y="188" font-size="11" fill="#854F0B" text-anchor="middle">token shifted by one</text>
  <line x1="188" y1="172" x2="256" y2="172" stroke="#BA7517" stroke-width="1.5" marker-end="url(#arrowEn1)"/>

  <rect x="236" y="50" width="250" height="210" rx="10" fill="#F4FAEF" stroke="#639922" stroke-width="1"/>
  <text x="361" y="72" font-size="14" font-weight="500" fill="#27500A" text-anchor="middle">MTP module (d=1)</text>

  <rect x="256" y="86" width="210" height="44" rx="6" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="361" y="113" font-size="13" font-weight="500" fill="#27500A" text-anchor="middle">MTP Transformer Block</text>

  <line x1="361" y1="130" x2="361" y2="146" stroke="#888780" stroke-width="1.2" marker-end="url(#arrowEn1)"/>
  <rect x="256" y="146" width="210" height="34" rx="6" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="361" y="167" font-size="13" fill="#27500A" text-anchor="middle">RMSNorm</text>

  <line x1="361" y1="180" x2="361" y2="196" stroke="#888780" stroke-width="1.2" marker-end="url(#arrowEn1)"/>
  <rect x="256" y="196" width="210" height="34" rx="6" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="361" y="217" font-size="13" fill="#27500A" text-anchor="middle">LM Head (shared)</text>

  <line x1="486" y1="213" x2="548" y2="213" stroke="#639922" stroke-width="1.5" marker-end="url(#arrowEn1)"/>
  <text x="552" y="209" font-size="12" fill="#3B6D11">predict</text>
  <text x="552" y="228" font-size="15" font-weight="500" fill="#173404">t+2</text>

  <text x="236" y="284" font-size="12" fill="#444441">D MTP modules can be stacked: MTP1 to t+2, MTP2 to t+3, ..., MTP-D to t+D+1 (multi-token supervision in training)</text>
  <text x="236" y="304" font-size="12" fill="#444441">At inference: MTP1 reuses the main model hidden state and predicts 1 extra token as a speculative draft head (near zero cost)</text>
</svg>
</div>

> Note: an MTP Head is not "four completely independent Linear layers." Modern MTP lets later prediction positions use earlier hidden/token information, and the exact MTP module structure differs across models.

## 2. Why MTP Can Be Used for Speculative Decoding

After MTP predicts `E F G H`, the target / verifier can **verify the whole chain in parallel**:

```text
A B C D -> E accepted
             F accepted
             G accepted
             H rejected
```

The final result accepts `E F G`, discards `H`, and lets the target correct after `G`. So **MTP is itself a ready-made speculative decoding drafter** — this is where the common DeepSeek pattern of "Target Model with MTP modules attached, verified in parallel" comes from.

## 3. How Is EAGLE-3 Different from MTP

This is the most important layer. EAGLE-3 is a **specifically trained Draft Head / Speculator**, not a few MTP Heads bolted onto the original model:

```text
Target Model -> hidden states -> Eagle3 Draft Head -> autoregressively generate t1->t2->t3->t4 -> Target Verify
```

EAGLE-3's draft tokens have **clear sequential dependency** (the vLLM docs describe it as "autoregressively predict draft tokens using Llama-style draft layers"). MTP, by contrast, is native to the model, a multi-token head embedded during training. Both can serve as drafters, but their origin and draft topology differ.

## 4. What Is DFlash Then

DFlash goes in a completely different direction — **one forward pass predicts an entire block in parallel**:

```text
Eagle3 (serial):   DFlash (parallel):
t1                Input |- t1
 |                      |- t2
t2                      |- t3
 |                      |- t4
t3                      |- t5
 |                      |- t6
t4
```

DFlash's advantage is that it is **very fast**; the cost is insufficient dependency between tokens, so the acceptance rate drops the further back you go — this is **acceptance decay / suffix decay** (the DSpark paper states explicitly that the main weakness of a pure parallel drafter is insufficient in-block token dependency).

## 5. How Does DSpark Solve It

The body of DSpark is still DFlash's parallel generation, but with a **lightweight Markov Head** on top so that the k-th token is aware of the previous token:

```text
Anchor -> DFlash Parallel Backbone -> t1 t2 t3 ... tN
                                        |
                                   Markov Head
                                        |
                            adds local token-to-token dependency
```

That is, "**parallel backbone + very light local sequential dependency**," which is where the paper title **Semi-Autoregressive Generation** comes from.

## 6. Why the Markov Head Helps

With pure parallelism, each `t_k` only depends on the context; after adding the Markov Head, `t_k` also depends on the previous token `t_{k-1}`. In the official implementation the Markov Head is a **low-rank logit bias**:

```text
B = W1 @ W2      (default rank = 256)
```

It adds a bias to the current draft logits based on the previous token. The cost is almost zero, yet it patches the weakness of a parallel draft tail being incoherent and easily rejected.

## 7. Only Then Comes the Confidence Head

The Confidence Head is a lightweight head:

```text
hidden state -> Linear -> sigmoid -> c_k
```

where:

```text
c_k = P(token k is accepted | all previous tokens are accepted)
```

This is very important — it is not "how probable this token is by itself," but "**if everything before is correct, what is the conditional probability that this token is also correct**."

## 8. Is "High Confidence" the Same as "High Probability"

**Yes, but it must be stated precisely**: "high confidence" here means the **acceptance probability predicted by the Confidence Head is high**. For example `c1=0.95, c2=0.90, c3=0.85, c4=0.40` corresponds to

```text
P(t1 accepted | previous correct)
P(t2 accepted | t1 correct)
P(t3 accepted | t1,t2 correct)
P(t4 accepted | t1,t2,t3 correct)
```

and not to `P(t1), P(t2), P(t3), P(t4)`, the tokens' own generation probabilities. So the strict name is **conditional acceptance probability**.

## 9. Why It Is Called Prefix Survival Probability

What we really care about is "can this whole prefix survive." Multiply the conditional probabilities:

```text
Prefix 1: 0.95
Prefix 2: 0.95 x 0.90 = 0.855
Prefix 3: 0.95 x 0.90 x 0.85 = 0.727
Prefix 4: 0.95 x 0.90 x 0.85 x 0.40 = 0.291
```

The definition given in the DSpark paper is `a_{r,j} = prod_{i=1}^{j} c_{r,i}` — the prefix survival probability is the product of the conditional probabilities at each position.

<div class="fig">
<svg viewBox="0 0 680 400" width="100%" role="img" font-family="-apple-system,PingFang SC,Microsoft YaHei,sans-serif">
  <title>Confidence Head predicts prefix survival probability</title>
  <desc>The draft backbone produces gamma hidden states in one forward pass; each position goes through a Confidence Head to output a conditional acceptance probability c_k; the prefix survival probability a_j is the product of c_1 to c_j, monotonically decreasing; the scheduler greedily truncates the tail in descending order of a_j.</desc>
  <defs>
    <marker id="arrowEn2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="20" y="22" font-size="15" font-weight="500" fill="#412402">Confidence Head to Prefix Survival Probability (core of DSpark verification scheduling)</text>
  <text x="380" y="22" font-size="11" fill="#3B6D11">i.e. "Confidence Head predicts prefix survival probability"</text>

  <rect x="24" y="52" width="84" height="42" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
  <text x="66" y="77" font-size="13" font-weight="500" fill="#0C447C" text-anchor="middle">Anchor D</text>

  <line x1="108" y1="73" x2="128" y2="73" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arrowEn2)"/>
  <rect x="128" y="52" width="172" height="42" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
  <text x="214" y="73" font-size="13" font-weight="500" fill="#26215C" text-anchor="middle">Draft Backbone</text>
  <text x="214" y="90" font-size="11" fill="#3C3489" text-anchor="middle">1 forward, gamma positions</text>

  <line x1="300" y1="94" x2="300" y2="108" stroke="#888780" stroke-width="1.2"/>
  <line x1="88" y1="108" x2="412" y2="108" stroke="#888780" stroke-width="1"/>
  <line x1="88" y1="108" x2="88" y2="118" stroke="#888780" stroke-width="1"/>
  <line x1="168" y1="108" x2="168" y2="118" stroke="#888780" stroke-width="1"/>
  <line x1="248" y1="108" x2="248" y2="118" stroke="#888780" stroke-width="1"/>
  <line x1="328" y1="108" x2="328" y2="118" stroke="#888780" stroke-width="1"/>
  <line x1="412" y1="108" x2="412" y2="118" stroke="#888780" stroke-width="1"/>

  <g>
    <rect x="62" y="118" width="52" height="38" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="88" y="137" font-size="12" fill="#0C447C" text-anchor="middle">h_1</text>
    <rect x="142" y="118" width="52" height="38" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="168" y="137" font-size="12" fill="#0C447C" text-anchor="middle">h_2</text>
    <rect x="222" y="118" width="52" height="38" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="248" y="137" font-size="12" fill="#0C447C" text-anchor="middle">h_3</text>
    <rect x="302" y="118" width="52" height="38" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="328" y="137" font-size="12" fill="#0C447C" text-anchor="middle">h_4</text>
    <rect x="386" y="118" width="52" height="38" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="412" y="137" font-size="12" fill="#0C447C" text-anchor="middle">h_5</text>
  </g>

  <g>
    <line x1="88" y1="156" x2="88" y2="176" stroke="#BA7517" stroke-width="1.2" marker-end="url(#arrowEn2)"/>
    <line x1="168" y1="156" x2="168" y2="176" stroke="#BA7517" stroke-width="1.2" marker-end="url(#arrowEn2)"/>
    <line x1="248" y1="156" x2="248" y2="176" stroke="#BA7517" stroke-width="1.2" marker-end="url(#arrowEn2)"/>
    <line x1="328" y1="156" x2="328" y2="176" stroke="#BA7517" stroke-width="1.2" marker-end="url(#arrowEn2)"/>
    <line x1="412" y1="156" x2="412" y2="176" stroke="#BA7517" stroke-width="1.2" marker-end="url(#arrowEn2)"/>
  </g>

  <g>
    <rect x="62" y="176" width="52" height="42" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="88" y="196" font-size="12" fill="#633806" text-anchor="middle">Conf</text>
    <text x="88" y="211" font-size="11" fill="#854F0B" text-anchor="middle">Head</text>
    <rect x="142" y="176" width="52" height="42" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="168" y="196" font-size="12" fill="#633806" text-anchor="middle">Conf</text>
    <text x="168" y="211" font-size="11" fill="#854F0B" text-anchor="middle">Head</text>
    <rect x="222" y="176" width="52" height="42" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="248" y="196" font-size="12" fill="#633806" text-anchor="middle">Conf</text>
    <text x="248" y="211" font-size="11" fill="#854F0B" text-anchor="middle">Head</text>
    <rect x="302" y="176" width="52" height="42" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="328" y="196" font-size="12" fill="#633806" text-anchor="middle">Conf</text>
    <text x="328" y="211" font-size="11" fill="#854F0B" text-anchor="middle">Head</text>
    <rect x="386" y="176" width="52" height="42" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="412" y="196" font-size="12" fill="#633806" text-anchor="middle">Conf</text>
    <text x="412" y="211" font-size="11" fill="#854F0B" text-anchor="middle">Head</text>
  </g>

  <g>
    <text x="88" y="246" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">c_1</text>
    <text x="168" y="246" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">c_2</text>
    <text x="248" y="246" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">c_3</text>
    <text x="328" y="246" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">c_4</text>
    <text x="412" y="246" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">c_5</text>
  </g>

  <text x="20" y="276" font-size="12" fill="#444441">c_k = P(position k accepted | first k-1 all accepted) - conditional acceptance probability, supervised by the TV distance between draft and target distributions</text>

  <g>
    <rect x="70" y="270" width="40" height="60" fill="#FAC775" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="150" y="282" width="40" height="48" fill="#FAC775" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="230" y="294" width="40" height="36" fill="#FAC775" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="310" y="306" width="40" height="24" fill="#FAC775" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="390" y="316" width="40" height="14" fill="#FAC775" stroke="#BA7517" stroke-width="0.5"/>
  </g>
  <text x="90" y="348" font-size="11" fill="#633806" text-anchor="middle">a_1</text>
  <text x="170" y="348" font-size="11" fill="#633806" text-anchor="middle">a_2</text>
  <text x="250" y="348" font-size="11" fill="#633806" text-anchor="middle">a_3</text>
  <text x="330" y="348" font-size="11" fill="#633806" text-anchor="middle">a_4</text>
  <text x="410" y="348" font-size="11" fill="#633806" text-anchor="middle">a_5</text>

  <text x="20" y="372" font-size="12" fill="#444441">prefix survival prob: a_j = c_1 c_2 ... c_j (probability the first j are all accepted, monotonically decreasing in j)</text>
  <text x="20" y="392" font-size="12" fill="#444441">Scheduler: greedily add candidates in descending a_j, stop early when expected accepted tokens saturate, truncating the low-confidence tail</text>
</svg>
</div>

## 10. The Complete DSpark Structure

Putting it all together, the end-to-end flow is: the target first generates the anchor, then DSpark produces `E F G H` plus confidence in parallel, the scheduler truncates it to a prefix, and the target verifies in parallel.

<div class="fig">
<svg viewBox="0 0 680 380" width="100%" role="img" font-family="-apple-system,PingFang SC,Microsoft YaHei,sans-serif">
  <title>DSpark end-to-end pipeline</title>
  <desc>The Target generates the Anchor, the DFlash parallel backbone emits a block in one pass, the Markov Head adds local dependency, the Confidence Head outputs c_k, the scheduler decides the verification length by prefix survival probability, and control returns to the Target for parallel verification.</desc>
  <defs>
    <marker id="arrowEn3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="20" y="22" font-size="15" font-weight="500" fill="#26215C">DSpark end-to-end pipeline</text>

  <rect x="250" y="30" width="180" height="40" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
  <text x="340" y="54" font-size="13" font-weight="500" fill="#0C447C" text-anchor="middle">Target Model (generates Anchor)</text>

  <line x1="340" y1="70" x2="340" y2="80" stroke="#888780" stroke-width="1.2" marker-end="url(#arrowEn3)"/>
  <rect x="300" y="80" width="80" height="34" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
  <text x="340" y="101" font-size="13" font-weight="500" fill="#0C447C" text-anchor="middle">Anchor D</text>

  <line x1="340" y1="114" x2="340" y2="126" stroke="#888780" stroke-width="1.2" marker-end="url(#arrowEn3)"/>
  <rect x="225" y="126" width="230" height="44" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
  <text x="340" y="148" font-size="13" font-weight="500" fill="#26215C" text-anchor="middle">DFlash Parallel Backbone</text>
  <text x="340" y="166" font-size="11" fill="#3C3489" text-anchor="middle">1 forward, block parallel</text>

  <line x1="340" y1="170" x2="340" y2="184" stroke="#888780" stroke-width="1"/>
  <line x1="138" y1="184" x2="458" y2="184" stroke="#888780" stroke-width="1"/>
  <line x1="138" y1="184" x2="138" y2="190" stroke="#888780" stroke-width="1"/>
  <line x1="218" y1="184" x2="218" y2="190" stroke="#888780" stroke-width="1"/>
  <line x1="298" y1="184" x2="298" y2="190" stroke="#888780" stroke-width="1"/>
  <line x1="378" y1="184" x2="378" y2="190" stroke="#888780" stroke-width="1"/>
  <line x1="458" y1="184" x2="458" y2="190" stroke="#888780" stroke-width="1"/>
  <g>
    <rect x="110" y="190" width="56" height="34" rx="5" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
    <text x="138" y="211" font-size="12" fill="#26215C" text-anchor="middle">t_1</text>
    <rect x="190" y="190" width="56" height="34" rx="5" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
    <text x="218" y="211" font-size="12" fill="#26215C" text-anchor="middle">t_2</text>
    <rect x="270" y="190" width="56" height="34" rx="5" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
    <text x="298" y="211" font-size="12" fill="#26215C" text-anchor="middle">t_3</text>
    <rect x="350" y="190" width="56" height="34" rx="5" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
    <text x="378" y="211" font-size="12" fill="#26215C" text-anchor="middle">t_4</text>
    <rect x="430" y="190" width="56" height="34" rx="5" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
    <text x="458" y="211" font-size="12" fill="#26215C" text-anchor="middle">t_N</text>
  </g>

  <line x1="138" y1="224" x2="138" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="218" y1="224" x2="218" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="298" y1="224" x2="298" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="378" y1="224" x2="378" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="458" y1="224" x2="458" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="138" y1="236" x2="458" y2="236" stroke="#888780" stroke-width="1"/>
  <line x1="340" y1="236" x2="340" y2="248" stroke="#888780" stroke-width="1"/>

  <rect x="225" y="248" width="230" height="38" rx="8" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="340" y="268" font-size="13" font-weight="500" fill="#27500A" text-anchor="middle">Markov Head</text>
  <text x="340" y="284" font-size="11" fill="#3B6D11" text-anchor="middle">prev token to logit bias (B=W1W2, r=256)</text>

  <line x1="340" y1="286" x2="340" y2="298" stroke="#888780" stroke-width="1.2" marker-end="url(#arrowEn3)"/>
  <rect x="225" y="298" width="230" height="34" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="340" y="319" font-size="13" font-weight="500" fill="#633806" text-anchor="middle">Confidence Head to c_k (sigmoid)</text>

  <line x1="340" y1="332" x2="340" y2="344" stroke="#888780" stroke-width="1.2" marker-end="url(#arrowEn3)"/>
  <rect x="225" y="344" width="230" height="32" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="340" y="364" font-size="12" font-weight="500" fill="#633806" text-anchor="middle">Hardware-aware Scheduler (truncate by a_j)</text>

  <path d="M455 360 C 580 360 580 50 430 50" fill="none" stroke="#BA7517" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#arrowEn3)"/>
  <text x="470" y="210" font-size="11" fill="#854F0B">back to Target, parallel verify</text>
  <text x="470" y="226" font-size="11" fill="#854F0B">accept / reject</text>
</svg>
</div>

## 11. How DSpark Relates to EAGLE-3 / DFlash / MTP

Put the four in one picture: **EAGLE-3 = shallow autoregressive drafter; DFlash = parallel drafter; MTP = native multi-token head; DSpark = DFlash + Markov Head + Confidence Head + Hardware-aware Scheduler.** So the relationship between DSpark and EAGLE-3 is **not "DSpark is an improved EAGLE-3,"** but "a sequential head and confidence scheduling stacked on top of a parallel draft (DFlash)." The official Speculators docs explicitly define DSpark as "extends DFlash with Markov head + confidence head."

<div class="fig">
<svg viewBox="0 0 680 410" width="100%" role="img" font-family="-apple-system,PingFang SC,Microsoft YaHei,sans-serif">
  <title>DSpark, EAGLE-3 tree, and MTP Head: three self-drafting paradigms compared</title>
  <desc>Left: DSpark parallel block plus sequential head produces a flat block; middle: EAGLE-3 feature-level autoregression produces a candidate tree; right: MTP attaches multiple heads to the main model to predict offset tokens. All three are self-drafting with no separate small LM.</desc>
  <defs>
    <marker id="arrowEn4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="20" y="22" font-size="15" font-weight="500" fill="#26215C">DSpark / EAGLE-3 tree / MTP Head: three self-drafting paradigms compared</text>

  <rect x="30" y="40" width="190" height="30" rx="6" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
  <text x="125" y="60" font-size="14" font-weight="500" fill="#27500A" text-anchor="middle">DSpark</text>
  <rect x="250" y="40" width="190" height="30" rx="6" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
  <text x="345" y="60" font-size="14" font-weight="500" fill="#0C447C" text-anchor="middle">EAGLE-3 (tree)</text>
  <rect x="470" y="40" width="190" height="30" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="565" y="60" font-size="14" font-weight="500" fill="#633806" text-anchor="middle">MTP Head</text>

  <g>
    <rect x="40" y="92" width="38" height="34" rx="5" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
    <text x="59" y="113" font-size="12" fill="#27500A" text-anchor="middle">E</text>
    <rect x="86" y="92" width="38" height="34" rx="5" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
    <text x="105" y="113" font-size="12" fill="#27500A" text-anchor="middle">F</text>
    <rect x="132" y="92" width="38" height="34" rx="5" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
    <text x="151" y="113" font-size="12" fill="#27500A" text-anchor="middle">G</text>
    <rect x="178" y="92" width="38" height="34" rx="5" fill="#EAF3DE" stroke="#639922" stroke-width="0.5"/>
    <text x="197" y="113" font-size="12" fill="#27500A" text-anchor="middle">H</text>
    <line x1="78" y1="109" x2="86" y2="109" stroke="#639922" stroke-width="1.4" marker-end="url(#arrowEn4)"/>
    <line x1="124" y1="109" x2="132" y2="109" stroke="#639922" stroke-width="1.4" marker-end="url(#arrowEn4)"/>
    <line x1="170" y1="109" x2="178" y2="109" stroke="#639922" stroke-width="1.4" marker-end="url(#arrowEn4)"/>
    <text x="125" y="152" font-size="12" fill="#3B6D11" text-anchor="middle">gamma flat tokens (block)</text>
  </g>

  <g>
    <rect x="335" y="88" width="40" height="30" rx="5" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="355" y="107" font-size="12" fill="#0C447C" text-anchor="middle">root</text>
    <rect x="295" y="134" width="36" height="28" rx="5" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="313" y="151" font-size="11" fill="#0C447C" text-anchor="middle">a</text>
    <rect x="375" y="134" width="36" height="28" rx="5" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
    <text x="393" y="151" font-size="11" fill="#0C447C" text-anchor="middle">b</text>
    <line x1="355" y1="118" x2="313" y2="134" stroke="#185FA5" stroke-width="1.4" marker-end="url(#arrowEn4)"/>
    <line x1="355" y1="118" x2="393" y2="134" stroke="#185FA5" stroke-width="1.4" marker-end="url(#arrowEn4)"/>
    <text x="345" y="186" font-size="12" fill="#0C447C" text-anchor="middle">a candidate tree (tree attention verify)</text>
  </g>

  <g>
    <rect x="495" y="88" width="46" height="26" rx="5" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="518" y="105" font-size="11" fill="#633806" text-anchor="middle">head1</text>
    <rect x="495" y="120" width="46" height="26" rx="5" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="518" y="137" font-size="11" fill="#633806" text-anchor="middle">head2</text>
    <rect x="495" y="152" width="46" height="26" rx="5" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="518" y="169" font-size="11" fill="#633806" text-anchor="middle">headD</text>
    <line x1="541" y1="127" x2="580" y2="127" stroke="#BA7517" stroke-width="1.4" marker-end="url(#arrowEn4)"/>
    <rect x="582" y="100" width="30" height="16" rx="3" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="582" y="124" width="30" height="16" rx="3" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <rect x="582" y="148" width="30" height="16" rx="3" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
    <text x="565" y="190" font-size="12" fill="#854F0B" text-anchor="middle">D offset heads, flat output</text>
  </g>

  <g font-size="12" fill="#444441">
    <text x="30" y="222">- gamma positions in one forward</text>
    <text x="30" y="244">- flat block (not a tree)</text>
    <text x="30" y="266">- independently trained light drafter</text>
    <text x="30" y="288">- shares embedding/LM head</text>
    <text x="30" y="310">- includes confidence-head scheduling</text>

    <text x="250" y="222">- autoregressive per-position feature prediction</text>
    <text x="250" y="244">- dynamic draft tree plus tree attention</text>
    <text x="250" y="266">- multi-layer fusion plus Training-Time Test</text>
    <text x="250" y="288">- up to 6.5x speedup</text>
    <text x="250" y="310">- founder of the self-drafting school</text>

    <text x="470" y="222">- D heads attached to the main model (pretrained)</text>
    <text x="470" y="244">- each head predicts offset d+1</text>
    <text x="470" y="266">- used as draft head at inference (zero cost)</text>
    <text x="470" y="288">- reuses main model weights</text>
    <text x="470" y="310">- replaced the V4 MTP-1 baseline</text>
  </g>

  <line x1="30" y1="338" x2="650" y2="338" stroke="#888780" stroke-width="0.5"/>
  <text x="20" y="362" font-size="12" fill="#444441">All three are "self-drafting, no separate small LM." DSpark adds a sequential head plus confidence scheduling on a parallel draft (DFlash), improving accepted length by 26-31% over EAGLE-3</text>
  <text x="20" y="382" font-size="12" fill="#444441">and 16-18% over DFlash, and replaces the V4 MTP-1 baseline. EAGLE-3 is a tree, MTP is offset multi-head, DSpark is a flat block: different draft topologies.</text>
</svg>
</div>

## 12. Does DSpark's Draft "Generate Multiple Tokens at Once"

**Yes.** But it should be seen in two layers:

- **DFlash Backbone**: one forward pass produces the candidate tokens of the whole block in parallel (`t1..t6`).
- **Markov Head**: uses `previous token -> logit bias` to correct each position, adding inter-token dependency.

So DSpark is "**parallel backbone + lightweight sequential dependency**" — neither pure parallel nor pure autoregressive, which is exactly Semi-Autoregressive Generation.

## 13. The Easiest Comparison Table to Remember

| Method | How the draft is produced | Inter-token dependency | Tokens per forward | Main issue / advantage |
|---|---|---|---|---|
| Plain decode | Target | strong | 1 token | slow |
| EAGLE-3 | autoregressive Draft Head | **strong** | multiple steps | draft is serial |
| DFlash | block parallel | **weak** | multiple tokens | suffix acceptance decay |
| **DSpark** | DFlash + Markov | **medium** | multiple tokens | balances speed and acceptance |
| MTP | native MTP Head | depends on the MTP structure | multiple tokens | needs native model support / training |

One line to close: **EAGLE-3 guesses one by one but guesses accurately; DFlash guesses a batch in one shot but the later ones drift; DSpark guesses a batch in one shot while using a Markov Head so later guesses refer to the previous token, then uses a Confidence Head to judge how far this chain is worth verifying.** And the precise meaning of "high confidence = high probability" is: the probability that this position is accepted by the target **conditioned on the prefix being correct**; multiplying these conditional probabilities gives the prefix survival probability, and the scheduler decides the verification length from it. On DeepSeek-V4 production traffic the paper reports a per-user generation speed improvement of about **60%-85%** over the production MTP-1 baseline.

> This is Ep7 of the Speculative Decoding Notes series. Previously: [DFlash deep dive](../ep3-dflash), [DSpark deep dive](../ep4-dspark), [DFlash vs DSpark](../ep5-dflash-vs-dspark), [EAGLE-3 deep dive](../ep6-eagle3); back to the [series index](../).
