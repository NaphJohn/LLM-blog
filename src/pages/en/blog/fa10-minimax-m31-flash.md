---
title: "Frontier Architecture Decoding Notes (10): MiniMax M3.1-Flash — All 60 Layers Sparsified, an Efficiency Upgrade That Leaves the Structure Alone"
description: "Continuing from fa3, another MiniMax teardown. Draw the full dataflow of M3 and M3.1-Flash from Embedding through the 60-layer backbone to LM Head, and locate the only structural divergence: M3 keeps Full Attention in Layer 0-2, M3.1-Flash swaps it for MSA, so all 60 layers become index-and-select plus block-sparse GQA under long context. The MoE layer count is unchanged; routing is tuned from top-4 + 1 shared to 129 experts / top-5. Plus QAT, DSpark, and the Day-0 deployment ecosystem."
pubDate: 2026-10-23
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa10-minimax-m31-flash
layout: ../../../layouts/BlogPost.astro
---

## 0. Why Watch M3.1-Flash

fa3 tore apart the M2.7 to M3 "leap of sparse attention." On 2026-10-19, MiniMax M3.1-Flash launched on the API; on 10-23, weights went open-source on Hugging Face / ModelScope (minimax-community license) — **428B total parameters, 23B activated per token, 1M context**, matching M3 in parameter count while dropping BF16 size from 854GB to 252GB.

Three changes over M3, in the official account:

1. **The three full-attention layers are all replaced with sparse attention** (the star of this post);
2. **Quantization-aware training (QAT) on the routed MoE experts and attention paths** — dense layers in MXFP8, experts in NVFP4, so the official weights are already the low-precision build;
3. **DSpark speculative decoding** replaces M3's EAGLE-like MTP, with the draft head shipped alongside the weights.

This post focuses only on the structural details of change 1, answering one question: **where in the dataflow do the two models actually differ?**

## 1. Full Dataflow: Embedding → 60 Layers → LM Head

Conclusion first: **the divergence is not at the entrance, not at the exit, and not in the 57 MoE layers — it is entirely in the attention path of Layer 0-2.**

<div class="fig">
<svg viewBox="0 0 680 754" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Full structure comparison of MiniMax-M3 and MiniMax-M3.1-Flash from Embedding through the 60-layer backbone to LM Head">
  <defs>
    <marker id="arrm" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Embedding → 60-layer backbone → LM Head: divergence sits in the middle</text>

  <rect x="265" y="40" width="150" height="40" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="340" y="60" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#444441">Token IDs</text>
  <line x1="340" y1="80" x2="340" y2="98" stroke="#5F5E5A" stroke-width="1.5" marker-end="url(#arrm)"/>

  <rect x="185" y="104" width="310" height="52" rx="10" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="340" y="123" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#444441">Embedding</text>
  <text x="340" y="143" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">vocab 200,064 · hidden 6,144</text>

  <line x1="340" y1="156" x2="340" y2="174" stroke="#888780" stroke-width="1.5"/>
  <line x1="185" y1="174" x2="495" y2="174" stroke="#888780" stroke-width="1.5"/>
  <line x1="185" y1="174" x2="185" y2="190" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arrm)"/>
  <line x1="495" y1="174" x2="495" y2="190" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrm)"/>

  <rect x="40" y="196" width="290" height="34" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.2"/>
  <text x="185" y="213" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#3C3489">MiniMax-M3 · hybrid attention, 60 layers</text>
  <rect x="350" y="196" width="290" height="34" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="495" y="213" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#085041">MiniMax-M3.1-Flash · MSA everywhere</text>
  <line x1="185" y1="230" x2="185" y2="246" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arrm)"/>
  <line x1="495" y1="230" x2="495" y2="246" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrm)"/>

  <rect x="40" y="252" width="290" height="112" rx="12" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.8"/>
  <text x="185" y="270" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#3C3489">Layer 0 – 2 · dense layers</text>
  <rect x="52" y="284" width="128" height="70" rx="8" fill="#CECBF6" stroke="#534AB7" stroke-width="1.2"/>
  <text x="116" y="300" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#3C3489">Full Attention</text>
  <text x="116" y="317" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">64 heads · 4 KV</text>
  <text x="116" y="332" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">GQA · QK-Norm</text>
  <text x="116" y="347" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">head_dim 128</text>
  <rect x="190" y="284" width="128" height="70" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="254" y="300" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">Dense MLP</text>
  <text x="254" y="317" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">intermediate</text>
  <text x="254" y="332" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">12,288 · SwiGLU</text>
  <text x="254" y="347" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">not MoE</text>
  <line x1="181" y1="319" x2="189" y2="319" stroke="#888780" stroke-width="1.5" marker-end="url(#arrm)"/>

  <rect x="350" y="252" width="290" height="112" rx="12" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.8"/>
  <text x="495" y="270" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">Layer 0 – 2 · Dense MLP kept</text>
  <rect x="362" y="284" width="128" height="70" rx="8" fill="#9FE1CB" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="426" y="300" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">MSA sparse attn</text>
  <text x="426" y="317" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">Indexer 4 heads</text>
  <text x="426" y="332" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">top-16 blocks</text>
  <text x="426" y="347" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">block-sparse GQA</text>
  <rect x="500" y="284" width="128" height="70" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="564" y="300" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">Dense MLP</text>
  <text x="564" y="317" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">intermediate</text>
  <text x="564" y="332" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">12,288 · SwiGLU</text>
  <text x="564" y="347" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">not swapped to MoE</text>
  <line x1="491" y1="319" x2="499" y2="319" stroke="#888780" stroke-width="1.5" marker-end="url(#arrm)"/>

  <line x1="185" y1="364" x2="185" y2="382" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arrm)"/>
  <line x1="495" y1="364" x2="495" y2="382" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrm)"/>

  <rect x="40" y="386" width="290" height="126" rx="12" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.8"/>
  <text x="185" y="404" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#3C3489">Layer 3 – 59 · MoE ×57</text>
  <rect x="52" y="418" width="128" height="84" rx="8" fill="#CECBF6" stroke="#534AB7" stroke-width="1.2"/>
  <text x="116" y="434" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#3C3489">MSA sparse attn</text>
  <text x="116" y="450" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">① Indexer select</text>
  <text x="116" y="465" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">② block-sparse GQA</text>
  <text x="116" y="480" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">≈2,048 tok/query</text>
  <rect x="190" y="418" width="128" height="84" rx="8" fill="#AFA9EC" stroke="#534AB7" stroke-width="1.2"/>
  <text x="254" y="434" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#3C3489">MoE FFN</text>
  <text x="254" y="450" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">128 experts</text>
  <text x="254" y="465" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">top-4 + 1 shared</text>
  <text x="254" y="480" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#3C3489">scaling 2.0</text>
  <line x1="181" y1="460" x2="189" y2="460" stroke="#888780" stroke-width="1.5" marker-end="url(#arrm)"/>

  <rect x="350" y="386" width="290" height="126" rx="12" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.8"/>
  <text x="495" y="404" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">Layer 3 – 59 · MoE ×57</text>
  <rect x="362" y="418" width="128" height="84" rx="8" fill="#9FE1CB" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="426" y="434" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">MSA sparse attn</text>
  <text x="426" y="450" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">① Indexer select</text>
  <text x="426" y="465" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">② block-sparse GQA</text>
  <text x="426" y="480" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">≈2,048 tok/query</text>
  <rect x="500" y="418" width="128" height="84" rx="8" fill="#5DCAA5" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="564" y="434" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">MoE FFN · fused</text>
  <text x="564" y="450" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">129 experts</text>
  <text x="564" y="465" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">top-5 active</text>
  <text x="564" y="480" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">layer count fixed ×57</text>
  <line x1="491" y1="460" x2="499" y2="460" stroke="#888780" stroke-width="1.5" marker-end="url(#arrm)"/>

  <line x1="185" y1="512" x2="185" y2="530" stroke="#534AB7" stroke-width="1.5" marker-end="url(#arrm)"/>
  <line x1="495" y1="512" x2="495" y2="530" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrm)"/>

  <rect x="85" y="534" width="200" height="32" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="185" y="550" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#444441">Final RMSNorm · Gemma-style</text>
  <rect x="395" y="534" width="200" height="32" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="495" y="550" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#444441">Final RMSNorm · Gemma-style</text>
  <line x1="185" y1="566" x2="185" y2="574" stroke="#5F5E5A" stroke-width="1.5" marker-end="url(#arrm)"/>
  <line x1="495" y1="566" x2="495" y2="574" stroke="#5F5E5A" stroke-width="1.5" marker-end="url(#arrm)"/>

  <rect x="85" y="580" width="200" height="40" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="185" y="593" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">LM Head</text>
  <text x="185" y="609" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">200,064 × 6,144</text>
  <rect x="395" y="580" width="200" height="40" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="495" y="593" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">LM Head</text>
  <text x="495" y="609" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">200,064 × 6,144</text>

  <rect x="40" y="638" width="600" height="76" rx="12" fill="#FAEEDA" stroke="#854F0B" stroke-width="1.2"/>
  <text x="340" y="658" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#633806">Key divergence: the attention path of Layer 0 – 2</text>
  <text x="340" y="677" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#854F0B">M3 = Full Attention + Dense MLP ｜ Flash = MSA + Dense MLP</text>
  <text x="340" y="696" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#854F0B">Embedding · Final RMSNorm · LM Head · 57 MoE layers — identical in both</text>
</svg>
<p class="cap">Fig. 1: The full dataflow of MiniMax-M3 (left, purple) and MiniMax-M3.1-Flash (right, green) from Embedding to LM Head. Gray components are shared and unchanged; the divergence sits in the Layer 0-2 attention path.</p>
</div>

### 1. Entrance and exit: identical in both models

Embedding and LM Head are inherited as-is from M3: vocabulary **200,064**, hidden size **6,144**, LM Head at 200,064 × 6,144, with no explicit change to weight tying. The difference does not happen at the entrance or the exit.

### 2. M3's "hybrid" nature is the Layer 0-2 vs Layer 3-59 boundary

- **Layer 0-2**: Full Attention + Dense MLP. Full Attention runs 64 heads / 4 KV heads (GQA), head_dim = 128, per-head QK-Norm; the Dense MLP has intermediate = 12,288, SwiGLU-OAI;
- **Layer 3-59**: MSA + MoE FFN — 128 routed experts, top-4 active + 1 shared expert, routed_scaling_factor = 2.0.

This boundary is strictly decided by the leading three 0s in the `moe_layer_freq` and `sparse_attention_freq` configs. Full-Attention layers have no MSA Indexer branch; they run full causal attention.

### 3. M3.1-Flash's change is precise to the attention path

Flash replaces the Full Attention of the first 3 layers with MSA, so all 60 layers have sparse attention; but **Layer 0-2 still uses Dense MLP and does not enter MoE early**. So Layer 0-2 becomes MSA + Dense MLP, while Layer 3-59 stays MSA + MoE. The MoE-side change is a routing-policy tune — from top-4 + 1 shared to the fused **129 experts / top-5** — and does not change the 57-layer MoE count itself.

### 4. MSA is not one attention, but two stages

<div class="fig">
<svg viewBox="0 0 680 396" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two-stage MSA internals: Indexer block selection plus block-sparse GQA">
  <defs>
    <marker id="arrm2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">Inside an MSA layer: two-stage sparse attention</text>

  <rect x="40" y="120" width="110" height="56" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="95" y="140" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">Past KV</text>
  <text x="95" y="160" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#5F5E5A">causal sequence</text>
  <line x1="154" y1="148" x2="182" y2="148" stroke="#5F5E5A" stroke-width="1.5" marker-end="url(#arrm2)"/>

  <rect x="186" y="93" width="160" height="110" rx="10" fill="#B5D4F4" stroke="#185FA5" stroke-width="1.2"/>
  <text x="266" y="118" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#0C447C">① Indexer</text>
  <text x="266" y="140" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">4 index heads</text>
  <text x="266" y="158" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">128-dim index space</text>
  <text x="266" y="176" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">score 128-token blocks</text>
  <text x="266" y="194" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">pick top-16 blocks</text>
  <line x1="350" y1="148" x2="378" y2="148" stroke="#185FA5" stroke-width="1.5" marker-end="url(#arrm2)"/>

  <rect x="382" y="93" width="160" height="110" rx="10" fill="#B5D4F4" stroke="#185FA5" stroke-width="1.2"/>
  <text x="462" y="118" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#0C447C">② Block-sparse GQA</text>
  <text x="462" y="140" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">compute on selected only</text>
  <text x="462" y="158" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">GQA 64 Q / 4 KV</text>
  <text x="462" y="176" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">per-query budget</text>
  <text x="462" y="194" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0C447C">≈2,048 tokens</text>
  <line x1="546" y1="148" x2="574" y2="148" stroke="#185FA5" stroke-width="1.5" marker-end="url(#arrm2)"/>

  <rect x="578" y="120" width="62" height="56" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="609" y="148" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#444441">Output</text>

  <line x1="364" y1="207" x2="364" y2="248" stroke="#888780" stroke-width="0.5" stroke-dasharray="4 3"/>

  <rect x="40" y="252" width="290" height="104" rx="12" fill="#FBEAF0" stroke="#993556" stroke-width="1.2"/>
  <text x="185" y="278" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#72243E">Full Attention (M3 Layer 0-2 only)</text>
  <text x="185" y="300" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#993556">no Indexer branch</text>
  <text x="185" y="320" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#993556">GQA over the full KV directly</text>
  <text x="185" y="340" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#993556">compute grows quadratically</text>

  <rect x="350" y="252" width="290" height="104" rx="12" fill="#E1F5EE" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="495" y="278" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#085041">MSA (Flash all 60 · M3 last 57)</text>
  <text x="495" y="300" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0F6E56">index-select + block-sparse GQA</text>
  <text x="495" y="320" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0F6E56">≈2,048 tokens per query</text>
  <text x="495" y="340" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0F6E56">no quadratic leftover under long context</text>
</svg>
<p class="cap">Fig. 2: The two-stage MSA — the Indexer scores 128-token blocks and picks the top-16, then block-sparse GQA runs on the selected blocks; the module-level difference from Full Attention is the presence of the Indexer branch.</p>
</div>

Inside each MSA layer: a lightweight **Indexer** (4 index heads, 128-dim index space) first scores the past context in 128-token blocks and selects the top-16 most relevant blocks; then **block-sparse GQA** runs on those blocks, locking each query's attention budget at roughly 2,048 tokens. Full-Attention layers have no Indexer and run GQA over the full KV directly — the most essential module-level difference between M3's Full-Attention layers and its MSA layers.

<div class="keybox">
<strong>In one sentence:</strong> the divergence is not in the embedding, not in the lm_head, and not in the 57 MoE layers — it sits in the attention path of Layer 0-2: M3 keeps Full Attention there, M3.1-Flash swaps it for MSA, so all 60 layers become "index-and-select plus block-sparse GQA" under long context, with no quadratic-complexity leftover layers.
</div>

## 2. Why the "first three layers" Deserve the Fuss

Three layers are only 5% of 60 — trivial at first glance. But Full Attention's compute grows **quadratically** with sequence length: at 1M context, these three layers are precisely the only part of the backbone still running at O(n²). M3.1-Flash erases the last quadratic, with the same payoff as MSA in fa3: compute does not explode with length under long context.

Two engineering moves pair with it:

- **NVFP4 KV cache**: both the indexer and the attention path keep KV in NVFP4; on Blackwell, vLLM uses the CUDA 13.4 QMUL4 instruction to dequantize inside the kernel — up to 1.7× faster indexer QK decode and 1.3× faster sparse-attention decode;
- **Indexer decode context parallelism (DCP)**: up to 3.3× faster indexer per sparse layer at 512K context on 4× GB300 — the same family of ideas as the CP / DCP slicing in sys7 / sys8.

## 3. Beyond Attention: QAT and DSpark

- **QAT**: routed experts in NVFP4, attention paths quantized during training, dense layers in MXFP8 — the official weights are the low-precision build, not a post-hoc compression of a finished model. The checkpoint drops from 854.2GB (BF16) to 252.4GB (including the ~2.3GB DSpark draft head).
- **DSpark**: speculative decoding switches from M3's EAGLE-like MTP to a DSpark draft head shipped with the weights and directly callable by serving frameworks — the same line as the DSpark posts ep4 / ep5 / ep8; in the vLLM recipe, `num_speculative_tokens` defaults to 7 on Blackwell and 3 on AMD MI355X.
- **Day-0 ecosystem**: on 10-23, vLLM / SGLang / Hugging Face / ModelScope were all wired up the same day; Ascend, T-Head, Kunlunxin, Moore Threads, MetaX and other domestic chips, plus Together AI, Fireworks, and Nebius, shipped support simultaneously. Multi-chip deployment goes through FlagOS.

```text
vLLM (Blackwell, TP4) key serving flags:
  --block-size 128                 # MSA KV block size, aligned with the Indexer's 128-token blocks
  --kv-cache-dtype nvfp4           # low-precision attention KV cache
  --attention-config '{"indexer_kv_dtype": "nvfp4"}'
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,...}'
  --disable-hybrid-kv-cache-manager
```

## 4. Benchmarks at a Glance (official)

| Benchmark | M3.1 Flash | Ref: GPT-5.6 Sol |
|---|---|---|
| Automation-Bench v1.0.6 | 62.8 | 50.3 |
| Terminal Bench 4.0 | 38.1 | 51.8 |
| DeepSWE v1.1 | 73.7 | 74.0 |
| OSWorld V2 | 61.7 | 70.6 |
| CyberGym | 82.2 | 83.6 |
| HLE (no tools) | 43.4 | 55.6 |
| LOCA-Bench 256K | 60.3 | 60.0 |

Agentic / tool-call benchmarks (Automation-Bench, BankerToolBench 76.2) are its strength, matching the computer-use and long-context positioning of the M3 family; hard coding and reasoning still trail the first tier. **All numbers above are vendor-scorecard figures; independent leaderboards are pending.**

## 5. Connections to the Series

- **fa3 (MiniMax M2.7 → M3)**: the first MSA leap is the first half of this post; this one adds the "last quadratic" of M3 → M3.1-Flash.
- **sys3 (MSA / CSA / HCA)**: MSA is the branch that "compresses no KV, only selects important positions"; this post shows its latest form and config parameters inside the M3 family.
- **ep4 / ep5 / ep8 (DSpark)**: M3.1-Flash turns DSpark into a draft head shipped with the weights, moving spec decoding from a paper/framework feature to a model-bundled capability.
- **fw2 / fw3 (vLLM / SGLang)**: the Day-0 serving recipe of M3.1-Flash (block-size 128, NVFP4 KV, indexer DCP) reads naturally next to the framework notes.

> Next up, the Operator Notes: writing MLA multi-head latent attention into line-by-line pseudocode you can implement (op1).
