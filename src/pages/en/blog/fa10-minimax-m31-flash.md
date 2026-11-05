---
title: "Frontier Architecture Decoding Notes (10): MiniMax M3.1-Flash — All 60 Layers Sparsified, an Efficiency Upgrade That Leaves the Structure Alone"
description: "Continuing from fa3, another MiniMax teardown. Run a field-by-field diff of the two config.json files and locate the only structural divergence: sparse_attention_freq goes from three leading zeros to sixty ones, so M3 keeps Full Attention in Layer 0-2 while M3.1-Flash swaps it for MSA. On the MoE side, 128 experts / top-4 + 1 shared / scaling 2.0 are byte-identical across both versions. Plus a doubled rope_theta, four-tier QAT precision, the three-stage DSpark draft head, and the Day-0 deployment ecosystem."
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
  <text x="564" y="434" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">MoE FFN · unchanged</text>
  <text x="564" y="450" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">128 experts</text>
  <text x="564" y="465" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">top-4 + 1 shared</text>
  <text x="564" y="480" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#085041">precision only · ×57</text>
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

Flash replaces the Full Attention of the first 3 layers with MSA, so all 60 layers have sparse attention; but **Layer 0-2 still uses Dense MLP and does not enter MoE early**. So Layer 0-2 becomes MSA + Dense MLP, while Layer 3-59 stays MSA + MoE.

**Note that the MoE topology did not actually move** — routing is still `128 experts / top-4 + 1 shared / scaling 2.0`, byte-identical in both versions. The mistaken claim is widespread, so the next section verifies it field by field against the config source.

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

## 2. Laying Both config.json Files Side by Side: Only 4 Lines Actually Differ

The dataflow diagram above is a "read-off conclusion." But a conclusion only holds if you go back to the config source. I pulled both repos' `config.json` and ran a field-by-field diff. For `text_config` the result is remarkably clean:

**Of 31 shared fields, 29 are identical and only 2 differ; another 2 fields were deleted in M3.1.**

<div class="fig">
<svg viewBox="0 0 680 470" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Field-by-field diff of the M3 and M3.1-Flash config.json files">
  <text x="20" y="26" font-size="15" font-weight="700" fill="#1a1a1a">text_config diff: 31 shared fields &#8594; 29 same / 2 changed / 2 removed</text>

  <rect x="20" y="44" width="200" height="86" rx="10" fill="#E1F5EE" stroke="#0F6E56" stroke-width="1.2"/>
  <text x="120" y="68" text-anchor="middle" dominant-baseline="central" font-size="20" font-weight="700" fill="#085041">29</text>
  <text x="120" y="92" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#085041">fields identical</text>
  <text x="120" y="112" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0F6E56">incl. all MoE routing params</text>

  <rect x="240" y="44" width="200" height="86" rx="10" fill="#FAEEDA" stroke="#854F0B" stroke-width="1.2"/>
  <text x="340" y="68" text-anchor="middle" dominant-baseline="central" font-size="20" font-weight="700" fill="#633806">2</text>
  <text x="340" y="92" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#633806">fields changed</text>
  <text x="340" y="112" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#854F0B">rope_theta · sparsity table</text>

  <rect x="460" y="44" width="200" height="86" rx="10" fill="#FBEAF0" stroke="#993556" stroke-width="1.2"/>
  <text x="560" y="68" text-anchor="middle" dominant-baseline="central" font-size="20" font-weight="700" fill="#72243E">2</text>
  <text x="560" y="92" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="700" fill="#72243E">fields removed</text>
  <text x="560" y="112" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#993556">MTP pair &#8594; DSpark</text>

  <line x1="340" y1="136" x2="340" y2="160" stroke="#888780" stroke-width="1.5"/>

  <rect x="20" y="166" width="640" height="284" rx="12" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="1.2"/>
  <text x="40" y="192" font-size="13" font-weight="700" fill="#444441">The four real differences, line by line</text>

  <text x="40" y="222" font-size="12" font-weight="700" fill="#0C447C">1. rope_theta</text>
  <text x="175" y="222" font-size="12" fill="#444441" font-family="ui-monospace, monospace">5,000,000 &#8594; 10,000,000</text>
  <text x="420" y="222" font-size="11" fill="#5F5E5A">RoPE base doubled &#8594; finer long-context positions</text>

  <text x="40" y="252" font-size="12" font-weight="700" fill="#0C447C">2. sparse_attention_freq</text>
  <text x="175" y="252" font-size="12" fill="#444441" font-family="ui-monospace, monospace">[0,0,0,1x57] &#8594; [1x60]</text>
  <text x="420" y="252" font-size="11" fill="#5F5E5A">first 3 layers Full Attn &#8594; MSA, the headline</text>

  <text x="40" y="282" font-size="12" font-weight="700" fill="#0C447C">3. sparse_disable_index_value</text>
  <text x="175" y="282" font-size="12" fill="#444441" font-family="ui-monospace, monospace">[0,0,0,1x57] &#8594; [1x60]</text>
  <text x="420" y="282" font-size="11" fill="#5F5E5A">in lockstep with 2, indexer on everywhere</text>

  <text x="40" y="312" font-size="12" font-weight="700" fill="#0C447C">4. num_mtp_modules / num_nextn_predict_layers</text>
  <text x="175" y="312" font-size="12" fill="#444441" font-family="ui-monospace, monospace">7 / 1 &#8594; absent</text>
  <text x="420" y="312" font-size="11" fill="#5F5E5A">MTP head removed, DSpark dir added</text>

  <line x1="40" y1="330" x2="640" y2="330" stroke="#B4B2A9" stroke-width="1" stroke-dasharray="4 3"/>

  <text x="40" y="352" font-size="12" font-weight="700" fill="#0F6E56">Plus two cross-file changes</text>
  <text x="40" y="376" font-size="11" fill="#444441">&#183; top level gains <tspan font-family="ui-monospace, monospace">quantization_config</tspan> (M3 has no such key at all)</text>
  <text x="40" y="400" font-size="11" fill="#444441">&#183; <tspan font-family="ui-monospace, monospace">vision_config.image_size</tspan> 2016 &#8594; 672, other 18 vision fields unchanged</text>
  <text x="40" y="428" font-size="12" font-weight="700" fill="#72243E">MoE structure: zero changes</text>
  <text x="175" y="428" font-size="11" fill="#993556">128 experts / top-4 / 1 shared / scaling 2.0 byte-identical</text>
</svg>
<p class="cap">Figure 3: Field-by-field diff of the two config.json files. The only structural change is the sparsity frequency table; the MoE topology is untouched.</p>
</div>

### What changed: 4 lines

| Field | M3 | M3.1-Flash | Nature |
|---|---|---|---|
| `rope_theta` | 5,000,000 | **10,000,000** | base frequency doubled |
| `sparse_attention_config.sparse_attention_freq` | `[0,0,0,1,1,…]` (60 entries) | `[1]*60` | **structural change** |
| `sparse_attention_config.sparse_disable_index_value` | `[0,0,0,1,1,…]` (60 entries) | `[1]*60` | in lockstep with the above |
| `num_mtp_modules` / `num_nextn_predict_layers` | `7` / `1` | **field absent** | MTP → DSpark |

Plus two cross-file changes: the top level **gains `quantization_config`** (M3 has no such key at all, meaning M3 shipped unquantized weights), and `vision_config.image_size` drops from **2016 to 672** while the other 18 vision fields stay identical.

### What did not change: 29 lines, including every MoE parameter

This "unchanged list" carries more information than the changed one:

| Field | M3 | M3.1-Flash |
|---|---|---|
| `num_hidden_layers` | 60 | 60 |
| `hidden_size` | 6,144 | 6,144 |
| `num_attention_heads` / `num_key_value_heads` | 64 / 4 | 64 / 4 |
| `head_dim` | 128 | 128 |
| `num_local_experts` | **128** | **128** |
| `num_experts_per_tok` | **4** | **4** |
| `n_shared_experts` | **1** | **1** |
| `routed_scaling_factor` | **2.0** | **2.0** |
| `scoring_func` | **sigmoid** | **sigmoid** |
| `use_routing_bias` | **true** | **true** |
| `moe_layer_freq` | `[0,0,0,1x57]` | identical |
| `hidden_act` | `swigluoai` | `swigluoai` |
| `swiglu_alpha` / `swiglu_limit` | 1.702 / 7.0 | 1.702 / 7.0 |
| `use_gemma_norm` | true | true |
| `use_qk_norm` / `qk_norm_type` | true / `per_head` | identical |
| `rotary_dim` / `partial_rotary_factor` | 64 / 0.5 | identical |
| `max_position_embeddings` | 1,048,576 | 1,048,576 |
| `vocab_size` | 200,064 | 200,064 |
| `dense_intermediate_size` | 12,288 | 12,288 |
| `architectures` | `MiniMaxM3SparseForConditionalGeneration` | identical |

Inside `sparse_attention_config`, apart from those two arrays, the remaining seven scalar parameters are also untouched: `sparse_block_size=128`, `sparse_topk_blocks=16`, `sparse_index_dim=128`, `sparse_num_index_heads=4`, `sparse_score_type="max"`, `sparse_local_block=1`, `sparse_init_block=0`.

The comparison above is only the skeleton. The full set of 31 config fields, per-layer tensor shapes, line-item parameter accounting, the quantization exemption list, and every vision tower parameter are collected in a standalone appendix: **<a href="/LLM-blog/m31-architecture.html">MiniMax M3.1 Flash Architecture Deep Dive (12 sections / 6 figures / 18 tables)</a>** (Chinese, with all field names and values in English). It reads well alongside this post.

<div class="warnbox">
<strong>Correcting a widely repeated error.</strong> Plenty of commentary (including an early draft of this post) claims M3.1 "retunes routing from top-4 + 1 shared to 129 experts / top-5." That is wrong: the config source reads <code>num_local_experts=128</code>, <code>num_experts_per_tok=4</code>, <code>n_shared_experts=1</code>, byte-identical in both versions. The official model card lists exactly three changes — sparse attention everywhere, QAT, DSpark replacing MTP — and none of them touches MoE structure. The real MoE-side change is in <strong>precision</strong>, not <strong>topology</strong>: routed experts moved from MXFP8 to NVFP4 QAT.
</div>

## 3. Why the "first three layers" Deserve the Fuss

Three layers are only 5% of 60 — trivial at first glance. But Full Attention's compute grows **quadratically** with sequence length: at 1M context, these three layers are precisely the only part of the backbone still running at O(n²). M3.1-Flash erases the last quadratic, with the same payoff as MSA in fa3: compute does not explode with length under long context.

Two engineering moves pair with it:

- **NVFP4 KV cache**: both the indexer and the attention path keep KV in NVFP4; on Blackwell, vLLM uses the CUDA 13.4 QMUL4 instruction to dequantize inside the kernel — up to 1.7× faster indexer QK decode and 1.3× faster sparse-attention decode;
- **Indexer decode context parallelism (DCP)**: up to 3.3× faster indexer per sparse layer at 512K context on 4× GB300 — the same family of ideas as the CP / DCP slicing in sys7 / sys8.

## 4. Quantization: Four Precision Tiers and 62 Exempt Layers

`quantization_config` is the new top-level key in M3.1 and the direct cause of the drop from 854.2 GB to 252.4 GB. Field by field:

| Field | Value | Meaning |
|---|---|---|
| `quant_method` | `mxfp8` | baseline scheme is MXFP8 |
| `weight_block_size` | `[1, 32]` | one scale per 32 weights |
| `activation_scheme` | `dynamic` | dynamic activation quantization |
| `moe_quant_algo` | `NVFP4` | expert layers drop to FP4 separately |
| `moe_quant_format` | `nvfp4-pack-quantized` | **packed storage** (two values per byte) |
| `group_size` | `16` | one scale per 16 elements for NVFP4 |

Four precision tiers by sensitivity:

| Component | Precision | Why |
|---|---|---|
| Routed expert MLP | NVFP4 (W4A4) | 97% of parameters; the largest absolute saving |
| Dense layers / attention linears | MXFP8 | medium numerical sensitivity |
| Indexer `q_proj` / `k_proj` | NVFP4 | only affects block selection; errors tolerable |
| MoE gate (57 of them) | high precision | one routing jitter shifts the whole expert distribution |
| `lm_head` | high precision | directly determines logits; rounding propagates linearly |
| `embed_tokens` | high precision | sparse lookups, little to gain |
| `vision_tower` | high precision | wide dynamic range in visual features |
| `multi_modal_projector` / `patch_merge_mlp` | high precision | cross-modal alignment layers amplify error |

`ignored_layers` holds **62 entries**, and the arithmetic checks out: 57 MoE gates (layer indices exactly layers.3 – layers.59, matching the 57 ones in `moe_layer_freq`) + `lm_head` + `model.embed_tokens` + `multi_modal_projector` + `patch_merge_mlp` + `vision_tower`.

<div class="keybox">
<strong>An easily misread detail:</strong> the exemption list contains <strong>no attention output projection</strong>. That means it tolerates less error than the FFN but more than the gate, and runs MXFP8 rather than NVFP4. If you plan to fine-tune or quantize further, this is the layer to protect first.
</div>

The memory arithmetic lines up too. KV cache elements per token = 4 KV heads × 128 dims × 60 layers × 2 (K and V) = **61,440 elements**. Under M3 each element takes 1 byte (FP8); under M3.1 NVFP4 averages about **0.5625 bytes** (including one FP8 scale per 16 elements), giving **34,560 bytes**. Note this is not exactly half — the scale overhead pushes it slightly above 0.5.

<div class="warnbox">
<strong>Why does Hugging Face report 220B parameters instead of 428B?</strong> Because HF counts stored tensor elements, and NVFP4 is packed — one byte holds two values. Once expert weights are packed, the automatic count only sees bytes, so 428B gets reported as roughly 220B. The M3 repo reports 427B (all MXFP8, unpacked), which matches. This is a counting-convention artifact, not a shrunken model.
</div>

## 5. DSpark: A Three-Stage Draft Head

M3.1 deletes `num_mtp_modules=7` / `num_nextn_predict_layers=1` and replaces them with a draft head living in its own `dspark/` directory. Its `config.json` is a complete model config that splits into three stages:

| Stage | Config | Role |
|---|---|---|
| Draft backbone | `num_hidden_layers=5`, `hidden_size=6144`, `dspark_ffn_hidden_size=12288`, `sliding_window=4096` | 5 SWA layers, drafting on local context only |
| Markov head | `dspark_markov_rank=256`, `markov_head_type="vanilla"` | low-rank transition matrix constraining adjacent-token coherence |
| Confidence head | `enable_confidence_head=true`, `confidence_head_with_markov=true` | predicts per-step acceptance probability for the truncation scheduler |

Fields worth memorizing:

- **`dspark_block_size = 7`** — seven tokens drafted per pass. Some sources say 8; that is the SPEED-Bench evaluation setting, not the model config. Do not mix them up.
- **`dspark_target_layer_ids = [3, 17, 31, 45, 59]`** — hidden states are drawn from these five target-model layers as draft conditioning. Note it takes Layer 59, the final layer.
- **`dspark_noise_token_id = 200058`** — the mask placeholder token, landing near the tail of the 200,064 vocabulary.
- **`dspark_use_dense_ffn = true`** — draft layers use dense FFN (`moe_layer_freq=[0,0,0,0,0]`), no MoE, to keep drafting cheap.
- **`dspark_hybrid_context_fusion = false`** — hybrid context fusion is off in this release.
- **`dspark_aux_gemm_precision = "bf16"`** — auxiliary GEMMs stay bf16 and are not quantized.

In the vLLM recipe `num_speculative_tokens` defaults to **7** on Blackwell and **3** on AMD MI355X — the former aligns with `dspark_block_size`, the latter is the conservative choice under memory/bandwidth trade-offs.

## 6. Vision Tower and Multimodality

The two models' vision towers are nearly identical (18 of 19 fields match); the only difference is `image_size` dropping from 2016 to 672:

| Field | Value | Note |
|---|---|---|
| `model_type` | `clip_vision_model` | CLIP-style, not SigLIP |
| `num_hidden_layers` | 32 | |
| `hidden_size` | 1,280 | |
| `num_attention_heads` | 16 | standard MHA, not GQA |
| `intermediate_size` | 5,120 | |
| `patch_size` | 14 | |
| `position_embedding_type` / `rope_mode` | `rope` / `3d` | 3D RoPE, `rope_theta=10000` |
| `hidden_act` | `gelu` | not SwiGLU |
| `layer_norm_eps` | 1e-05 | **LayerNorm**, not RMSNorm |
| `img_token_compression_config` | `patch_merge`, `spatial_merge_size=2`, `temporal_patch_size=2` | 2×2 spatial + 2-frame temporal merge |
| `vision_segment_max_frames` | 4 | at most 4 frames per segment |
| `projector_hidden_size` | 6,144 | aligns with text hidden |
| `image_seq_length` | 576 | default tokens per image |
| `process_image_mode` | `dynamic_res` | dynamic resolution |
| `image_grid_pinpoints` | 36 pairs, 336–2016 | aspect-ratio adaptive tiling |

<div class="keybox">
<strong>The vision tower is the least "MiniMax-like" part of the whole config:</strong> LayerNorm + GELU + standard MHA + 3D RoPE — textbook CLIP ViT, a world apart from the text side's Gemma Norm + swigluoai + GQA 16:1 + QK-Norm. Multimodal models typically reuse open CLIP weights, and this config confirms it.
</div>

## 7. Deployment: Three Hard Constraints

```bash
vllm serve MiniMaxAI/MiniMax-M3.1-Flash \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --kv-cache-dtype nvfp4 \
  --trust-remote-code \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}'
```

Three traps to avoid:

1. **`--block-size 128` is a hard constraint, not a tunable.** The MSA Indexer scores in 128-token blocks, so the KV cache block granularity must align; otherwise the index-to-physical-block mapping is simply wrong.
2. **On Hopper you must fall back.** The NVFP4 QMUL4 dequant instruction exists only on Blackwell (CUDA 13.4). On H100/H800 you drop to the Marlin MoE kernel, cap context at 262,144, and disable video input plus DSpark.
3. **Video and speculative decoding cannot be on together.** The vision tower's `vision_segment_max_frames=4` and DSpark's `sliding_window=4096` are mutually exclusive within the memory budget.

**Day-0 ecosystem**: on 10-23, vLLM / SGLang / Hugging Face / ModelScope were all wired up the same day; Ascend, T-Head, Kunlunxin, Moore Threads, MetaX and other domestic chips, plus Together AI, Fireworks, and Nebius, shipped support simultaneously. Multi-chip deployment goes through FlagOS. For a 428B multimodal MoE, the speed of that rollout is itself what makes it usable in practice — few teams are willing to pull 252GB of weights themselves.

Backing the numbers: `--block-size 128` matches `sparse_block_size=128`, and `num_speculative_tokens=7` matches `dspark_block_size=7`. Both are read off the config, not guessed.

## 8. Parameter Accounting: Where 426B Comes From

The official figure is "roughly 428B total, 23B activated per token." Doing the arithmetic yourself verifies the convention.

| Component | Formula | Parameters |
|---|---|---|
| Routed experts (57 layers) | 57 × 128 × 3 × 6144 × 3072 | 413.1 B |
| Shared experts (57 layers) | 57 × 3 × 6144 × 3072 | 3.23 B |
| MoE gates (57) | 57 × 6144 × 128 | 0.045 B |
| Dense FFN (first 3 layers) | 3 × 3 × 6144 × 12288 | 0.68 B |
| Attention (60 layers) | 60 × (6144×8192 + 2×6144×512 + 8192×6144) | 6.42 B |
| Embedding + LM Head | 2 × 200064 × 6144 | 2.46 B |
| **Subtotal** | | **425.9 B** |
| Vision tower + projector (not counted in text side) | 32-layer ViT + 30720×6144 projector | ≈ 0.8 B |
| **Total** | | **≈ 426.7 B** |

The ~1.3B gap against the official 428B comes from RMSNorm weights, vision positional encodings, and assorted biases.

**Read the 23B activated figure carefully**: `num_experts_per_tok=4` plus `n_shared_experts=1` means each token traverses 5 experts, i.e. the expert portion of 5 × 3 × 6144 × 3072 over 57 layers, plus 60 layers of attention, 3 dense FFN layers, and two projections. This figure **excludes embedding and lm_head** (1.23B each, but they are pure lookups, not per-token compute); counted strictly as per-token compute it comes to about **25.5B**.

## 9. Benchmarks at a Glance (official)

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

## 10. Open Questions in the Data

A few things did not line up while writing this teardown. Listing them beats glossing over:

| Question | Symptom | My read |
|---|---|---|
| 220B vs 428B parameters | HF page shows ~220B, official says 428B | **Counting convention.** NVFP4 packed storage makes HF undercount by byte; see Section 4 |
| `image_size` 2016 → 672 | Vision pretraining resolution cut to one third | **Likely a default change.** `image_grid_pinpoints` still reachably 2016, so capability is intact; only the default tier moved |
| `sparse_init_block=0` | No forced first-block retention | Means **no attention sink**. Whether that hurts opening tokens at extreme context is unstated officially |
| `dspark_block_size` 7 vs 8 | Config says 7, some sources say 8 | 7 is the model definition; 8 is the SPEED-Bench evaluation setting. **Do not mix them** |
| 23B vs 25.5B activated | Official 23B, per-item arithmetic 25.5B | The official figure **excludes embedding and lm_head**, each 1.23B of pure lookup |
| Benchmark scores | All from the vendor scorecard | **Independent leaderboards pending**, especially where Terminal Bench and HLE trail |

## 11. Connections to the Series

- **fa3 (MiniMax M2.7 → M3)**: the first MSA leap is the first half of this post; this one adds the "last quadratic" of M3 → M3.1-Flash.
- **sys3 (MSA / CSA / HCA)**: MSA is the branch that "compresses no KV, only selects important positions"; this post shows its latest form and config parameters inside the M3 family.
- **ep4 / ep5 / ep8 (DSpark)**: M3.1-Flash turns DSpark into a draft head shipped with the weights, moving spec decoding from a paper/framework feature to a model-bundled capability.
- **fw2 / fw3 (vLLM / SGLang)**: the Day-0 serving recipe of M3.1-Flash (block-size 128, NVFP4 KV, indexer DCP) reads naturally next to the framework notes.

**Appendix**: this post follows a single thread — the structural divergence. Everything that needs the full parameter surface spread out went into the <a href="/LLM-blog/m31-architecture.html">MiniMax M3.1 Flash Architecture Deep Dive</a> — 12 sections, 6 figures, 18 tables covering every config field verbatim, the single-layer dataflow and tensor shapes, MoE routing internals, line-item parameter accounting, the three-stage DSpark head, four-tier quantization with 62 exempt layers, all 19 vision tower fields, a 12-item M3 → M3.1 diff, deployment traps, and the open data questions.

> Next up, the Operator Notes: writing MLA multi-head latent attention into line-by-line pseudocode you can implement (op1).
