---
title: "Frontier Architecture Decoding Notes (11): LingBot-VLA 2.0 — MoE Lives Only in the Action Expert, a Game of Width Conservation"
description: "A field-by-field read of LingBot-VLA 2.0's config class and real_robot.yaml: 6.38B total parameters, a Qwen3-VL-4B dense backbone plus a 768-wide sparse action expert, with 36 layers of per-layer joint attention. The core finding is a width identity — 32 experts of width 512 with Top-4 activation plus a 704-wide shared expert sum to exactly 2752, the width of the dense FFN they replace. Compute stays flat while FFN capacity grows 6.2×. Also unpacks Dual-Query distillation (8+8 tokens, not two), a 50 × 55 action chunk, 10-step flow matching, and the 25-step receding horizon used on real robots."
pubDate: 2026-11-27
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa11-lingbot-vla2
layout: ../../../layouts/BlogPost.astro
---

## 0. Why LingBot-VLA 2.0

Eleven posts into this series, we have stayed inside LLMs the whole time — MoE, sparse attention, linear attention, residual connections. This time the battlefield changes: **VLA for embodied AI**.

LingBot-VLA 2.0 (Ant Lingbo, arXiv:2607.06403) earns a spot for two reasons. First, it carries the most mature efficiency tool from the LLM world — **sparse MoE** — into the action-expert side of a VLA. Second, it opens up thoroughly: `configs/vla/real_robot/real_robot.yaml` exposes even knobs like `routed_scaling_factor` and `router_z_loss_coeff`, so we can read the config instead of guessing.

It shares a lineage with the π family (vla3): **a dense VLM backbone, a narrow action expert, and flow matching**. The difference is that π0's action expert is a plain dense FFN, while LingBot-VLA 2.0 replaces it with a 32-expert sparse MoE.

This post answers one question: **where exactly that MoE sits, and why it is designed this way.**

<div class="keybox">
<strong>The conclusion up front:</strong> the MoE is not in the VLM backbone. It lives only in the action expert's FFN, across all 36 layers. And the expert widths are not arbitrary — <strong>Top-4 activation (4 × 512) plus the shared expert (704) equals 2752, exactly the <code>intermediate_size</code> of the dense FFN being replaced.</strong> That is not a coincidence; it is the standard move of "lock the compute, then grow the capacity."
</div>

## 1. Overall shape: a 4B dense backbone with a 768-wide action expert

Start with the part sizes. Every number below comes from the HF safetensors count or the repository config — none are estimates:

| Component | Spec | Source |
|---|---|---|
| Total parameters | **6,375,907,511 (≈6.38B)**, stored as FP32 | HF `robbyant/lingbot-vla-v2-6b` |
| VLM backbone | Qwen3-VL-4B-Instruct, **36 dense layers** | README + `tokenizer_path` |
| Backbone width | hidden **2560**, 32 Q / 8 KV, head_dim 128, FFN 9728 | Qwen3-VL-4B `config.json` |
| Action expert | Qwen2 architecture, **36 layers**, hidden **768** | `modeling_lingbot_vla_v2.py`, `num_layers = 36` |
| Expert attention | 16 Q / 2 KV (GQA 8:1), head_dim 128 | `action_num_attention_heads: 16`, `action_num_key_value_heads: 2` |
| Action / state dim | **55** (breakdown in section 7) | `action_dim: 55`, `max_state_dim: 55` |
| Action chunk | **50** steps | `configuration_lingbot_vla.py`, `chunk_size = 50` |
| Denoising steps | **10** flow-matching steps | same file, `self.num_steps = 10` |
| Inference latency | about **130 ms** per chunk (RTX 4090D) | README |

The two towers differ a lot in width — 2560 versus 768, a factor of 3.3. That ratio drives everything downstream.

<div class="fig">
<svg viewBox="0 0 680 806" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="LingBot-VLA 2.0 full dataflow: the left side is the Qwen3-VL-4B dense backbone processing an image-language prefix, the right side is a 768-wide action expert processing state and noised action suffix, joined by 36 layers of per-layer joint attention and ending in an action output projection that produces a velocity field" id="fig-lb1">
  <style>
    #fig-lb1{--t1:#1a1a1a;--t2:#5F5E5A;--ln:#96918A;
      --vb:#E6F1FB;--vs:#185FA5;--vt:#0C447C;
      --gb:#E1F5EE;--gs:#1D9E75;--gt:#04342C;
      --nb:#F1EFE8;--ns:#5F5E5A;--nt:#444441;
      --ab:#FAEEDA;--as:#854F0B;--at:#633806;
      --pb:#FBEAF0;--ps:#993556;--pt:#72243E;}
    :root[data-theme="dark"] #fig-lb1{--t1:#F1EFE8;--t2:#D3D1C7;--ln:#727E8A;
      --vb:#042C53;--vs:#85B7EB;--vt:#B5D4F4;
      --gb:#04342C;--gs:#5DCAA5;--gt:#9FE1CB;
      --nb:#2C2C2A;--ns:#727E8A;--nt:#D3D1C7;
      --ab:#3a2e0c;--as:#F0C078;--at:#FBE0A8;
      --pb:#3d1220;--ps:#E08CA6;--pt:#F4C4D2;}
    #fig-lb1 text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
    #fig-lb1 .h1{font-size:15px;font-weight:700;fill:var(--t1)}
    #fig-lb1 .h2{font-size:12.5px;font-weight:700;fill:var(--t1)}
    #fig-lb1 .lb{font-size:11.5px;font-weight:700}
    #fig-lb1 .bd{font-size:11px;fill:var(--t2)}
    #fig-lb1 .mi{font-size:10.5px;fill:var(--t2)}
    #fig-lb1 .ct{text-anchor:middle;dominant-baseline:central}
  </style>
  <text class="h1" x="20" y="26">Two towers: a 2560-wide dense backbone and a 768-wide sparse expert</text>
  <rect x="40" y="44" width="290" height="88" rx="10" fill="var(--vb)" stroke="var(--vs)" stroke-width="1.2"/>
  <text class="lb ct" x="185" y="64" fill="var(--vt)">Prefix · VLM tower</text>
  <text class="bd ct" x="185" y="86">images + instruction + 16 alignment query tokens</text>
  <text class="mi ct" x="185" y="106">token count varies with image patch count</text>
  <text class="mi ct" x="185" y="122">hidden 2560</text>
  <rect x="350" y="44" width="290" height="88" rx="10" fill="var(--gb)" stroke="var(--gs)" stroke-width="1.2"/>
  <text class="lb ct" x="495" y="64" fill="var(--gt)">Suffix · action expert tower</text>
  <text class="bd ct" x="495" y="86">state (55) + noised action 50 × 55 + time</text>
  <text class="mi ct" x="495" y="106">state_proj / action_in_proj → 768</text>
  <text class="mi ct" x="495" y="122">action_time_mlp fuses the time embedding</text>
  <line x1="185" y1="132" x2="185" y2="152" stroke="var(--vs)" stroke-width="1.5"/>
  <line x1="495" y1="132" x2="495" y2="152" stroke="var(--gs)" stroke-width="1.5"/>
  <rect x="40" y="158" width="600" height="46" rx="8" fill="var(--nb)" stroke="var(--ns)" stroke-width="1.2"/>
  <text class="h2 ct" x="340" y="175">Embedding / projection happen separately; the sequences split here</text>
  <text class="mi ct" x="340" y="193">left length = prefix length; right length = 1 (state) + 50 (actions)</text>
  <line x1="185" y1="204" x2="185" y2="224" stroke="var(--vs)" stroke-width="1.5"/>
  <line x1="495" y1="204" x2="495" y2="224" stroke="var(--gs)" stroke-width="1.5"/>
  <rect x="40" y="230" width="600" height="278" rx="12" fill="none" stroke="var(--ln)" stroke-width="1.2" stroke-dasharray="5 4"/>
  <text class="h2" x="56" y="252">36 Transformer layers × per-layer joint attention</text>
  <text class="mi" x="330" y="252">layer counts must match; enforced by a hard assert</text>
  <rect x="56" y="264" width="258" height="70" rx="8" fill="var(--vb)" stroke="var(--vs)" stroke-width="1"/>
  <text class="lb ct" x="185" y="282" fill="var(--vt)">Layer 0 – 35 · dense</text>
  <text class="bd ct" x="185" y="301">32 Q / 8 KV · FFN 9728</text>
  <text class="mi ct" x="185" y="320">Qwen3-VL-4B weights kept as-is</text>
  <rect x="366" y="264" width="258" height="70" rx="8" fill="var(--gb)" stroke="var(--gs)" stroke-width="1"/>
  <text class="lb ct" x="495" y="282" fill="var(--gt)">Layer 0 – 35 · MoE</text>
  <text class="bd ct" x="495" y="301">16 Q / 2 KV · FFN → 32 experts</text>
  <text class="mi ct" x="495" y="320">Top-4 + shared 704 per layer</text>
  <line x1="314" y1="299" x2="366" y2="299" stroke="var(--as)" stroke-width="2" stroke-dasharray="4 3"/>
  <text class="mi ct" x="340" y="289" fill="var(--at)">joint</text>
  <text class="mi ct" x="340" y="358">↑ per layer: both towers compute Q/K/V → concat along seq → one attention → slice back ↓</text>
  <rect x="56" y="376" width="258" height="54" rx="8" fill="var(--vb)" stroke="var(--vs)" stroke-width="1" opacity="0.55"/>
  <text class="bd ct" x="185" y="395">Layers 1 … 34 (34 identical repeats)</text>
  <text class="mi ct" x="185" y="414">action tokens see visual-language tokens throughout</text>
  <rect x="366" y="376" width="258" height="54" rx="8" fill="var(--gb)" stroke="var(--gs)" stroke-width="1" opacity="0.55"/>
  <text class="bd ct" x="495" y="395">Layers 1 … 34 (34 identical repeats)</text>
  <text class="mi ct" x="495" y="414">prefix KV computed once and cached</text>
  <rect x="56" y="440" width="258" height="50" rx="8" fill="var(--vb)" stroke="var(--vs)" stroke-width="1"/>
  <text class="bd ct" x="185" y="458">Layer 35 · final hidden</text>
  <text class="mi ct" x="185" y="476">conditioning only, no logits</text>
  <rect x="366" y="440" width="258" height="50" rx="8" fill="var(--gb)" stroke="var(--gs)" stroke-width="1"/>
  <text class="bd ct" x="495" y="458">Layer 35 · final hidden</text>
  <text class="mi ct" x="495" y="476">last 50 positions go to output proj</text>
  <line x1="495" y1="490" x2="495" y2="516" stroke="var(--gs)" stroke-width="1.5"/>
  <rect x="350" y="522" width="290" height="58" rx="10" fill="var(--pb)" stroke="var(--ps)" stroke-width="1.2"/>
  <text class="lb ct" x="495" y="542" fill="var(--pt)">action_out_proj: 768 → 55</text>
  <text class="bd ct" x="495" y="562">emits velocity field v_t, shaped 50 × 55</text>
  <line x1="495" y1="580" x2="495" y2="606" stroke="var(--ps)" stroke-width="1.5"/>
  <rect x="350" y="612" width="290" height="86" rx="10" fill="var(--ab)" stroke="var(--as)" stroke-width="1.2"/>
  <text class="lb ct" x="495" y="632" fill="var(--at)">Flow-matching loop × 10</text>
  <text class="bd ct" x="495" y="652">x_t ← x_t + dt · v_t, dt = −1/10</text>
  <text class="mi ct" x="495" y="670">t starts at 1.0, while t ≥ −dt/2</text>
  <text class="mi ct" x="495" y="687">prefix KV reused 10 times, never recomputed</text>
  <rect x="40" y="716" width="600" height="76" rx="12" fill="var(--nb)" stroke="var(--ns)" stroke-width="1.2"/>
  <text class="h2" x="56" y="738">Output: a 50 × 55 continuous action chunk</text>
  <text class="bd" x="56" y="759">55 = 14 arm + 14 end-effector pose + 2 gripper + 12 hand + 4 waist + 2 head + 3 base + 4 reserved</text>
  <text class="mi" x="56" y="778">real-robot deployment executes only the first 25 steps, then replans; chunk 50 is generation length</text>
</svg>
<p class="fig-cap">Figure 1: the full dataflow of LingBot-VLA 2.0. The towers differ 3.3× in width (2560 vs 768) yet match exactly in depth (36); each layer runs a single joint attention so the narrow tower's action tokens can read the wide tower's visual-language context.</p>
</div>

### ① Matching depth is a hard constraint

`modeling_lingbot_vla_v2.py` contains this assert:

```python
models = [self.qwenvl.model.language_model, self.qwen_expert.model]
assert action_num_layers == num_layers, (
    "Action expert and VLM must have the same number of layers "
    f"(got action={action_num_layers}, vlm={num_layers})."
)
```

Why must they match? Because the towers interlock **layer by layer** — layer *i* of the action expert consumes layer *i* of the backbone's attention output. Unequal depth breaks the pairing. This descends directly from π0, whose action expert is likewise a narrow isomorph aligned to PaliGemma's depth.

### ② Joint attention: one computation, split two ways

Each layer works like this: both towers compute Q/K/V with their own projections → concatenate along the sequence dimension → run **one** attention → slice the result back by `start`/`end`:

```python
for layer_idx in range(num_layers):
    for i, hidden_states in enumerate(inputs_embeds):
        q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True, ...)
        query_states.append(q.float())
        key_states.append(k.float())
        value_states.append(v.float())
    query_states = torch.cat(query_states, dim=1)   # concatenate the two towers
    ...
    att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)
    # att_output is then sliced back per tower by start/end
```

One easily missed detail: **the towers use different head counts** (backbone 32 Q / 8 KV, expert 16 Q / 2 KV), but because their widths also differ (2560 vs 768), each tower's projected QKV matches its own head layout. The concatenation happens along the sequence dimension, so the attention matrix is **cross-tower dense** — action tokens attend directly to every image and language token, with no separate cross-attention adapter.

### ③ KV reuse at inference

`sample_actions()` runs the prefix once with `fill_kv_cache=True` to obtain `past_key_values`; the subsequent 10 denoising steps run only the suffix. Image encoding and language understanding therefore happen **once for the entire denoising loop**, and nearly all marginal cost of the 10 steps sits in the 768-wide narrow tower.

This is what makes 130 ms achievable — recomputing the vision tower each step would be impossible on a 4090D.

### ④ AdaNorm: time conditioning without cross-attention

`adanorm_time: true` is one of the config switches. The timestep embedding is not injected through cross-attention; it modulates the scale and shift of LayerNorm directly (`replace_lnorm_with_adanorm`), and the final norm is also AdaNorm (`final_norm_adanorm`).

This is the same lineage as π0 — AdaNorm is far cheaper than cross-attention for time conditioning and consumes no sequence positions. For a 10-step denoiser, that saving matters.

## 2. Where the MoE sits: only in the action expert's FFN

This is the easiest place to misread. Plenty of write-ups say "LingBot-VLA 2.0 is a sparse MoE model," which sounds like the backbone is MoE too. **It is not.**

Section 4.1 of the paper is explicit:

> we instantiate sparse MoE layers inside the action expert, which replace the feed-forward network (FFN) in the action expert… all action-expert transformer blocks are equipped with MoE FFNs

The code confirms it — `_install_moe_blocks()` only rewrites `qwen_expert`'s MLP, leaving the backbone's `qwenvl` untouched:

```python
for idx in token_moe_layers:
    self.qwen_expert.model.layers[idx].mlp = Qwen2TokenMoeBlock(token_config)
```

So the accurate statement is: **the Qwen3-VL-4B backbone is dense, and all 36 FFNs in the action expert are replaced by sparse MoE.**

<div class="warnbox">
<strong>A common error worth flagging:</strong> the phrase "sparse MoE action expert, 32 experts / Top-4" is numerically correct (<code>token_num_experts: 32</code>, <code>token_top_k: 4</code>), but it describes only the action expert and <strong>must not be generalized to the whole model</strong>. The Qwen3-VL-4B backbone keeps a dense FFN with intermediate size 9728.
</div>

The full MoE block from `real_robot.yaml`:

```yaml
use_moe: true
token_moe_layers: [0,1,2,...,35]      # all 36 layers, none exempt
token_num_experts: 32
token_top_k: 4
token_moe_intermediate_size: 512     # per-expert width
token_shared_intermediate_size: 704  # shared-expert width
router_activation: "sigmoid"
routed_scaling_factor: 4.0
bias_update_speed: 0                 # bias correction disabled
sequence_wise_loss_coeff: 1e-3
router_z_loss_coeff: 1e-4
use_shared_expert_gate: false
```

### The width identity

Those two numbers — `512` and `704` — look arbitrary. Put them next to the config class defaults and something clicks:

```
Top-4 active experts + shared expert = 4 × 512 + 704 = 2752
dense FFN intermediate_size being replaced  = 2752
                                              equal ✓
```

`configuration_lingbot_vla.py` sets `expert_intermediate_size: int = 2752` — that is the width of the dense FFN being replaced. After the MoE swap, **the FFN width actually activated per forward pass is identical to the dense original**.

<div class="fig">
<svg viewBox="0 0 680 580" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Width identity: a 2752-wide dense FFN is replaced by 32 experts of width 512, Top-4 activation of 2048 plus a 704-wide shared expert, summing to exactly 2752; per-layer parameters rise from 6.34M to 39.37M" id="fig-lb2">
  <style>
    #fig-lb2{--t1:#1a1a1a;--t2:#5F5E5A;--ln:#96918A;
      --db:#F1EFE8;--ds:#5F5E5A;--dt:#444441;
      --eb:#E6F1FB;--es:#185FA5;--et:#0C447C;
      --onb:#185FA5;--shb:#1D9E75;--offb:#CDC9BC;
      --okb:#E1F5EE;--oks:#1D9E75;--okt:#04342C;}
    :root[data-theme="dark"] #fig-lb2{--t1:#F1EFE8;--t2:#D3D1C7;--ln:#727E8A;
      --db:#2C2C2A;--ds:#727E8A;--dt:#D3D1C7;
      --eb:#042C53;--es:#85B7EB;--et:#B5D4F4;
      --onb:#378ADD;--shb:#28B589;--offb:#5F5E5A;
      --okb:#04342C;--oks:#5DCAA5;--okt:#9FE1CB;}
    #fig-lb2 text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
    #fig-lb2 .h1{font-size:15px;font-weight:700;fill:var(--t1)}
    #fig-lb2 .h2{font-size:12px;font-weight:700;fill:var(--t1)}
    #fig-lb2 .mi{font-size:10.5px;fill:var(--t2)}
    #fig-lb2 .ct{text-anchor:middle;dominant-baseline:central}
  </style>
  <text class="h1" x="20" y="26">The width identity: flat compute, 6.2× capacity</text>
  <text class="h2" x="40" y="56">Before · dense FFN</text>
  <rect x="40" y="66" width="600" height="56" rx="6" fill="var(--db)" stroke="var(--ds)" stroke-width="1.2"/>
  <text class="ct" x="340" y="86" font-size="12.5" font-weight="700" fill="var(--dt)">intermediate_size = 2752</text>
  <text class="ct" x="340" y="106" font-size="11" fill="var(--t2)">3 × 768 × 2752 = 6.34 M parameters per layer</text>
  <line x1="340" y1="122" x2="340" y2="148" stroke="var(--ln)" stroke-width="1.5"/>
  <text class="mi ct" x="352" y="135">replace</text>
  <text class="h2" x="40" y="172">After · 32-expert MoE (Top-4)</text>
  <rect x="40" y="182" width="600" height="32" rx="6" fill="var(--db)" stroke="var(--ds)" stroke-width="0.8" stroke-dasharray="3 2"/>
  <text class="mi ct" x="340" y="198">32 experts of width 512 — all in memory, only 4 chosen per step (16 per row)</text>
  <g>
    <rect x="40" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="78" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="116" y="226" width="30" height="30" rx="3" fill="var(--onb)"/>
    <rect x="154" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="192" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="230" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="268" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="306" y="226" width="30" height="30" rx="3" fill="var(--onb)"/>
    <rect x="344" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="382" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="420" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="458" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="496" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="534" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="572" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="610" y="226" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="40" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="78" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="116" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="154" y="262" width="30" height="30" rx="3" fill="var(--onb)"/>
    <rect x="192" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="230" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="268" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="306" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="344" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="382" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="420" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="458" y="262" width="30" height="30" rx="3" fill="var(--onb)"/>
    <rect x="496" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="534" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="572" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
    <rect x="610" y="262" width="30" height="30" rx="3" fill="var(--offb)"/>
  </g>
  <text class="mi" x="40" y="312">4 blue = experts picked by Top-4 this round (4 × 512 = 2048)</text>
  <text class="mi" x="40" y="328">28 grey = inactive, weights resident but not computed</text>
  <rect x="40" y="344" width="600" height="44" rx="6" fill="var(--okb)" stroke="var(--oks)" stroke-width="1.2"/>
  <text class="ct" x="340" y="359" font-size="11.5" font-weight="700" fill="var(--okt)">Shared expert 704 · always active, not one of the 32</text>
  <text class="ct" x="340" y="377" font-size="10.5" fill="var(--okt)">holds the embodiment-agnostic priors</text>
  <rect x="40" y="400" width="600" height="58" rx="8" fill="var(--okb)" stroke="var(--oks)" stroke-width="1.2"/>
  <text class="ct" x="340" y="420" font-size="12.5" font-weight="700" fill="var(--okt)">4 × 512 + 704 = 2752 = the dense FFN width replaced</text>
  <text class="ct" x="340" y="441" font-size="11" fill="var(--okt)">Active width per forward is identical; FLOPs essentially unchanged</text>
  <text class="h2" x="40" y="484">Cost and payoff</text>
  <rect x="40" y="496" width="290" height="66" rx="8" fill="var(--eb)" stroke="var(--es)" stroke-width="1"/>
  <text class="ct" x="185" y="514" font-size="11.5" font-weight="700" fill="var(--et)">Params: 6.34 M → 39.37 M per layer</text>
  <text class="ct" x="185" y="534" font-size="10.5" fill="var(--t2)">36 FFN layers: 0.228 B → 1.42 B</text>
  <text class="ct" x="185" y="551" font-size="10.5" fill="var(--t2)">Memory footprint up 6.2×</text>
  <rect x="350" y="496" width="290" height="66" rx="8" fill="var(--eb)" stroke="var(--es)" stroke-width="1"/>
  <text class="ct" x="495" y="514" font-size="11.5" font-weight="700" fill="var(--et)">Compute: unchanged</text>
  <text class="ct" x="495" y="534" font-size="10.5" fill="var(--t2)">active width pinned at 2752</text>
  <text class="ct" x="495" y="551" font-size="10.5" fill="var(--t2)">inference FLOPs on par with dense</text>
</svg>
<p class="fig-cap">Figure 2: the width identity. Top-4 activation across 32 experts of width 512 yields 2048; adding the 704-wide shared expert gives exactly 2752, the width of the dense FFN being replaced. Per-layer parameters grow from 6.34M to 39.37M while the activated width per forward pass stays fixed.</p>
</div>

The intent is plain: **trade memory for capacity**. Thirty-two experts let the model dedicate knowledge partitions to different embodiments and tasks, while each forward pass costs the same as the dense model it grew out of. The paper puts it as "fine-grained expert segmentation and shared expert isolation allow universal priors and specialized embodiment/task patterns to coexist under the same active compute budget" — fine granularity at 512 for flexible composition, an isolated shared expert to hold the cross-embodiment priors.

<div class="keybox">
<strong>Why 512 and 704 are worth remembering:</strong> if the expert width were anything but 512, the identity would fail. Conversely, seeing the identity lets you infer the design order: the authors fixed "keep total FFN width at 2752" as a constraint, then derived "32 experts × 512 plus a shared 704." This is the safest way to MoE-ify anything — <strong>lock compute first, then scale capacity</strong>, so the structural change never drags inference cost along with it.
</div>

### Four non-default routing choices

Compared with the config class defaults, `real_robot.yaml` deviates in four places:

| Field | Class default | real_robot.yaml | Meaning |
|---|---|---|---|
| `router_activation` | `softmax` | **`sigmoid`** | each expert scores independently, no forced normalization |
| `routed_scaling_factor` | `1.0` | **`4.0`** | route outputs scaled 4×, compensating Top-4 dilution |
| `bias_update_speed` | `0.001` | **`0`** | bias correction off; balance handled by auxiliary losses |
| `use_shared_expert_gate` | `true` | **`false`** | shared expert added directly, no learned gate |

sigmoid plus scaling 4.0 is the classic DeepSeek-family combination: sigmoid stops experts competing for share, scaling restores activation magnitude. `bias_update_speed: 0` deserves a note — it gives up inference-time expert-load correction and relies on two auxiliary losses instead:

- `sequence_wise_loss_coeff: 1e-3` with mode `per_sequence`: balances expert usage **within a single sequence** rather than across the batch
- `router_z_loss_coeff: 1e-4`: constrains router logit magnitude so routing stays unconfident

Per-sequence balancing is an interesting choice. A VLA batch mixes embodiments and tasks, so batch-global balancing averages away the case where one sequence monopolizes a few experts; per-sequence balancing suppresses exactly that skew.

## 3. Dual-Query distillation: not two queries, but 8 + 8 tokens

The paper writes this as `[Q_t, Q_{t+T}]`, two queries, which reads like two tokens. **In fact each group is 8 tokens, 16 in total.**

The config sets `num_task_tokens: 8`, and `_get_align_tokens()` averages the learnable embedding table in groups of 8, emitting 8 tokens:

```python
def _get_align_tokens(tokens):
    tk_weights = tokens.view(self.num_task_tokens,
                             tokens.shape[0] // self.num_task_tokens,
                             tokens.shape[1])
    tk_weights = tk_weights.mean(dim=1)
    return tk_weights            # shape (8, hidden)
```

Both groups are then spliced into the prefix sequence and take part in joint attention:

```python
for segment_name in prefix_query_segments(use_depth_align=True, use_future_depth=..., ...):
    if segment_name == "current_depth":   _append(align_embs, ...)
    elif segment_name == "future_video":  _append(future_video_align_embs, ...)
    elif segment_name == "future_depth":  _append(future_align_embs, ...)
```

So "Dual" refers to **two groups** (current frame and future frame), not "two tokens." The symbolic notation (`Q_t` / `Q_{t+T}`) and the implemented token count are different things — a distinction technical writing must keep straight.

<div class="fig">
<svg viewBox="0 0 680 452" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dual-Query distillation structure: the prefix sequence consists of image tokens, 8 current-frame query tokens, language tokens, and 8 future-frame query tokens; the two query groups distil scene geometry from LingBot-Depth and temporal evolution from DINO-Video respectively" id="fig-lb3">
  <style>
    #fig-lb3{--t1:#1a1a1a;--t2:#5F5E5A;--ln:#96918A;
      --ib:#F1EFE8;--is:#5F5E5A;--it:#444441;
      --qb:#B5D4F4;--qs:#185FA5;--qt:#0C447C;
      --lb2:#E1F5EE;--ls:#1D9E75;--lt:#04342C;
      --tb:#FAEEDA;--ts:#854F0B;--tt:#633806;
      --wb:#FBEAF0;--ws:#993556;--wt:#72243E;}
    :root[data-theme="dark"] #fig-lb3{--t1:#F1EFE8;--t2:#D3D1C7;--ln:#727E8A;
      --ib:#2C2C2A;--is:#727E8A;--it:#D3D1C7;
      --qb:#185FA5;--qs:#85B7EB;--qt:#B5D4F4;
      --lb2:#04342C;--ls:#5DCAA5;--lt:#9FE1CB;
      --tb:#3a2e0c;--ts:#F0C078;--tt:#FBE0A8;
      --wb:#3d1220;--ws:#E08CA6;--wt:#F4C4D2;}
    #fig-lb3 text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
    #fig-lb3 .h1{font-size:15px;font-weight:700;fill:var(--t1)}
    #fig-lb3 .h2{font-size:12px;font-weight:700;fill:var(--t1)}
    #fig-lb3 .lb{font-size:11px;font-weight:700}
    #fig-lb3 .bd{font-size:10.5px;fill:var(--t2)}
    #fig-lb3 .ct{text-anchor:middle;dominant-baseline:central}
  </style>
  <text class="h1" x="20" y="26">Anatomy of the prefix: where the 16 learnable query tokens live</text>
  <rect x="40" y="46" width="150" height="52" rx="6" fill="var(--ib)" stroke="var(--is)" stroke-width="1.2"/>
  <text class="ct" x="115" y="64" font-size="11.5" font-weight="700" fill="var(--it)">image tokens</text>
  <text class="ct" x="115" y="83" font-size="10" fill="var(--t2)">variable length</text>
  <rect x="196" y="46" width="118" height="52" rx="6" fill="var(--qb)" stroke="var(--qs)" stroke-width="1.4"/>
  <text class="ct" x="255" y="64" font-size="11.5" font-weight="700" fill="var(--qt)">Q_t × 8</text>
  <text class="ct" x="255" y="83" font-size="10" fill="var(--qt)">current geometry</text>
  <rect x="320" y="46" width="150" height="52" rx="6" fill="var(--ib)" stroke="var(--is)" stroke-width="1.2"/>
  <text class="ct" x="395" y="64" font-size="11.5" font-weight="700" fill="var(--it)">language tokens</text>
  <text class="ct" x="395" y="83" font-size="10" fill="var(--t2)">cap 72</text>
  <rect x="476" y="46" width="118" height="52" rx="6" fill="var(--qb)" stroke="var(--qs)" stroke-width="1.4"/>
  <text class="ct" x="535" y="64" font-size="11.5" font-weight="700" fill="var(--qt)">Q_t+T × 8</text>
  <text class="ct" x="535" y="83" font-size="10" fill="var(--qt)">future evolution</text>
  <text class="bd" x="40" y="118">↑ segment order comes from prefix_query_segments; the layout above is one valid arrangement</text>
  <line x1="255" y1="124" x2="255" y2="152" stroke="var(--qs)" stroke-width="1.5"/>
  <line x1="535" y1="124" x2="535" y2="152" stroke="var(--qs)" stroke-width="1.5"/>
  <rect x="40" y="158" width="270" height="96" rx="10" fill="var(--lb2)" stroke="var(--ls)" stroke-width="1.2"/>
  <text class="ct" x="175" y="178" font-size="12" font-weight="700" fill="var(--lt)">LingBot-Depth (teacher)</text>
  <text class="ct" x="175" y="200" font-size="10.5" fill="var(--t2)">monocular depth estimation</text>
  <text class="ct" x="175" y="219" font-size="10.5" fill="var(--t2)">loss = L1, weight 0.004</text>
  <text class="ct" x="175" y="238" font-size="10.5" fill="var(--t2)">aligned to queries at 8 tokens → eq. (9)</text>
  <rect x="372" y="158" width="268" height="96" rx="10" fill="var(--wb)" stroke="var(--ws)" stroke-width="1.2"/>
  <text class="ct" x="506" y="178" font-size="12" font-weight="700" fill="var(--wt)">DINO-Video (teacher)</text>
  <text class="ct" x="506" y="200" font-size="10.5" fill="var(--t2)">303.13 M · 5 M clips · 16 frames</text>
  <text class="ct" x="506" y="219" font-size="10.5" fill="var(--t2)">loss = squared Frobenius, weight 0.004</text>
  <text class="ct" x="506" y="238" font-size="10.5" fill="var(--t2)">block-wise causal attention + 3D-RoPE</text>
  <rect x="40" y="272" width="600" height="76" rx="10" fill="var(--tb)" stroke="var(--ts)" stroke-width="1.2"/>
  <text class="ct" x="340" y="292" font-size="12" font-weight="700" fill="var(--tt)">Distillation direction: teacher → query tokens, which join the forward pass</text>
  <text class="ct" x="340" y="313" font-size="10.5" fill="var(--t2)">at inference the teachers are dropped entirely; only the 16 tokens' weights remain</text>
  <text class="ct" x="340" y="332" font-size="10.5" fill="var(--t2)">purpose: compress "current scene geometry + future scene evolution" into positions the backbone reads</text>
  <rect x="40" y="364" width="600" height="72" rx="10" fill="var(--ib)" stroke="var(--is)" stroke-width="1.2"/>
  <text class="ct" x="340" y="384" font-size="12" font-weight="700" fill="var(--it)">One easy confusion: num_task_tokens = 8 is the count *per group*</text>
  <text class="ct" x="340" y="405" font-size="10.5" fill="var(--t2)">the paper's Q_t and Q_t+T denote two groups, not two tokens</text>
  <text class="ct" x="340" y="424" font-size="10.5" fill="var(--t2)">video.share_future_depth_query: true → future video and future depth share one group</text>
</svg>
<p class="fig-cap">Figure 3: Dual-Query distillation. Two groups of 8 learnable query tokens ride inside the prefix and take part in joint attention, distilled from LingBot-Depth and DINO-Video during training; the teachers are unloaded at inference, leaving only the query weights.</p>
</div>

### Why queries rather than feeding features directly

If the goal is "let the backbone know current depth and the future frame," the obvious approach is to feed depth maps or future frames as extra inputs. That causes two problems: the input sequence gets longer (depth maps and video frames are not cheap), and inference must actually run the teacher models.

Query distillation sidesteps both: **the knowledge is compressed into the weights of 16 tokens**, so teachers appear only during training. The cost is that those 16 tokens must squeeze through joint attention to be read by the backbone — limited bandwidth — which is why all three distillation weights are set very small (`depth_loss_weight`, `future_depth_loss_weight`, `future_video_loss_weight` are all **0.004**), keeping the auxiliary objectives from overpowering the main action loss.

`video.share_future_depth_query: true` is another detail: future video and future depth **share one query group** rather than getting 8 each. `real_robot.yaml` pairs `future_video_share_future_depth_query: true` with `use_future_depth: true`, and the code enforces it:

```python
if (not self.use_future_depth and getattr(self, "use_future_video", False)
        and getattr(self, "future_video_share_future_depth_query", False)):
    raise ValueError("share_future_depth_query=True requires depth.use_future_depth=True.")
```

The rationale: future depth and future video describe **two aspects of the same thing** (geometry versus appearance), so sharing a query forces them into one semantic space and saves 8 positions.

## 4. Action generation: flow matching, 10 steps, chunk 50

The loop in `sample_actions()` is plain explicit Euler:

```python
dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
x_t = noise
time = torch.tensor(1.0, dtype=dtype, device=device)
while time >= -dt / 2:
    v_t = predict_velocity_fn(state, prefix_pad_masks, past_key_values,
                              x_t, expanded_time, ...)
    x_t += dt * v_t
    time += dt
```

`num_steps = 10` is hardcoded in `configuration_lingbot_vla.py`:

```python
self.num_steps = 10
self.n_obs_steps = 1
```

With `dt = -0.1` and `t` descending from 1.0 while `t >= 0.05`, that is exactly 10 iterations — matching the README's "10 denoising steps / 130 ms."

<div class="warnbox">
<strong>One trap:</strong> line 309 of <code>deploy/lingbot_vla_v2_policy.py</code> contains a commented-out <code># config.num_steps = 4</code>. Some readers take that as the default. It is not — it is leftover from an experiment, commented code does not execute, and the config class's 10 applies.
</div>

Training uses standard flow-matching interpolation:

```python
x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions      # target velocity field
```

Note the time direction: `t=1` is pure noise, `t=0` is clean action, hence `dt` is negative and `t` descends from 1.0 to 0 at inference. Same convention as π0.

The action chunk length is **50**, set in the config class:

```python
chunk_size: int = 50,
...
self.chunk_size = chunk_size
self.n_action_steps = chunk_size
```

`real_robot.yaml` does not override it, so it is 50. Noise is allocated as `(bsize, n_action_steps, max_action_dim)`, i.e. `(B, 50, 55)`.

## 5. Parameter accounting: what fills 6.38B

The HF safetensors count is **6,375,907,511**, the most authoritative figure (all counted as FP32, corresponding to roughly 25.5 GB of weights). Breaking it down:

| Component | Expression | Parameters |
|---|---|---|
| Qwen3-VL-4B backbone | 36 layers, hidden 2560, FFN 9728 | ≈ 4.0 B |
| Action-expert attention (36 layers) | 36 × (768×2048 + 2 × 768×256 + 2048×768) | ≈ 0.13 B |
| Action-expert MoE FFN (36 layers) | 36 × (32 × 3 × 768 × 512 + 3 × 768 × 704) | 1.42 B |
| I/O projections and time MLP | 55×768 ×2 + 768×55 + 1536×768 + 768×768 | ≈ 0.002 B |
| Vision tower and deepstack | Qwen3-VL vision side | ≈ 0.8 B |
| **Total** | | **≈ 6.35 B** |

The ≈0.03B gap versus the exact 6.376B comes from RMSNorm weights, router gates, and various biases.

That 1.42B MoE FFN is by far the largest single addition. As a contrast, had those 36 layers stayed dense (width 2752), they would be 0.228B — **the MoE conversion spends 1.19B extra parameters to buy 6.2× the FFN capacity**.

<div class="keybox">
<strong>How to read activated parameters:</strong> counted per token, the MoE portion only runs Top-4 plus the shared expert, so activated width equals "4 × 512 + 704," the same as the dense version. The model therefore has <strong>6.38B total parameters and an activated scale also on the order of 6.38B</strong> — it is not the kind of sparse giant with "100B total, 5B active." Here MoE does not save compute; it <strong>grows capacity at fixed compute</strong>, the opposite of what MiniMax does with MoE in fa3 and fa10.
</div>

## 6. Training recipe

The `train` block of `real_robot.yaml` is dense with detail; the notable entries:

| Field | Value | Note |
|---|---|---|
| `optimizer` | `muon` | not AdamW |
| `lr` | 5e-5, `constant` | no decay |
| `micro_batch_size` / `global_batch_size` | 32 / 256 | 8 gradient-accumulation steps |
| `max_steps` | 60,000 | |
| `vlm_causal` | **true** | backbone switched to causal attention (standard for VLA) |
| `attention_implementation` | `flex_cached` | FlexAttention with KV cache |
| `enable_fp32` | **true** | FP32 weights, matching the HF F32 count |
| `data_parallel_mode` | `fsdp2` | |
| `tokenizer_max_length` | 72 | instruction length cap |
| `freeze_vision_encoder` | false | vision tower trains |

### Distributed Muon: 37.6% faster training

The README reports single-node measurements on H20:

| Optimizer | Clean | Randomized | Iteration time |
|---|---|---|---|
| Muon | 93.52 | 92.80 | 5.93 s/it |
| Distributed Muon | 91.56 | 91.34 | **4.31 s/it** |

That is **37.6%** faster (5.93 → 4.31) at the cost of about 2 points of success rate. The config for the comparison lives in `robotwin_dist_muon.yaml` — publishing the trade-off openly is more useful than reporting only the best number.

Distributed Muon adapts TorchTitan's FlexShard implementation and requires FSDP2 with at least two data-parallel ranks.

### Pretraining data

- About **60,000 hours**, of which **50,000 hours are robot data spanning 20 embodiments**
- **10,000 hours of first-person video** (human manipulation viewpoints)

Twenty embodiments across a unified 55-dimensional action space — that is the data foundation for the "one brain, many bodies" story.

## 7. 55 dimensions: a unified cross-embodiment interface

This is the most engineering-heavy part of the design. Robot degrees of freedom vary wildly — a 6-axis single arm, a 14-axis dual arm, plus 12 more for dexterous hands, plus 3 for a mobile base. LingBot-VLA 2.0's answer is to **define a space large enough for all of them and zero-pad the unused dimensions**:

| Segment | Dims | Content |
|---|---|---|
| Arm joint positions | 14 | dual arm, 7 axes each |
| End-effector poses | 14 | dual arm, 7 dims each (3 position + 4 orientation) |
| Grippers | 2 | two-finger open/close |
| Dexterous hand joints | 12 | 6 axes per hand |
| Waist | 4 | |
| Head | 2 | |
| Base / mobility signal | 3 | |
| Reserved | 4 | |
| **Total** | **55** | |

Three config fields are all pinned to 55:

```yaml
action_dim: 55
max_action_dim: 55
max_state_dim: 55
```

Every embodiment then shares one pair of I/O projections (`state_proj: 55 → 768`, `action_out_proj: 768 → 55`), fully reusing weights. The cost is waste on simple bodies — a 7-axis arm leaves most dimensions empty — but compared with maintaining per-embodiment projections, the waste buys cross-embodiment knowledge sharing, which is worth more.

<div class="keybox">
<strong>Contrast with π0:</strong> π0 uses 14 dimensions (7 axes per arm), and π0.5 stays in that range. LingBot-VLA 2.0 jumps straight to 55, packing in dexterous hands, waist, head, and base. This is the "one brain, many bodies" narrative expressed concretely in the data structure: <strong>the interface must be wide enough to hold differently shaped bodies.</strong>
</div>

## 8. Deployment: generate 50 steps, execute 25

This section addresses the easiest point to confuse.

The README's open-loop evaluation command uses `--use_length 50`, while real-robot deployment uses `--use_length 25`. These are not two different chunk lengths — it is **the same 50-step chunk, truncated at execution time**:

```python
if self.use_length > 0:
    assert self.use_length <= unnormalized_actions[output_key].shape[1]
    unnormalized_actions[output_key] = unnormalized_actions[output_key][:, :self.use_length]
```

That is receding horizon control: **generate the full 50-step chunk, execute only the first 25, then re-observe and replan.** Prediction error grows for later actions, so truncating improves robustness.

`use_length` also governs replanning frequency:

```python
if ... or self.global_step % self.use_length == 0 or self.use_length == -1:
```

`use_length = -1` means replan every step. Open-loop evaluation uses 50 because it must compute error over the full trajectory offline.

So the answer to "is the chunk 50 or 25": **generation length 50, execution length 25 — no contradiction.**

## 9. Performance numbers

### Real robot · GM-100 bimanual manipulation (progress / success)

| Platform | GR00T N1.7 | π0.5 | LingBot-VLA-1.0 | LingBot-VLA 2.0 |
|---|---|---|---|---|
| AgileX Cobot Magic | 36.3 / 17.8 | 59.1 / 32.2 | 58.2 / 30.0 | **66.2 / 34.4** |
| Galaxea R1Pro | 16.4 / 5.6 | 27.4 / 8.9 | 32.7 / **15.6** | **34.6 / 15.6** |

### Real robot · long-horizon mobile manipulation

| Embodiment | Task | Setting | LingBot-VLA 2.0 | π0.5 |
|---|---|---|---|---|
| Astribot S1 | Refrigerator sorting | in-domain | **77.1 / 60.0** | 65.3 / 46.7 |
| Astribot S1 | Refrigerator sorting | out-of-domain | **37.0 / 13.3** | 30.3 / 6.7 |
| Cobot Magic-ARX X5 | Stove cleaning | in-domain | **84.3 / 66.7** | 79.9 / 60.0 |
| Cobot Magic-ARX X5 | Stove cleaning | out-of-domain | **67.5 / 40.0** | 62.5 / 33.3 |

### Simulation · RoboTwin 2.0 success rate

| Task | π0.5 | LingBot-VLA-1.0 | LingBot-VLA 2.0 |
|---|---|---|---|
| Clean | 82.74% | 88.56% | **93.52%** |
| Randomized | 76.76% | 86.68% | **92.80%** |

<div class="warnbox">
<strong>Two caveats when reading these:</strong> first, all figures are <strong>vendor-reported</strong> with no independent reproduction; second, on Galaxea R1Pro the 2.0 success rate is identical to 1.0 (15.6) while only the progress score moves (32.7 → 34.6) — it does the task "more precisely" but not "more successfully." Do not mix progress and success.
</div>

Clean versus Randomized differs by only 0.72 points (93.52 vs 92.80). That is a remarkably small generalization gap, suggesting the 32 experts learned representations insensitive to visual perturbation rather than memorizing clean data.

## 10. Points of uncertainty

A few figures did not line up while writing this. Listing them is more useful than glossing over:

| Question | Observation | My read |
|---|---|---|
| `num_task_tokens` semantics | paper writes `Q_t`, `Q_{t+T}` | **8 tokens per group**, 16 total. The notation is conceptual, not a token count |
| Future-frame interval T | paper only says "future" | **not disclosed**. The yaml has `use_future_image: true` but no interval field |
| Dual-Query "two branches" | some write-ups describe "Understanding Branch / Action Branch" | paper's wording is **causal VLM + Action Expert** with per-layer joint attention, not two independent branches |
| `num_steps` | README says 10; deploy script has `# config.num_steps = 4` | **10 is in effect**; the comment does not run |
| Expert load distribution | no public expert-utilization curve | `bias_update_speed: 0` disables bias correction, leaving balancing to a `1e-3` per-sequence loss; actual balance unknown |
| 6.38B parameters | official "6B" is approximate | HF safetensors exact count **6,375,907,511**; "6B" is rounding |

## 11. Links to the rest of the series

- **vla3 (the π family)**: this model shares the same skeleton as π0 / π0.5 — dense VLM, narrow action expert, flow matching — differing in whether the expert's FFN is MoE-ified. Read vla3 first to see two efficiency routes on one architectural frame.
- **fa3 / fa10 (MiniMax M3 family)**: there, MoE exists to **save compute** (128 experts, Top-4, activated far below total). Here, MoE exists to **grow capacity at fixed compute**. One tool, two uses — that contrast is itself a footnote to MoE design space.
- **fa7 / fa8 (Qwen3.8 dual checkpoints)**: this model's backbone is Qwen3-VL-4B, so Qwen's dense architecture appears here as a fixed base — useful for comparing Qwen's width choices on the LLM and VLM sides.
- **fw9 (Wan2.2 dual experts)**: that post covers **diffusion-timestep routing** between two large experts (one high-noise 14B, one low-noise 14B). Here it is **token routing** among 32 small experts. Both are "experts," but one routes along the time axis and the other along the token axis.

<div class="keybox">
<strong>Closing thought:</strong> the MoE in LingBot-VLA 2.0 is a conservative but correct design — it does not chase an impressive parameter number, it first pins compute down with a width identity and then grows capacity 6.2×. For a model that has to run on a robot body, <strong>predictable inference latency matters far more than parameter count</strong>. Half of the 130 ms comes from KV reuse; the other half comes from that identity.
</div>
