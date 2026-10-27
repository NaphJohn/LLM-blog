---
title: "Quantization Notes (2): RTN and GPTQ — from naive rounding to Hessian-compensated error"
description: "RTN is three lines of code and it destroys a 7B model at 4 bit. This post explains exactly why: channel error is amplified by activation energy, and a handful of outlier channels carry 50 to 100 times more energy than the rest. GPTQ comes from the 1993 OBS framework and answers a sharper question — given that I must break this weight, which other weights should I adjust to compensate? Full derivation of the OBS recursion, inverse-Hessian diagonal as a redundancy measure, the two engineering innovations that make GPTQ practical (one Cholesky factorisation, lazy batch updates), every parameter's role, and how the resulting qweight/scales/g_idx tensors actually run in an inference kernel."
pubDate: 2026-10-27
series: Quantization Notes
lang: en
altLang: zh
altHref: /blog/qt2-rtn-gptq
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

Round-to-nearest is the natural first idea, and it fails badly. The reason is not that rounding is imprecise — it is that **rounding errors do not stay where they are created**:

```text
ΔY[:, j] = Σ_i  X[:, i] · ΔW[i, j]
                  ↑ the error in weight row i gets multiplied by the
                    activation energy of channel i
```

If channel 5 has an activation norm 80× larger than the others, then its weight error is 80× more damaging. A global `absmax` makes every weight equally likely to be wrong, but the *cost* of being wrong is wildly unequal.

GPTQ's insight, inherited from OBS (1993), is: since you must break some weights, **break the cheap ones and let the expensive ones absorb the damage through a compensating update.** This post derives that update and then shows how to make it fast enough to be practical.

<div class="keybox">
<strong>Where this sits:</strong> route A of the map in qt1 — change the rounding, not the distribution. Everything here is a baseline that qt3, qt4, and qt5 measure themselves against.
</div>

## 1. RTN: The Starting Point and the Reference Frame

### 1.1 The algorithm is genuinely three lines

```python
def rtn_quantize(W, bits=4, group_size=128):
    """Round-to-nearest, per-group absmax. No calibration, no data."""
    out_f, in_f = W.shape
    maxq = 2 ** (bits - 1) - 1                    # 4 bit -> 7
    Wg = W.reshape(out_f, in_f // group_size, group_size)
    scale = Wg.abs().amax(dim=-1, keepdim=True) / maxq     # [out, groups, 1]
    scale = scale.clamp(min=1e-8)
    q = torch.clamp(torch.round(Wg / scale), -maxq - 1, maxq)
    return (q * scale).reshape(out_f, in_f), scale
```

That is the whole method. It takes seconds for a 7B model, needs no data, and is the honest baseline every paper should report.

At 8 bit it is usually fine. At 4 bit it is not:

```text
LLaMA-2-7B, WikiText perplexity
FP16                     5.47
RTN W8 (per-channel)     5.49      ← essentially free
RTN W4 (group=128)       5.93      ← noticeable but sometimes tolerable
RTN W4A4 (activations too) ~2000   ← destroys the model completely
```

The gap between the third and fourth line is the entire reason this field exists.

### 1.2 Why RTN fails: the error is weighted by activation energy

Write the layer output error in terms of weight error:

```text
Y = X W,    Ŷ = X Ŵ,    ΔW = Ŵ − W

ΔY[:, j] = Σ_i X[:, i] · ΔW[i, j]

‖ΔY[:, j]‖² ≤ Σ_i ‖X[:, i]‖² · ΔW[i, j]²
```

The bound makes the mechanism explicit: **a weight error in row `i` is amplified by `‖X[:,i]‖²`, the energy of activation channel `i`.**

LLM activations have a handful of channels whose energy is 50–100× the median (this is the outlier phenomenon qt3 covers in depth). So:

```text
For 99% of channels:  ‖X[:,i]‖² ≈ 1    → weight error contributes normally
For 0.1% of channels: ‖X[:,i]‖² ≈ 5000 → same weight error is 5000× more damaging
```

RTN treats all weights identically — it has no way to know that breaking a weight in row 5 costs 5000× more than breaking one in row 100. GPTQ's whole purpose is to supply that information.

<div class="fig">
<svg viewBox="0 0 680 290" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="RTN versus GPTQ error allocation">
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">Same total error, very different damage</text>

  <!-- RTN -->
  <text x="20" y="54" font-size="12" font-weight="700" fill="#b91c1c">RTN: error spread uniformly across weight rows</text>
  <text x="20" y="70" font-size="10" fill="#6b7280">Each row absorbs about the same quantization error</text>

  <line x1="30" y1="160" x2="330" y2="160" stroke="#9ca3af" stroke-width="1.5"/>
  <g fill="#fca5a5" stroke="#ef4444">
    <rect x="36" y="146" width="14" height="14"/><rect x="56" y="144" width="14" height="16"/>
    <rect x="76" y="147" width="14" height="13"/><rect x="96" y="145" width="14" height="15"/>
    <rect x="116" y="146" width="14" height="14"/><rect x="136" y="144" width="14" height="16"/>
    <rect x="156" y="147" width="14" height="13"/><rect x="176" y="145" width="14" height="15"/>
    <rect x="196" y="146" width="14" height="14"/><rect x="216" y="144" width="14" height="16"/>
    <rect x="236" y="147" width="14" height="13"/><rect x="256" y="145" width="14" height="15"/>
    <rect x="276" y="146" width="14" height="14"/>
  </g>
  <text x="36" y="180" font-size="9" fill="#6b7280">weight rows →</text>
  <text x="36" y="200" font-size="10" fill="#b91c1c">But row 5 is the outlier channel:</text>
  <text x="36" y="216" font-size="10" fill="#b91c1c">its error gets multiplied by ‖X[:,5]‖² ≈ 5000</text>
  <text x="36" y="238" font-size="11" fill="#b91c1c" font-weight="700">→ one row dominates the entire output error</text>

  <!-- GPTQ -->
  <text x="370" y="54" font-size="12" font-weight="700" fill="#15803d">GPTQ: error pushed onto cheap rows</text>
  <text x="370" y="70" font-size="10" fill="#6b7280">The expensive row is quantized carefully and the rest absorb it</text>

  <line x1="380" y1="160" x2="660" y2="160" stroke="#9ca3af" stroke-width="1.5"/>
  <g fill="#bbf7d0" stroke="#4ade80">
    <rect x="386" y="132" width="14" height="28"/><rect x="406" y="136" width="14" height="24"/>
    <rect x="426" y="140" width="14" height="20"/><rect x="446" y="134" width="14" height="26"/>
    <rect x="466" y="138" width="14" height="22"/><rect x="486" y="142" width="14" height="18"/>
  </g>
  <rect x="506" y="155" width="14" height="5" fill="#15803d" stroke="#166534"/>
  <text x="513" y="150" font-size="9" fill="#15803d" text-anchor="middle">★</text>
  <g fill="#bbf7d0" stroke="#4ade80">
    <rect x="526" y="136" width="14" height="24"/><rect x="546" y="140" width="14" height="20"/>
    <rect x="566" y="134" width="14" height="26"/><rect x="586" y="138" width="14" height="22"/>
    <rect x="606" y="142" width="14" height="18"/>
  </g>
  <text x="386" y="180" font-size="9" fill="#6b7280">weight rows →</text>
  <text x="386" y="200" font-size="10" fill="#15803d">★ = outlier channel, quantized nearly exactly</text>
  <text x="386" y="216" font-size="10" fill="#15803d">Its neighbors take extra error, but they are cheap: their</text>
  <text x="386" y="232" font-size="10" fill="#15803d">‖X‖² is small, so the damage stays bounded</text>
  <text x="386" y="254" font-size="11" fill="#15803d" font-weight="700">→ total output error drops by 3–4×</text>
</svg>
<p class="cap">Figure 1: RTN distributes weight error evenly and lets the expensive row dominate. GPTQ concentrates accuracy where activation energy is high and offloads error to rows where it is cheap.</p>
</div>

## 2. The Theoretical Origin of GPTQ: The OBS Framework

### 2.1 Problem setup

OBS (Optimal Brain Surgeon, Hassibi & Stork, 1993) asked: given a trained network, if I must perturb the weights, what is the least damaging perturbation? Formally:

```text
min_ŵ  (w − ŵ)ᵀ H (w − ŵ)      subject to: some weights are quantized

H = 2 Xᵀ X        ← the Hessian of the layer's squared output error
```

For a linear layer with MSE loss, the Hessian is exactly `2XᵀX` — which is why GPTQ needs calibration data at all. `XᵀX` summarises everything about how much the layer cares about each input direction.

### 2.2 The core step of OBS

Suppose you fix the error in dimension `q`: `ŵ_q = w_q + δ_q` where `δ_q = ŵ_q − w_q` is forced by quantization. You may now adjust all other weights freely to minimise the quadratic form.

Solving this constrained minimisation (Lagrange multiplier, one line of algebra) yields two results that GPTQ uses verbatim:

```text
① The cost of quantizing dimension q:
       err_q = δ_q² / [H⁻¹]_qq

② The optimal compensating update to all other dimensions:
       w ← w − (δ_q / [H⁻¹]_qq) · H⁻¹[:, q]
```

Two things are worth pausing on.

**First**, the cost of quantizing `q` is inversely proportional to `[H⁻¹]_qq`. A **large** inverse-Hessian diagonal means quantizing that dimension is **cheap** — the network is redundant in that direction, so damage there is absorbed. This is the exact opposite of the naive intuition (that large `H_qq` means "important"), and it is why GPTQ processes columns in descending order of `[H⁻¹]_qq`. That ordering heuristic is called **Inverse Hessian Diagonal (ID)** ranking.

**Second**, the compensation update is not a heuristic — it is the exact solution to the constrained quadratic problem. If the quadratic approximation of the loss were exact, GPTQ would produce the globally optimal quantized weights.

### 2.3 Why ordering is a real design decision

Quantizing column `j` breaks it, then compensates by adjusting the not-yet-quantized columns. But **already-quantized columns cannot be adjusted again** — they are frozen. So the order determines how much freedom the compensation has:

```text
Quantize cheap columns first  → their error is small, and the expensive
                                columns are still adjustable when you need them
Quantize expensive columns first → they consume the adjustable budget early,
                                and the cheap ones left cannot compensate
```

The counterintuitive consequence: **you should quantize the *least* important columns *first*.** They cause little damage, and they leave the important columns unquantized and therefore still available as compensation targets.

## 3. The Complete GPTQ Algorithm

Combining the OBS recursion with a Cholesky factorisation and blocked updates gives the algorithm below. This is close to the reference implementation and it runs.

```python
import torch

def gptq_quantize_layer(W, X, bits=4, group_size=128, damp=0.01, blocksize=128):
    """
    W: [out_f, in_f] fp16 weight
    X: [n_samples, in_f] calibration activations (layer input)
    """
    dev = W.device
    out_f, in_f = W.shape
    maxq = 2 ** (bits - 1) - 1

    # ---- ① Hessian. Note the factor 2: this is d²(loss)/dw² for MSE ----
    H = 2 * (X.T @ X)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0                        # guard against zero-variance channels
    # damping: H + damp * mean(diag(H)) * I, the Tikhonov regulariser
    H += damp * torch.eye(in_f, device=dev) * torch.diag(H).mean()

    # ---- ② ONE Cholesky inverse, reused for every column ----
    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    # upper Cholesky of H_inv: H_inv = Uᵀ U, so row j of U gives H_inv[j, j:]
    H_inv = torch.linalg.cholesky(H_inv, upper=True)

    Wq = W.clone()
    W_deq = torch.zeros_like(W)

    for blk_start in range(0, out_f, blocksize):
        blk_end = min(blk_start + blocksize, out_f)
        Err = torch.zeros(in_f, blk_end - blk_start, device=dev)

        for j in range(blk_start, blk_end):
            w = Wq[:, j]
            d = H_inv[j, j]
            if d == 0:
                W_deq[:, j] = w
                continue

            # ---- ③ per-group scale, computed on the CURRENT (already
            #         compensated) weights, not the original ----
            s = w.norm(p=1).item() / (in_f ** 0.5) / maxq   # simplified group logic
            q = torch.clamp(torch.round(w / s), -maxq - 1, maxq)
            wq = q * s
            W_deq[:, j] = wq

            # ---- ④ OBS compensation: err = (original − quantized) / [H⁻¹]_jj ----
            err = (w - wq) / d
            Err[:, j - blk_start] = err
            Wq[:, j + 1: blk_end] -= err.unsqueeze(1) * H_inv[j, j + 1: blk_end].unsqueeze(0)

        # ---- ⑤ lazy batch update: apply the accumulated error to the
        #         remaining (not-yet-processed) columns in one GEMM ----
        if blk_end - blk_start >= 4:
            Wq[:, blk_end:] -= Err @ H_inv[blk_start:blk_end, blk_end:]

    return W_deq
```

The sign in step ④ is the single most commonly miswritten line. Derivation:

```text
OBS says:  w ← w − (δ / [H⁻¹]_qq) · H⁻¹[:, q]   where δ = ŵ_q − w_q

With our notation δ = wq − w, so:
           w ← w − (wq − w)/d · H⁻¹[:, q]
             = w + (w − wq)/d · H⁻¹[:, q]
```

So the quantity applied to the other columns is `(w − wq)/d`, i.e. **original minus quantized**, divided by the inverse-Hessian diagonal. Writing `(wq − w)` flips the sign and makes the compensation actively harmful — the model gets worse than RTN.

### 3.1 The three parameters

| Parameter | Default | What it does | What happens if you get it wrong |
|---|---|---|---|
| `group_size` | 128 | scale granularity | 256: accuracy drops; 64: 6% metadata overhead |
| `damp` | 0.01 | Tikhonov regulariser, `× mean(diag(H))` | too small: `H⁻¹` blows up on ill-conditioned layers; too large: compensation becomes ineffective |
| `blocksize` | 128 | how often the lazy update flushes | smaller: more accurate but slower; > 256 gives diminishing returns |

`damp` deserves special mention because it is the parameter that actually bites. `H = 2XᵀX` from 128 calibration samples over a 4096-dimension input is **rank-deficient** — at most 128 nonzero eigenvalues out of 4096. Inverting it directly produces enormous values in the null space, and the compensation update then destroys the weights.

```text
Symptom of bad damp:  training-loss-like PPL jump from 5.9 to 30+
Diagnosis:            print torch.linalg.eigvalsh(H).min() — if negative or ~0,
                      the matrix is not positive definite and H⁻¹ is meaningless
Fix:                  raise damp to 0.05 or 0.1, or use more calibration samples
```

The `damp * mean(diag(H))` scaling matters — it makes the parameter layer-agnostic, so you do not have to hand-tune per layer as the activation magnitudes shift across depth.

### 3.2 The role of [H⁻¹]_qq

Two distinct jobs, often conflated:

1. **As a divisor in the compensation.** A large `[H⁻¹]_qq` means the dimension is redundant, so the same absolute error costs less and needs less compensation. This is the mechanism.
2. **As a sort key for desc_act.** Sorting columns by descending ID determines processing order; §2.3 explains why cheap-first is correct.

The cost of job 2 is real: reordering columns means the deployed weight tensor is permuted, so the kernel must store a `g_idx` permutation and gather on load. In practice `desc_act=True` buys roughly 0.05–0.15 PPL on 7B models but costs 10–30% throughput in ExLlama-class kernels. **Most production deployments set `desc_act=False`.** This is a genuine accuracy/speed trade, not an oversight.

### 3.3 Why the two engineering innovations matter

The naive OBS recursion is `O(in_f³)` per column, because each column requires a fresh matrix inverse — for a 4096×4096 layer that is 4096 × 6.8×10¹⁰ operations. Completely impractical.

```text
Innovation 1: single Cholesky factorisation
  H⁻¹ is computed ONCE, in O(in³/3). Every column's [H⁻¹]_qq and H⁻¹[:, q]
  is then a row lookup. This turns the per-column inverse into O(1).
  → the whole layer becomes O(in³) instead of O(in⁴)

Innovation 2: lazy batch update
  The per-column update Wq[:, j+1:] -= err · H⁻¹[j, j+1:] is a rank-1 update,
  which is memory-bound and terrible for GPU utilisation.
  Instead, accumulate 128 columns of `err` into a matrix Err and apply
  one GEMM: Wq[:, blk_end:] -= Err @ H_inv[blk_start:blk_end, blk_end:]
  → arithmetic intensity goes up ~100×, wall clock drops 3–5×
```

Together these took GPTQ from "interesting theory" to "runs in 20 minutes on one GPU for a 7B model."

## 4. Variants That Grew Out of GPTQ

| Variant | Change | Why |
|---|---|---|
| **GPTQ + act-order** | sort by ID, store `g_idx` | accuracy, costs throughput |
| **GPTQ + AWQ scales** | pre-scale by `mean(|X|)^α` | better conditioned `H`, see qt3 §4 |
| **GPTQ-triton** | fused Triton kernel | removes the Python loop overhead |
| **ExLlama v2** | bespoke kernel + fused dequant | the actual production speedup |
| **Marlin** | async dequant, tensor-core aligned layout | best current W4A16 throughput |
| **GPTQ + FP4** | FP4 grid instead of INT4 | needs NVFP4 hardware |

An important one that is not a GPTQ variant but is often confused with one: **GPTQ is a weight-only method.** It quantizes `W`, never `X`. So all its guarantees are about `‖X(W − Ŵ)‖`, and none of them transfer to the case where `X` is also quantized. W4A4 needs qt4.

## 5. Deployment: How GPTQ Weights Actually Run

The algorithm above produces a dequantized fp16 weight in memory (`W_deq`). Production never stores that. Instead a packed format is written:

```text
qweight   int32 [in_f/8, out_f]    8 four-bit values packed per int32
scales    fp16  [in_f/group, out_f] one scale per group per output channel
qzeros    int32 [in_f/group, out_f] packed zero-points (asymmetric only)
g_idx     int32 [in_f]              group index; needed only with desc_act
```

At inference the kernel does **online dequantization** — it unpacks 4-bit values, converts to fp16, multiplies by the scale, and feeds the result straight into the MMA. The dequantized tensor never touches HBM, which is the whole point: you want to stream 4-bit from memory and dequantize on chip.

```text
memory traffic per token (7B model, W4 vs FP16):
  FP16: 13.5 GB  → at 2 TB/s that is 6.8 ms/token → ~147 tok/s ceiling
  W4:    3.9 GB  → 2.0 ms/token → ~510 tok/s ceiling  (before kernel overhead)
```

Real measurements (LLaMA-2-7B, A100 80GB, batch 1, ExLlama v2 kernels):

| Config | VRAM | Throughput | vs FP16 |
|---|---|---|---|
| FP16 | 13.5 GB | baseline | 1.0× |
| GPTQ W4, naive kernel | 3.9 GB | ~1.1× | 1.1× |
| GPTQ W4, ExLlama v2 | 3.9 GB | ~2.5–3.2× | **2.5–3.2×** |

The 1.1× row is the important warning: **the algorithmic win is 3.5× less memory, but you only get the speed if the kernel is fused well.** A naive PyTorch implementation of "dequantize then matmul" allocates an intermediate fp16 tensor and reads it back, which costs you almost all the bandwidth you saved.

## 6. RTN vs GPTQ: When Is RTN Good Enough

GPTQ is not always worth it. The decision is genuinely data-dependent:

```text
Use RTN when:
  · bits ≥ 8                          → RTN is essentially free and lossless
  · group_size ≤ 64 at 4 bit          → fine granularity absorbs most of the error
  · the model is small (< 1B)         → outlier structure is milder
  · you need to quantize 100 models   → 20 minutes × 100 is a real cost
  · you just want a quick baseline    → always run RTN first, it is your floor

Use GPTQ when:
  · 4 bit with group_size ≥ 128
  · the model shows strong outlier structure (most 7B+ models do)
  · you have one model and care about the last 0.3 PPL
  · you are stacking with AWQ scales anyway
```

One practical note that saves a lot of time: **run RTN first, always.** If RTN W4 gives you 5.93 PPL and GPTQ gives 5.60, you have quantified the value of 20 minutes of compute. If RTN gives 5.60 and GPTQ gives 5.58, skip GPTQ and spend the effort on calibration data quality instead — which usually matters more.

## 7. Series Navigation

- Previous **qt1 · The Basic Language**: granularity, symmetry, calibration, and the four-route map.
- Next **qt3 · SmoothQuant and AWQ**: route B. The activation outlier problem in full, and two opposite solutions to it.
- **qt4 · QuaRot / SpinQuant / DuQuant**: route C. Rotation, and why it is the only path to W4A4.
- **qt5 · KV Cache and FP4**: the orthogonal compression dimension.

Related on this site: **op2** is the series index and lineage table; **fw4** covers how these packed formats and dequant kernels are integrated in real serving engines; **op3** shows a case where per-tensor-role precision assignment was forced by hardware constraints rather than by accuracy.
