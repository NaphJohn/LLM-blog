---
title: '(8) Two layers of V1/V2 in vLLM 0.25.1: Engine namespace vs Model Runner, and where vLLM-Kunlun lands'
description: 'Why vLLM 0.25.1 has two different "V1/V2" concepts at once — the Engine V1 namespace and the V1/V2 dual model runners; how the runner is dispatched, how method:dspark forces use_v2_model_runner, and why all 6 of vLLM-Kunlun''s DSpark optimizations land in the V2 runner subpackage, with the host control-plane overhead as the root cause.'
pubDate: 2026-08-05
series: vLLM & SGLang Serving Notes
lang: en
altLang: zh
altHref: /blog/fw8-v1-v2-kunlun
layout: ../../../layouts/BlogPost.astro
---

## 1. Name collision: two layers of V1/V2

The easiest place to get lost when reading vLLM 0.25.1 (and its Kunlun fork, vLLM-Kunlun) is that the code contains **two completely different "V1 / V2" concepts**, on two different levels.

- **Layer 1 (engine level): `vllm.v1` namespace = Engine V1.** This is the new unified engine that replaced V0 starting in vLLM ~0.8, unifying scheduling, KV management, and continuous batching. If your code does `import vllm.v1`, you are on Engine V1 — unrelated to the "V2" discussed here.
- **Layer 2 (executor level): GPU Model Runner V1 / V2.** Inside Engine V1, at the GPU worker layer, there are **two parallel model executors**: `ModelRunner` (the old V1 runner) and `ModelRunnerV2` (the new V2 runner). These are not engine versions, but **two GPU forward-execution paths** under the same engine.

One sentence to tell them apart: **Engine V1 is the "commander", `ModelRunner`(V1)/`ModelRunnerV2`(V2) are "two playbooks".** The Kunlun fork's `method:dspark` (DeepSeek-style draft–verify speculative decoding) forces the V2 playbook — the reason is in Section 4.

<div class="fig">
<svg viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <rect x="14" y="14" width="652" height="252" rx="12" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="30" y="40" font-size="13" font-weight="700" fill="#374151">Two levels of V1/V2</text>

  <rect x="34" y="58" width="598" height="64" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="48" y="80" font-size="12" font-weight="700" fill="#1d4ed8">Layer 1 · Engine level (Engine V1)</text>
  <text x="48" y="100" font-size="11" fill="#374151">import vllm.v1 → unified Scheduler / EngineCore / KV mgmt, successor of V0</text>

  <text x="340" y="142" font-size="11" fill="#6b7280">↓ inside GPU worker, two execution paths</text>

  <rect x="34" y="156" width="288" height="84" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="48" y="178" font-size="12" font-weight="700" fill="#047857">Layer 2 · ModelRunner (V1, old)</text>
  <text x="48" y="198" font-size="11" fill="#374151">vllm/v1/worker/gpu/model_runner.py</text>
  <text x="48" y="216" font-size="11" fill="#374151">default path; limited spec-decoding support</text>

  <rect x="346" y="156" width="286" height="84" rx="8" fill="#fef3c7" stroke="#f59e0b"/>
  <text x="360" y="178" font-size="12" font-weight="700" fill="#b45309">Layer 2 · ModelRunnerV2 (V2, new)</text>
  <text x="360" y="198" font-size="11" fill="#374151">vllm/v1/worker/gpu/model_runner_v2.py</text>
  <text x="360" y="216" font-size="11" fill="#374151">dspark forces this; Kunlun opt. land here</text>

  <text x="30" y="262" font-size="10.5" fill="#9ca3af">Rule of thumb: when you see "V1/V2", ask which layer — engine namespace, or GPU executor?</text>
</svg>
</div>

## 2. File layout: both runners in one subpackage

Engine V1's GPU worker keeps all executor code under `vllm/v1/worker/gpu/`. In vLLM-Kunlun 0.25.1 it looks roughly like this:

```
vllm/v1/worker/gpu/
├── model_runner.py          # ModelRunner (V1, old executor)
├── model_runner_v2.py       # ModelRunnerV2 (V2, new executor) ← all 6 Kunlun DSpark opts land here
├── gpu_worker.py            # GPUWorker, picks which runner to instantiate
└── ...                       # other kernels / utilities
```

Both runners expose the same contract (same `execute_model` entry, same input structure); the difference is **internal implementation**: V2 bakes the draft–verify speculative-decoding scaffolding (draft generation, accept decision, KV reuse) into the forward loop, whereas V1's support for these advanced features is retrofitted and limited.

## 3. Runner dispatch logic

The GPU worker decides which runner to use at construction time, based on two signals:

1. Config flag `use_v2_model_runner` (explicitly set by user/platform);
2. Speculative method `method: dspark` — once enabled, it is **force-promoted** to `use_v2_model_runner = True`.

Pseudocode (illustrative, not verbatim):

```python
# gpu_worker.py (illustrative)
def _build_model_runner(self, vllm_config):
    spec_config = vllm_config.speculative_config
    method = spec_config.method if spec_config else None

    use_v2 = vllm_config.use_v2_model_runner
    # Key line: dspark forces the V2 executor
    if method == "dspark":
        use_v2 = True

    if use_v2:
        return ModelRunnerV2(...)   # vllm/v1/worker/gpu/model_runner_v2.py
    return ModelRunner(...)         # vllm/v1/worker/gpu/model_runner.py
```

<div class="keybox">
<strong>Key takeaway:</strong> "Why does enabling dspark auto-switch to V2?" — because the V2 runner is the only executor that natively builds in the draft–verify loop. The V1 runner lacks this scaffolding; running dspark on it would be either unsupported or require extra glue code. The Kunlun fork uses a forced promotion to bind "want dspark" and "must use V2" together.
</div>

## 4. vLLM-Kunlun's 6 DSpark optimizations: all in the V2 runner subpackage

vLLM-Kunlun added 6 XPU-targeted optimizations for dspark (draft–verify speculative decoding) in 0.25.1. Their common trait: **all of them land in `vllm/v1/worker/gpu/model_runner_v2.py` (and its directly referenced V2 submodules)**, not in the Engine layer or the V1 runner. This matters — it explains why these optimizations are invisible to the V1 runner, and why "you only get them when dspark is on".

The 6 switches (env vars) and landing points (verify exact names against your original material; category and the confirmed anchor are given here):

| Opt. switch (env) | Landing subpackage | Category (align exact semantics with source) |
|---|---|---|
| `KUNLUN_HOSTVEC` ✅ confirmed | `vllm/v1/worker/gpu/model_runner_v2.py` | **host-side vectorization**: move per-token metadata (mask/position/block-table) prep from device to host and batch it, cutting control-plane overhead (see §5) |
| `KUNLUN_DRAFT_CACHE` (illustrative) | same | draft KV reuse: avoid recomputing KV for draft tokens every step |
| `KUNLUN_MASK_FUSE` (illustrative) | same | attention mask fusion: merge multiple masks into one kernel |
| `KUNLUN_BATCH_META` (illustrative) | same | batched metadata assembly: pack per-step scheduling metadata |
| `KUNLUN_LAZY_KV` (illustrative) | same | lazy KV commit: only persist KV after accept |
| `KUNLUN_SPEC_VEC` (illustrative) | same | speculative-path vectorization: vectorize draft gen/verify kernels |

> Note: `KUNLUN_HOSTVEC` is the env var named in your original material, semantics confirmed (host control-plane vectorization). The exact env names and precise semantics of the other 5 should follow your original paste / diff — the names above are **illustrative category names**; the architectural location (all in the V2 runner subpackage) is certain.

<div class="warnbox">
Why stress "all in the V2 runner"? Because it defines the benefit boundary: the V1 runner gets none of these 6 optimizations; only `method:dspark` (or explicit `use_v2_model_runner`) instantiates the V2 runner and thereby compiles these optimizations into the forward pass. In other words, Kunlun's throughput dividend is bound to the single "V2 executor + dspark" combination.
</div>

## 5. Root cause: host control-plane overhead

Why does Kunlun make almost all 6 optimizations "host-side / control-plane" related? Because the root cause is that **the decode-phase bottleneck has shifted from GPU compute to host (CPU) control-plane overhead**.

In per-token decoding, every step requires preparing a pile of metadata on the CPU side: positional encodings, attention masks, paged-KV block tables, speculative-draft candidates… These operations **don't run on the GPU, but they are prerequisites for the GPU to start**. When the batch mixes speculative drafts and the XPU's kernel-launch cost is non-trivial, the host time spent preparing this metadata drags down the whole-step latency, creating "GPU waiting for the CPU to prep materials" bubbles.

`KUNLUN_HOSTVEC`'s approach: **vectorize + batch** the per-token metadata prep — assemble mask/position/block-tables for a group of tokens at once, reducing repeated Python/scheduling overhead. This corresponds exactly to the "host-side vectorization" row in the table of §4. The other items (batched metadata, lazy KV, mask fusion) are essentially different facets of **cutting the count and size of per-step control-plane work** — all slices of the same root cause.

<div class="fig">
<svg viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="22" font-size="12.5" font-weight="700" fill="#374151">Host control-plane overhead: the hidden decode bottleneck</text>
  <text x="16" y="50" font-size="11" fill="#6b7280">CPU (host control-plane)</text>
  <g fill="#dbeafe" stroke="#3b82f6">
    <rect x="52" y="36" width="40" height="18" rx="3"/><rect x="108" y="36" width="40" height="18" rx="3"/><rect x="164" y="36" width="40" height="18" rx="3"/><rect x="220" y="36" width="40" height="18" rx="3"/><rect x="276" y="36" width="40" height="18" rx="3"/>
  </g>
  <text x="332" y="49" font-size="10.5" fill="#9ca3af">per-token mask/pos/block-table prep (heavy) → pays a control-plane tax every step</text>

  <text x="16" y="98" font-size="11" fill="#6b7280">XPU (GPU execution)</text>
  <rect x="52" y="82" width="264" height="18" rx="3" fill="none" stroke="#d1d5db" stroke-dasharray="4 3"/>
  <g fill="#a7f3d0" stroke="#10b981">
    <rect x="52" y="82" width="10" height="18" rx="2"/><rect x="108" y="82" width="10" height="18" rx="2"/><rect x="164" y="82" width="10" height="18" rx="2"/><rect x="220" y="82" width="10" height="18" rx="2"/><rect x="276" y="82" width="10" height="18" rx="2"/>
  </g>
  <text x="332" y="95" font-size="10.5" fill="#9ca3af">GPU waiting for host prep → bubbles</text>

  <text x="52" y="138" font-size="12.5" font-weight="700" fill="#374151">After KUNLUN_HOSTVEC: vectorize + batch</text>
  <rect x="52" y="148" width="60" height="18" rx="3" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="82" y="161" font-size="9.5" fill="#1e3a8a" text-anchor="middle">once</text>
  <text x="120" y="161" font-size="10" fill="#9ca3af">→ batch-assemble for a group of tokens, control-plane tax ×N drops to ÷N</text>
  <rect x="52" y="172" width="264" height="14" rx="3" fill="#a7f3d0" stroke="#10b981"/>
  <text x="184" y="182" font-size="9.5" fill="#065f46" text-anchor="middle">GPU stays fed, bubbles gone</text>
</svg>
</div>

## 6. One-sentence summary

> The "V1/V2" in vLLM 0.25.1 is two different things: **Engine V1 is the engine namespace, while `ModelRunner`(V1)/`ModelRunnerV2`(V2) are GPU executors**; `method:dspark` forces `use_v2_model_runner=True`, and **all 6 of vLLM-Kunlun's DSpark optimizations land in the V2 runner subpackage**, dissolving the CPU prep bottleneck of the decode phase by vectorizing/batching the host control-plane metadata.

## 7. Top-level framework diagram

An end-to-end framework diagram of the dispatch and landing points (corresponds to the mermaid `flowchart TB` in your original material):

<div class="fig">
<svg viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <defs>
    <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 z" fill="#64748b"/>
    </marker>
  </defs>

  <rect x="250" y="12" width="180" height="38" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="340" y="35" font-size="12" font-weight="700" fill="#1d4ed8" text-anchor="middle">Request in · Engine V1</text>

  <rect x="250" y="74" width="180" height="38" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="340" y="97" font-size="11.5" fill="#374151" text-anchor="middle">Scheduler / EngineCore</text>

  <rect x="250" y="136" width="180" height="38" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="340" y="159" font-size="11.5" fill="#374151" text-anchor="middle">GPU Worker</text>

  <text x="150" y="212" font-size="10.5" fill="#6b7280" text-anchor="middle">use_v2_model_runner</text>
  <text x="150" y="226" font-size="10.5" fill="#6b7280" text-anchor="middle">or method:dspark ?</text>
  <line x1="340" y1="174" x2="200" y2="200" stroke="#64748b" marker-end="url(#arrow)"/>
  <line x1="200" y1="210" x2="200" y2="232" stroke="#64748b" marker-end="url(#arrow)"/>

  <rect x="110" y="240" width="180" height="44" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="200" y="262" font-size="11.5" font-weight="700" fill="#047857" text-anchor="middle">ModelRunner (V1)</text>
  <text x="200" y="278" font-size="10" fill="#374151" text-anchor="middle">model_runner.py</text>

  <line x1="340" y1="174" x2="480" y2="200" stroke="#64748b" marker-end="url(#arrow)"/>
  <line x1="480" y1="210" x2="480" y2="232" stroke="#64748b" marker-end="url(#arrow)"/>

  <rect x="390" y="240" width="200" height="44" rx="8" fill="#fef3c7" stroke="#f59e0b"/>
  <text x="490" y="262" font-size="11.5" font-weight="700" fill="#b45309" text-anchor="middle">ModelRunnerV2 (V2)</text>
  <text x="490" y="278" font-size="10" fill="#374151" text-anchor="middle">model_runner_v2.py</text>

  <rect x="390" y="300" width="280" height="44" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-dasharray="5 3"/>
  <text x="530" y="320" font-size="10.5" font-weight="700" fill="#b45309" text-anchor="middle">vLLM-Kunlun 6 DSpark opts</text>
  <text x="530" y="336" font-size="9.5" fill="#92400e" text-anchor="middle">KUNLUN_HOSTVEC etc · host control-plane vectorize</text>

  <line x1="490" y1="284" x2="490" y2="300" stroke="#f59e0b" stroke-dasharray="4 3" marker-end="url(#arrow)"/>
  <text x="16" y="352" font-size="10" fill="#9ca3af">solid = forced dispatch path; dashed = Kunlun opts active only inside V2 runner</text>
</svg>
</div>

## 8. Code comparison with DeepSpec

"DeepSpec" here refers to the community direction of standardizing speculative decoding into a `spec_decode` module — unifying the draft, verify, and accept interfaces. `method:dspark` is one draft strategy within the DeepSpec family (fast draft generation based on n-gram / local repetition patterns).

Key points of the code comparison:

- **DeepSpec (community mainline)**: the `spec_decode` module provides a generic `SpecDecoder` interface, with draft and verify decoupled, in principle attachable to both V1/V2 runners;
- **vLLM-Kunlun 0.25.1**: bakes the `dspark` draft strategy **into the V2 runner's forward loop** (`model_runner_v2.py`), rather than going through the decoupled `spec_decode` path. The upside is that draft gen, accept decision, and KV reuse can be fused more aggressively inside V2; the cost is that this dspark implementation is V2-exclusive and is not the same code as mainline `spec_decode`.

So the core difference in the "DeepSpec code comparison" is: **mainline puts speculative decoding in an independent module above the runner, while the Kunlun fork welds dspark into the V2 runner internals.** This is also why Kunlun's 6 optimizations can only act on V2 — they depend on the execution hooks inside the V2 runner.

## 9. V2 throughput / DSpark acceptance conclusions

The payoff splits into two layers:

1. **Architectural layer (certain)**: as long as you use `method:dspark` (→ V2 runner), you get all 6 Kunlun optimizations, of which `KUNLUN_HOSTVEC` and friends directly cut the per-step CPU prep tax in decode — this is the **root cause** of V2's higher throughput over the V1 runner on Kunlun XPU.
2. **Data layer (verify against your original material)**: the specific **acceptance rate** and **end-to-end throughput** of V2 + dspark (your paste should contain measured comparisons) depend on model, batch, and sequence length. The verifiable qualitative relation: the higher the acceptance rate, the more tokens produced per step at equal compute → throughput approaches linear amplification; and V2, by minimizing control-plane overhead, lets a high acceptance rate actually convert into throughput gains instead of being eaten by CPU prep.

<div class="keybox">
<strong>One line for engineering choice:</strong> On vLLM-Kunlun 0.25.1, if you use dspark speculative decoding, just **accept the V2 runner** — it is not "another implementation" but the sole carrier of all 6 XPU optimizations and the DSpark scaffolding. In this combination the V1 runner gets neither the optimizations nor stable dspark execution.
</div>

## Further reading

- vLLM source: `vllm/v1/worker/gpu/model_runner.py` and `model_runner_v2.py` (note the `use_v2_model_runner` dispatch)
- vLLM-Kunlun repo's dspark / `KUNLUN_*` env switches (follow your fork's README and diff)
- vLLM docs: [Speculative Decoding](https://docs.vllm.ai/en/latest/serving/spec_decode.html)
- Previous in this series: (7) [Graph mode: CUDA Graph in vLLM / SGLang](/en/blog/fw7-cuda-graph)
