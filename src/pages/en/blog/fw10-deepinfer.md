---
title: "(10) DeepInfer Dissected: What DeepSeek Handed Over, and What It Kept"
description: "A breakdown of DeepInfer's four layers: operator, parallelism, decoding, engineering. Attention DP plus Wide-EP with EPLB greedy redundancy, DSpark confidence-scheduled verification dropping verified drafts from 5.0 to 3.2-4.4, SWA Bounded Replay with 128-aligned cache hit rules, the zero-overhead batch scheduler, DeferredRelease, and the three pieces deliberately left out: KV management, PD disaggregation, and replica load balancing."
pubDate: 2026-10-27
series: vLLM & SGLang Serving Notes
lang: en
altLang: zh
altHref: /blog/fw10-deepinfer
layout: ../../../layouts/BlogPost.astro
---

## 0. Why this deserves its own post

On 20 October 2026, DeepSeek open-sourced DeepInfer, the inference engine that powers large-scale production serving for DeepSeek-V4.1-Flash. It shipped alongside `deepseek-recipe`, a production-grade token encode-decode component.

The weight of this release is not that another framework went open. It is that what opened is the **middle layer**. A model sets its own capability boundary, but serving a very large user base at a sane cost depends entirely on the inference framework underneath. DeepSeek had already given away the model layer (the DeepSeek family) and the operator layer (FlashMLA, DeepGEMM, DeepEP, TileLang). This closes the last gap.

Two numbers came with it, an unusually candid look at the books. **A single cluster serves 155 trillion tokens per day, and revenue at off-peak pricing is 38 USD per GPU-hour.** That is revenue, not profit, and 155T is the official cluster-throughput figure; the material does not break out input, output, and cache-hit proportions, so you cannot reconstruct the real incremental compute from it.

So far this site has only mentioned DeepInfer in scattered `aihot` dailies (five issues, 10-22 through 10-26), all from the angle of how the community caught up within a week. This post takes the opposite angle. **Take DeepInfer's own design apart**, and separate real engineering from the inevitable coupling to internal infrastructure, and from the model-structure advantages nobody else can copy.

The one-sentence conclusion up front. **The thing worth learning from DeepInfer is not any single kernel. It is co-design between model structure and the cache system: 128 is the same number in the model config, the cache block size, the replay window, and the hit rule, because they were chosen together.**

## 1. What opened, and where it sits in the stack

You have to separate three layers first, otherwise you end up with the lazy verdict that the release is incomplete.

<div class="fig">
<svg viewBox="0 0 680 330" width="100%" font-size="13" font-weight="400" role="img" aria-label="Where DeepInfer sits in the inference stack">
<title>Figure 1: Three layers, and DeepInfer is the middle one</title>
<defs>
<marker id="a1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<rect x="60" y="40" width="560" height="60" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="340" y="62" font-size="13" fill="#047857" font-weight="600" text-anchor="middle" dominant-baseline="central">Layer 3 - Cluster service (design partly published, code not fully released)</text>
<text x="340" y="82" font-size="11" fill="#1a1a1a" text-anchor="middle" dominant-baseline="central">Request routing - cross-replica load balancing - KV placement - failure rerouting - autoscaling</text>

<rect x="60" y="120" width="560" height="70" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.2"/>
<text x="340" y="144" font-size="13" fill="#1d4ed8" font-weight="600" text-anchor="middle" dominant-baseline="central">Layer 2 - The DeepInfer engine (the actual release)</text>
<text x="340" y="164" font-size="11" fill="#1a1a1a" text-anchor="middle" dominant-baseline="central">Accept request - build batches - manage in-flight sequences and cache blocks - run model - return</text>
<text x="340" y="180" font-size="11" fill="#2563eb" text-anchor="middle" dominant-baseline="central">It threads local operators into one continuously executing chain</text>

<rect x="60" y="210" width="560" height="60" rx="8" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="340" y="232" font-size="13" fill="#1a1a1a" font-weight="600" text-anchor="middle" dominant-baseline="central">Layer 1 - Operators and communication (previously loose, now packaged and wired together)</text>
<text x="340" y="252" font-size="11" fill="#6b7280" text-anchor="middle" dominant-baseline="central">FlashMLA (attention) - DeepGEMM (matmul) - DeepEP (expert comms) - TileLang kernels</text>

<rect x="60" y="292" width="270" height="30" rx="6" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="195" y="307" font-size="11" fill="#854f0b" text-anchor="middle" dominant-baseline="central">Open: offline core, LLM.generate()</text>
<rect x="350" y="292" width="270" height="30" rx="6" fill="#fef2f2" stroke="#ef4444" stroke-width="0.5"/>
<text x="485" y="307" font-size="11" fill="#b91c1c" text-anchor="middle" dominant-baseline="central">Not open: full KV, PD split, replica LB</text>
</svg>
</div>

The key distinction. **No complete KV cache system does not mean the engine has no cache.** The engine obviously allocates and reclaims cache blocks, otherwise a single instance could not run. What is missing is cross-instance KV placement and transfer. Likewise, publishing EPLB (expert load balancing) does not mean cross-instance request routing shipped with it. Different layers, different problems.

The official README is refreshingly honest about lineage. DeepInfer started out mainly on top of vLLM, borrowing ideas from SGLang and TensorRT-LLM, and the entry and engine files still carry derivation markers. But the current implementation has C++ scheduling and resource management, a dedicated model execution path, and optimizations tied to DeepSeek model structure. **Calling it a thin wrapper misses the parts that actually do the executing. Calling it unrelated to existing frameworks does not match the source or the official notes.**

`deepseek-recipe` is the other half, and it is widely underrated. It is not a tokenizer wrapper but a set of Rust libraries with Python bindings acting as an **API compatibility front door**. It normalizes OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages into one internal Conversation structure, renders the prompt per model version, encodes it, and parses the output back into the target shape as a stream. It also distinguishes plain text from real control tokens at encode time, so a literal chat delimiter appearing in user input stays as ordinary text. Model inference, tool execution, and HTTP transport are yours to supply.

## 2. Four layers, one table

<div class="fig">
<svg viewBox="0 0 680 300" width="100%" font-size="13" font-weight="400" role="img" aria-label="The four layers of DeepInfer">
<title>Figure 2: DeepInfer bottom-up in four layers</title>
<rect x="40" y="36" width="600" height="54" rx="8" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="60" y="56" font-size="12" fill="#1a1a1a" font-weight="600">Operator layer</text>
<text x="60" y="76" font-size="11" fill="#6b7280">FlashMLA - DeepGEMM - DeepEP - TileKernels - DeepSelect plus custom TileLang kernels</text>

<rect x="40" y="98" width="600" height="54" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="0.5"/>
<text x="60" y="118" font-size="12" fill="#1a1a1a" font-weight="600">Parallelism layer</text>
<text x="60" y="138" font-size="11" fill="#2563eb">Attention DP / balanced CP plus MoE EP plus EPLB expert-parallel load balancing</text>

<rect x="40" y="160" width="600" height="54" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="60" y="180" font-size="12" fill="#1a1a1a" font-weight="600">Decoding layer</text>
<text x="60" y="200" font-size="11" fill="#047857">DSpark speculative decoding with confidence-scheduled verification, plus SWA Bounded Replay</text>

<rect x="40" y="222" width="600" height="54" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="60" y="242" font-size="12" fill="#1a1a1a" font-weight="600">Engineering layer</text>
<text x="60" y="262" font-size="11" fill="#854f0b">Async scheduling (zero-overhead batch scheduler), structured output (Regex / JSON Schema / EBNF), reproducibility switches</text>
</svg>
</div>

Of these four, **only the decoding layer is distinctly DeepSeek**. The rest is existing engineering pushed to its limit. Below, I split things into what you can copy and what you cannot.

## 3. Parallelism: prefill wants compute, decode wants bandwidth

V4.1-Flash is a MoE model with 384 routed experts. The two phases have completely different resource profiles. **Prefill processes tens of thousands of tokens at once and competes on compute. Decode emits one token at a time and spends most of its time reading weights and caches, so it competes on memory bandwidth.**

Hence completely different parallelism scales per phase.

| Phase | Strategy | Values | Why |
|---|---|---|---|
| Prefill | Attention DP + balanced CP + MoE EP | EP2-EP8 / DP2-DP8 | Input blocks are large, one batch saturates compute, no need for wide parallelism |
| Decode | Attention DP + MoE Wide-EP | EP64-EP128 / DP64-DP128 | The more spread out the experts, the less weight memory each MoE step reads, so a huge batch can amortize overhead |
| Production | P to D close to 1 to 1 | - | Smallest efficient unit is 64 GPUs prefill plus 64 GPUs decode |

**Wide-EP is really about spreading experts across more GPUs.** Take 64-GPU decode: 384 routed experts spread out means 6 per GPU, plus 1 replica for a hot expert, so 7 total. Fewer experts per GPU means fewer parameters to read from memory each step, and the same number of GPUs can absorb more requests.

The official offline benchmark (64 SM100 GPUs in one scale-up domain, fixed 3 draft tokens, effective output discounted at 75 percent per-token acceptance) has an interesting divergence:

- At 100 tok/s per user, EP64 delivers about 26.6k tok/s/GPU and EP8 about 25k, so **peak throughput is nearly identical**;
- At 250 tok/s per user, EP64 still holds about 15.5k while EP8 collapses to about 1.7k.

**So Wide-EP does not buy peak. It buys headroom in the high-interactivity regime.** This is easy to miss in capacity planning: if your workload is low-concurrency long-output, turning EP up only burns communication.

### 3.1 EPLB: greedily replicate the hot experts

MoE inference fears expert hotspots. Some experts get routed to saturation while other GPUs idle. Training uses an auxiliary loss for balance, but **at inference there is no loss, only measurement plus movement.**

EPLB assigns redundant replicas to hot logical experts based on historical token load, then places physical replicas on different GPUs. The objective is **minimax**: replicate whichever expert reduces the worst case across all steps the most, not simply the hottest one.

| Phase | Redundant experts per GPU (start) | Stats window | Rebalance period | Measured moe_lb |
|---|---|---|---|---|
| Prefill | 4 | 100 steps | 1000 steps | 1.074 (EP4DP4) / 1.045 (EP2DP2) |
| Decode | 1 | 1000 steps | 10000 steps | **1.090** |

`moe_lb` is slowest-rank load over average load, closer to 1 is better. **1.090 means the busiest GPU does only 9 percent more work than average**, which is remarkably tight across 64-way expert parallelism.

One engineering detail worth stealing. EPLB scores each expert using `tokens_per_expert` plus a tiny offset (`arange_logical` times 1e-4) so no two experts tie. **That 0.0001 perturbation buys reproducible scheduling**, avoiding different layers returning different results when argmax ties. Trading a sliver of optimality for determinism matters far more in production than one extra percent of throughput.

The cost is concrete too. EP64 means 64 GPUs form a single communication domain, and **one dead GPU takes the whole instance down**. Recovery is not a fast restart but recomputing the EP size from all currently healthy GPUs and limping along with the remainder. The safety net is spare capacity, not recovery speed.

## 4. Decoding part one: DSpark, generating 5 is not verifying 5

Our `ep4` post already covers the DSpark draft structure (semi-autoregressive drafting plus a Markov head). Here I add **how it lands on the scheduling side inside DeepInfer**.

Speculative decoding has a light drafter propose several candidates, then the target model verifies them in parallel. A fixed verification length is easy to implement, but draft quality varies a lot across requests and positions, and **sending late candidates that will almost certainly be rejected is pure wasted compute.**

DSpark adds a Confidence Head to the drafter, estimating acceptance probability per position. The scheduler then combines that with measured hardware throughput curves to decide how long a draft prefix each request gets verified, aiming at **effective output throughput**, not a bigger verification batch.

The critical logic: **you cannot threshold each position independently.** A later draft only helps if all earlier candidates were accepted. So the scheduler converts predictions into **cumulative prefix value**, estimates execution time for different verification totals, and picks the budget with the best expected effective output per unit time.

Measured gains depend heavily on concurrency, and this is the most misread table:

| Concurrent sequences per rank | Fixed verify 5 (tok/s) | DSpark variable (tok/s) | Relative gain |
|---|---|---|---|
| 16 | 4,891 | 4,816 | **-1.5%** |
| 32 | 8,964 | 9,111 | +1.6% |
| 48 | 12,032 | 12,467 | +3.6% |
| **60 (production level)** | 13,730 | 14,466 | **+5.4%** |
| 72 | 14,962 | 15,961 | +6.7% |
| 96 | 16,817 | 18,660 | +11.0% |

**At low load the gain is small or negative, and it grows with traffic.** DeepSeek estimates +5.4 percent at production level (about 60 sequences per rank) and +4 to 7 percent overall in production (also stated as about 6 percent). Mechanically, verified draft tokens per step drop from 5.0 to 3.2-4.4, **about 30 percent fewer**, while average accepted length stays roughly flat (3.29 versus 3.20).

> These numbers come from an official estimate under a specific EP64/DP64 configuration, not a universal DSpark speedup. They also measure **execution efficiency** discounted at 75 percent per-token acceptance, not draft quality on real tasks.

Two implementation details to remember:

1. Max draft length defaults to 5, but **variable-length verification is off by default** and needs `DEEPINFER_ENABLE_MTP_VARLEN=1`. Without a usable performance curve, the scheduler falls back to verifying all available drafts.
2. Under async scheduling the CPU may not have the latest confidence values, so **coarse budget selection happens on CPU and the GPU refines per-request lengths with fresher information**.

One absolute rule. **The Confidence Head only allocates verification resources. It must never be used to accept tokens that the target model did not verify.** That is the losslessness floor.

## 5. Decoding part two: SWA Bounded Replay, the densest section

This is the highest-density part of the post and the clearest demonstration of co-design. First, pin down the vocabulary:

- **Prefill and Decode** are two **runtime phases** (process input, continue generating);
- **CED encoder-decoder** refers to the two **internal compute halves** of V4.1-Flash (Causal Encoder-Decoder).

These are easy to conflate. The encoder here is **not** a vision encoder, and the decoder is **not** the decode phase.

### 5.1 The problem: a cache that lives three minutes held nearly half the persistent cache

Besides the main KV cache, V4.1-Flash keeps separate **sliding-window attention** (SWA) state. Every decoder layer has a local window looking at only the last 128 tokens. During prefill, the front half (encoder) builds historical representations and the back half (decoder) reuses part of that state, so **a single prefill pass also runs through the decoder layers**.

The trouble: decoder-layer global keys and values can be projected from the encoder final state, **but the SWA window state cannot.** It is per layer and projected from nothing. Reconstructing it exactly for a cached prefix means pushing the last 2,560 tokens (20 layers times 128) through 20 decoder layers.

The V4 approach persisted all 20 layers of 128-token state honestly. Result: **SWA consumed nearly half the persistent KV cache capacity, while its measured TTL in production was about 180 seconds** (minute-scale reuse within a session, discarded once the session ends).

> Using SSDs that retain data for days to store data with a three-minute lifetime is a textbook mismatch between storage tier and data lifetime.

### 5.2 The fix: no persistence, replay a bounded suffix

<div class="fig">
<svg viewBox="0 0 680 340" width="100%" font-size="13" font-weight="400" role="img" aria-label="Exact reconstruction versus bounded replay">
<title>Figure 3: Exact reconstruction versus bounded replay</title>
<defs>
<marker id="a2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<text x="40" y="34" font-size="12" font-weight="600" fill="#b91c1c">V4: exact reconstruction = 2560 tokens x 20 layers</text>
<rect x="40" y="46" width="520" height="26" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="0.5"/>
<text x="300" y="59" font-size="11" fill="#b91c1c" text-anchor="middle" dominant-baseline="central">History to recompute (20 layers x 128 = 2560 tokens)</text>
<rect x="560" y="46" width="80" height="26" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="600" y="59" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">new tokens</text>

<text x="40" y="130" font-size="12" font-weight="600" fill="#047857">V4.1: bounded replay = recompute only the last 128 (140 kept by default)</text>
<rect x="40" y="142" width="520" height="26" rx="4" fill="#f6f8fa" stroke="#6b7280" stroke-width="0.5"/>
<text x="300" y="155" font-size="11" fill="#6b7280" text-anchor="middle" dominant-baseline="central">Long history still travels another cache path, not recomputed</text>
<rect x="440" y="142" width="120" height="26" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1"/>
<text x="500" y="155" font-size="11" fill="#1d4ed8" text-anchor="middle" dominant-baseline="central">replay 128</text>
<rect x="560" y="142" width="80" height="26" rx="4" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="600" y="155" font-size="11" fill="#047857" text-anchor="middle" dominant-baseline="central">new tokens</text>

<rect x="40" y="196" width="600" height="52" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-width="0.5"/>
<text x="60" y="216" font-size="12" fill="#854f0b" font-weight="600">During replay the attention range is truncated to the replayed segment</text>
<text x="60" y="236" font-size="11" fill="#1a1a1a">So the result is a good-enough approximation, not a bit-equivalent restoration, and the cost becomes a constant independent of context length</text>

<rect x="40" y="262" width="600" height="56" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="0.5"/>
<text x="60" y="282" font-size="12" fill="#047857" font-weight="600">Payoff</text>
<text x="60" y="302" font-size="11" fill="#1a1a1a">Persistent KV footprint is 1/8 of V4-Flash (-87.5%), global KV at 890 B per token is 1/437 of V1, decoder-side prefill compute nearly halved</text>
</svg>
</div>

V4.1 instead goes **no persistence plus bounded replay**. SWA KV stays only in the memory of the machine that produced it (TTL about 180 seconds, hit rate above 98 percent). When it is missing, the engine recomputes only the last 128 tokens to rebuild an approximate state, with the attention window clamped to those 128.

**This is an approximation, not exact reconstruction.** Because the inter-layer receptive field exceeds 128, the replayed result deviates slightly from the original path, and DeepSeek aligns it by simulating bounded replay during post-training. This matters enormously: **you cannot bolt this onto any off-the-shelf model.** The window must match training, or quality drops.

Defaults are `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` and `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`. **Note that 140 is not 128 with a bigger window.** The retained suffix must cover the SWA window plus maximum backtracking; get it wrong and you break semantics.

`compact` mode genuinely deletes the query rows in the back half that do not need redoing, rather than masking a full computation. But `_store_yoco_full_kv()` still builds the required cache from the complete historical representation shared by the front half. **Computing fewer query positions does not mean the model forgets what came before.**

### 5.3 Why 128 shows up everywhere

This is the single paragraph I most want you to remember. Line up the constants:

| Constant | Value | Source |
|---|---|---|
| Sliding window | 128 tokens | Model config |
| KV cache block size | 128 tokens | Storage unit |
| SWA store positions | multiples of 128 | Stored once after prefill, once at end of decode |
| Draft block | 5 tokens | DSpark, `DEEPINFER_USE_NEXTN_DECODE` |
| Speculative backtracking | 6 tokens | Draft block plus one |
| Maximum hit length | L minus 134 | Window plus backtracking |
| Decoder replay | 140 tokens | Must be at least window plus backtracking |
| Stored SWA state per sequence | 20 layers x 134 x 528 B, about 1.4 MB | Encoder layers only, MXFP8 |
| Exact reconstruction cost | 2560 tokens x 20 layers | What replay avoids |

**The 128 in the model and the 128 in the storage system are the same number because they were chosen together. So were 5, 6, 134, and 140.** This is what serving-side co-design looks like: the model has a window cheap enough to store, and the cache has a block size the window fits into.

### 5.4 Hit rules: why a short follow-up costs as much as a long one

The rule itself: **the hit length is capped at L minus 134, rounded down to a 128-token block boundary, and backed off one more full block if no SWA state was stored at that boundary.**

Take two realistic requests. A conversation has accumulated 1,460 tokens (1,100 prompt plus 360 reply), and the previous turn left SWA state at two positions, 1,024 after its prefill and 1,408 at the end of decode.

| Case | Added | L | Effective hit | Encoder recompute | Decoder replay |
|---|---|---|---|---|---|
| A: 300-token tool result | 300 | 1760 | 1408 (SWA state present) | 352 | 140 |
| B: 40-token user message | 40 | 1500 | capped to 1280, **no SWA state**, back off one block to 1152 | 348 | 140 |

**Case B is the counterintuitive one. A 40-token follow-up pays almost the same prefill as a 300-token one**, because the L minus 134 cap pulls the hit below the last block boundary where SWA state exists, and the rule then backs off another block.

What do you get in exchange? **The system never stores decoder window state, never transfers it, and never pushes more than 140 tokens through the decoder at prefill.** A predictable context-independent constant replaces an entire class of cross-machine state synchronization problems, which for a very large cluster is an absurdly good trade.

This also explains why the block resists simple optimization. Any tweak that raises hit rate puts SWA state back into the cache system.

## 6. Engineering: the zero-overhead batch scheduler and DeferredRelease

The conventional flow is serial. CPU picks a batch, GPU executes, CPU takes results back for postprocessing, then prepares the next batch. **When the GPU finishes a batch it waits for the CPU to ready the next one, and that gap does not close itself.**

DeepInfer async scheduling overlaps the two. While the GPU computes step N, the CPU is already preparing step N+1 and postprocessing earlier rounds. The order in `worker.py` makes this concrete: **the worker submits the current forward pass, then does the previous round postprocessing, then submits the current round sampling.** The current round result may be returned only on the next call.

That creates two hard problems, and they are where the real engineering lives:

1. **Variable-length draft verification must tolerate lagging timing.** When preparing the next round, the previous sampling result may not be back on the CPU yet. `csrc/engine/zos.cpp` and the worker metadata-fixup path build a remap between adjacent batches, mapping this round sequences to the previous round device results, so the GPU corrects inputs and sequence lengths from real outcomes. Metadata that must return to CPU is handed over via pinned buffers and events rather than synchronizing the whole round first.
2. **Request finished is not the same moment as device no longer using its storage.** `DeferredRelease` handles resource lifetime after asynchrony: sequence metadata, output objects, and cache blocks pinned by clones must all be reclaimed at a safe point.

**It does not guarantee an idle-free device under all loads.** Low traffic, communication, or other resource bottlenecks still hurt utilization. What it improves is the waiting relationship along the execution chain. Gains are largest with small batches and fast decode steps, exactly the regime of Flash-class high-throughput models.

### 6.1 Batch-invariant and deterministic sampling

These matter most for enterprises, because they decide whether the engine can enter production at all (reconciliation, audit, regression testing):

- `DEEPINFER_BATCH_INVARIANT=1`: results for a single request are identical regardless of which requests share the batch, avoiding logits differences from floating-point reduction order;
- Deterministic sampling: on top of the above, set `DEEPINFER_ENABLE_MTP_VARLEN=0` and pass a fixed `seq_seed`, so identical inputs and sampling parameters yield identical output.

> **Limitation: both are currently NVIDIA-only. Ascend does not support them.** If you need reproducible regression on domestic accelerators, this is a hard gate.

## 7. What it deliberately withholds: three gaps and what each means

These are not unfinished work. They are **tightly coupled to internal infrastructure and cannot be extracted.**

| Gap | What is missing | What you must build |
|---|---|---|
| KV cache management | The full cross-instance cache system, on the 3FS side | A paged block cache pool of your own: roughly 100 PiB of SSD, peak aggregate read about 10 TiB/s and write about 850 GiB/s; 3-day retention for agent traffic and 7-day for chat, targeting 72 hours or more after inactivity |
| PD disaggregation | P/D split plus routing plus handoff | The README says most engine work is in place, but you build the split execution; see our `sys8` on ratio algorithms and `sys12` on the Mooncake approach |
| Replica load balancing | Multi-replica cluster routing and traffic balancing | Bound to internal cluster management, so you must wire up your own Kubernetes or gateway layer |

This also explains the underlying reason for the verdict in our `aihot` dailies. **vLLM and SGLang closed the speed gap within a week using the same family of kernels, and the real moat difference sits in these three engineering components**, which happen to be exactly where the open frameworks are strong. So do not treat DeepInfer as the only answer, not merely because it was a step slower, but because **the boundary of what it delivers never included a complete serving stack.**

As for the 3FS numbers (100 PiB, 10 TiB/s), they reveal something else. **DeepSeek cache policy rests on an assumption of near-unlimited SSD capacity.** Copying those TTL policies at small or medium scale is meaningless, because your SSD does not have 100 PiB and both hit rate and payback period are completely different.

## 8. What you can copy, and what you cannot

| Layer | Copyable | Not copyable |
|---|---|---|
| Operators | FlashMLA, DeepGEMM, DeepEP are open and already integrated into vLLM and SGLang | Custom kernels bound to V4.1-Flash structure |
| Parallelism | The idea of phase-specific Wide-EP sizing; EPLB minimax redundancy with the 1e-4 perturbation for determinism | The specific 384-expert plus supernode hardware values (EP64-128 is what DeepSeek computed for itself) |
| Decoding | The causal chain from Confidence Head to cumulative prefix value to variable verification budget | SWA Bounded Replay approximate reconstruction (the window must match training; bolting it onto existing models degrades quality) |
| Engineering | The async ordering of submit forward, then previous postprocess, then submit sampling | 3FS-scale cache assumptions; PD and cross-instance LB you build yourself |

## 9. Ties to other posts on this site

- **`fw1` (why an inference engine)** names three walls: memory fragmentation, static batching, GPU idling. DeepInfer answers all three anew with paged block caching, Wide-EP huge batches, and a zero-overhead scheduler.
- **`fw2` / `fw3` (vLLM and SGLang internals)** are the lineage reference. DeepInfer C++ Scheduler and its entry layer ultimately grew out of the vLLM `LLMEngine + Scheduler + Worker` shape.
- **`ep4` (DSpark deep dive)** covers the draft side. This post adds the scheduling side and that low-concurrency-negative table. **Miss either and you misjudge DSpark.**
- **`ep7` (MTP and confidence heads)** puts the confidence head in its production role here: allocating verification budget.
- **`sys9` / `sys10` (KV cache landscape and engineering)** computed how big KV is and how to manage it. The SWA tiering here is a reverse lesson: **first ask how long the data lives, then decide which tier stores it.**
- **`sys6` (HiSparse)** and SWA Bounded Replay are two forms of one idea, trading exactness of history for constant cost.
- **`sys8` (PD disaggregation) / `sys12` (Mooncake)** are precisely what DeepInfer did not open, so use them to fill the gap.
- **`fa4` (DeepSeek V4)** covers the CSA plus HCA compressed attention that underlies the 890 B per token global KV figure. **Model-side compression and systems-side replay together are what bring long-context cost down.**
