---
title: 'Agent Notes (1): The Agent Knowledge System — capability stack, context engineering, and inference rebuilt for Agentic workloads'
description: 'Threading the scattered Agent notes into one system: the capability stack (model/planning/tools/memory), tool calling and MCP, context engineering, execution paradigms and orchestration, and how Agentic workloads turn KV Cache from a memory problem into a scheduling problem.'
pubDate: 2026-09-08
series: Agent Notes
lang: en
altLang: zh
altHref: /blog/ag1-agent-knowledge-system
layout: ../../../layouts/BlogPost.astro
---

## 0. The One-Line Thread

An Agent is not "a model that chats better" — it is **an LLM embedded in a perceive → plan → act → reflect loop**. And the deepest engineering consequence of that is not on the model side but on the **inference-infrastructure** side: the variable-length, branching, multi-turn nature of Agentic workloads upgrades KV Cache from a question of "is there enough HBM" to "how do we schedule state across GPU / host memory / disk". This post consolidates the scattered Agent notes into one system and ties them to this blog's own HiCache (sys1) / HiSparse (sys6) / DCP (sys7) / KV quantization (op2) thread.

## 1. Capability stack: Agent = model + planning + tools + memory

| Layer | Role | Key question |
|---|---|---|
| Model (the brain) | reasoning, decisions, generation | instruction following, long context, output stability |
| Planning | break a goal into executable steps | step budget, failure rollback |
| Tools | reach the outside world | calling protocol, permissions, sandbox |
| Memory | persist across turns / sessions | what to keep, for how long, on which tier |

Drop any one of the four and you do not have an Agent — just a chatbot with a prompt.

## 2. Tool calling: from Function Calling to MCP

- **Function Calling**: the model emits a structured call intent (function name + arguments); the host executes it and feeds the result back into context. This is the minimal mechanism that lets an Agent "act".
- **MCP (Model Context Protocol)**: collapses the M×N problem of "every Agent adapting to every tool" into the M+N problem of "integrate a tool once, use it everywhere". It is the USB-C of the Agent era.
- **Engineering pitfalls**: tool descriptions themselves consume context; argument hallucination; timeout/retry policy; and worst of all — **irreversible side effects** (write operations need a human confirmation gate).

## 3. Context engineering: the real Agent moat

Agent failures are rarely "the model is not smart enough" — they are almost always **poorly managed context**.

- Context is a **costed resource**: every turn replays the prefix, so token cost grows linearly with turns.
- Standard tools: summarization/compression, retrieval injection, offloading context to external storage, KV reuse (prefix caching).
- **Directory attention for 1M-token Agents** (DeepSeek V4 line): fuse 128 tokens into one "directory entry" and run dense attention over roughly 8K entries. The key point: **it must be Dense, not Sparse** — 8K KV entries run faster on a Dense kernel because the access pattern is more regular. This is structural innovation driven by the need that "distant history still needs a global outline".

## 4. Execution paradigms and orchestration: simple wins

- **ReAct**: think → act → observe loop. **Plan-and-Execute**: plan first, then execute. **Reflection/self-correction**: replan after failure.
- **Orchestration pattern**: an Orchestrator (thinking / planning / success detection) plus executors. DeepMind's Gemini Robotics 1.5 brings this to embodiment: GR-ER 1.5 (a thinking VLM) is the Orchestrator, GR 1.5 (a generalist VLA) is the executor, "think before acting". Result: long-horizon task progress score about **80%**, versus only **44%** for a Thinking VLA alone — **orchestration itself is performance**.
- **Reality check (Goldman Sachs)**: AI has entered the "execution era" and competition has shifted from models to workflows; **68% of production Agents use simple, controllable designs — at most 10 steps before human intervention**. Do not overestimate today's autonomy.

## 5. Agentic workloads: why they rebuild the inference system (the core)

This is where Agent knowledge meets the rest of this blog.

### 5.1 Three traits that break the static assumptions

What separates Agentic workloads from classic single-turn Q&A:

- **Variable length**: context length changes every turn, across a huge range.
- **Branching**: one task spawns multiple attempt paths (sampling, retries, branches).
- **Multi-turn + cold sessions**: a session can sit idle for half an hour, then resume.

Consequence: the engine is static assumptions — "specialize kernels by length", "evict KV by LRU" — **all break**.

### 5.2 The evaluation metric has to change

- Old metric: fixed 8K prompt throughput / first-token TTFT.
- New metric: **cross-turn KV hit rate + state migration cost**.

If you are building an inference substrate for Agents or multi-turn dialogue, switch the metric from "first-token throughput" to "can the next turn still hit its historical KV" — that is the real bottleneck in Agent scenarios.

### 5.3 Four engine-side response lines

1. **Tiered offload**: a cold session's KV should not permanently occupy the most expensive HBM. HiCache (sys1) defines placement and data paths across GPU / host memory / disk; HiSparse (sys6) treats HBM as a cache and host DRAM as the authoritative full KV; the newest step is **adding a disk tier** — a session idle for half an hour can let its KV sit quietly on disk and yield HBM to requests actually running.
2. **Cross-instance sharing and resumption**: TensorCast (Peking University × StepFun, arXiv:2608.06007) decouples KV migration and weight materialization into a shared TaaS layer, cutting **median TTFT by up to −93.2% for high-concurrency multi-turn Agents** (note the gain concentrates on the high-concurrency + multi-turn combination; single-turn / short-context / low-concurrency loads see none of it); Mooncake Store added `save_decode_cache` so decode-phase KV also lands in the shared Store, letting multi-turn / Agent workloads resume across instances **without recomputation**.
3. **Parallelism and capacity**: DCP (sys7) shards KV along the sequence dimension to remove TP redundant replication; KV Cache quantization (op2) cuts capacity directly via FP16→FP8 / INT4. Prefill-dominated loads (Coding Agents: long context, high prefix hit rate) further need schemes like LayerSplit that keep only part of the per-layer KV on each GPU, with extra communication of just an Indexer Cache broadcast ≈ **1/8 of KV**.
4. **Correctness over speed**: ROCm ring-cache once reused a slot that was still referenced, producing **wrong output rather than slow output**. In Agent scenarios that class of silent error is far more lethal than latency.

### 5.4 The shift in one sentence

The inference engine is evolving from "a runtime that manages HBM" into "a runtime that schedules three classes of state across GPU / host memory / disk".

## 6. Deployment and industry observations

- **Infrastructure is crystallizing**: Alibaba Cloud launched KV Cache Store plus **Agentic FS**, with daily token calls surpassing **140 trillion** — Agents are no longer demos, they are real infrastructure load.
- **Form factors are spreading**: Tencent TairosAgent embodied-agent framework, QClaw / WorkBuddy multimodal Agent apps across three clients, Alibaba office Agents, SenseTime Seko video agent, StepFun / Nubia agent phones — Agents are growing out of the chat box into operating systems, office suites and robots.
- **Embodied side**: NitroGen ports the GR00T architecture into virtual worlds (1000+ games, 40k hours of interaction) to train generalist embodied Agents, using generated environments to relieve embodied data hunger.
- **Keep expectations rational**: 68% of production Agents are simple ≤10-step controllable designs; genuine long-horizon autonomy (like the 80%-progress embodied agentic system) is still confined to specific vertical scenarios.

## 7. Practice checklist

1. Ask about the workload shape first: single-turn Q&A → classic optimization; multi-turn Agent → prioritize KV hit rate and state migration cost.
2. Change the metric: cross-turn KV hit rate + state migration cost, replacing fixed 8K throughput.
3. Tiered offload needs a disk tier; cold sessions should not hold HBM.
4. Context engineering before model swapping: summarization, retrieval, offload, KV reuse.
5. Use MCP to collapse tool-integration cost; keep a human gate on writes.
6. Do not worship autonomy: design for ≤10 steps by default, with rollback and human takeover.
7. Watch for silent errors: state/cache reuse needs occupancy validation.

**Tying it together**: Agents define the workload shape (variable length, branching, multi-turn) → that shape breaks static scheduling assumptions → engines are forced into cross-tier state schedulers (HiCache / HiSparse / DCP / KV quantization) → and the need that "distant history still needs a global outline" in turn spawns new structures like directory attention. That is the full loop in which Agents and inference systems reshape each other today.

*Note: this post consolidates the 2026-08-13 to 2026-09-08 daily AI hotspot tracking and AI knowledge-base entries into a systematic version.*
