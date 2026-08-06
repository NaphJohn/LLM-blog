---
title: 'Frontier Architecture Decoding Notes (3): MiniMax M2.7 → M3 — The Leap of Sparse Attention'
description: 'Following the roadmap, the second model torn apart is MiniMax. First the M2.7 baseline (standard attention, 200K, text-only, API-only), then how M3 uses MiniMax Sparse Attention (MSA) to pull context from 200K to 1M while cutting per-token compute at 1M to ~1/20 of M2.7 (prefill ~9×, decode ~15× faster). M3 also adds native multimodal (text+image+video) and computer use, and open-weights.'
pubDate: 2026-08-06
series: Frontier Architecture Decoding Notes
lang: en
altLang: zh
altHref: /blog/fa3-minimax-m3
layout: ../../../layouts/BlogPost.astro
---

## 0. Why Watch MiniMax's "Evolution"

Kimi K3 "adds a delta on MLA" — a gentle change. MiniMax gives a more直观 lesson: **how one series goes from standard attention to sparse, step by step**. See the M2.7 starting point clearly, and the M3 leap becomes obvious.

## 1. Starting Point: MiniMax M2.7 (baseline)

- **standard softmax attention**, no sparsity;
- **200K context**, text-only, no native multimodality;
- **API-only**, weights not open;
- mostly text training data.

It represents the "previous generation" dense / standard-attention model: capable enough, but both context and compute hit a ceiling.

## 2. The Leap: MiniMax M3 + MSA

M3's core innovation is **MiniMax Sparse Attention (MSA)**:

- **context 200K → 1M** (5×);
- at 1M context, **per-token compute is ~1/20 of M2.7**;
- **prefill ~9×, decode ~15× faster**;
- trained on **100T tokens**, **interleaved data**.

The intuition of sparse attention: not every query must attend to all historical keys. MSA uses a sparse-selection mechanism so each position only attends to the "truly relevant" fraction of history, holding compute down at 1M length.

<div class="keybox">
<strong>Key:</strong> sparse attention's payoff comes from "computing less" — at 1M length M3's per-token compute is only <strong>1/20</strong> of M2.7's, buying order-of-magnitude prefill/decode speedups. This is the canonical "context grows but cost doesn't explode" solution.
</div>

## 3. Beyond Attention: Native Multimodal + Computer Use

M3 doesn't only change attention; it fills in "usability":

- **native multimodal**: text + image + video modeled uniformly, not a text model with vision bolted on;
- **computer use**: can operate GUIs / browsers, fitting Agent scenarios;
- **open weights**: vs M2.7's "API-only," M3 opens weights, helping ecosystem and secondary development.

## 4. Benchmarks (official)

| Benchmark | M3 score |
|---|---|
| SWE-Bench Pro | 59.0% |
| Terminal-Bench 2.1 | 66.0% |
| MCP Atlas | 74.2% |

These agentic / tool-call / terminal-operation benchmarks map exactly to M3's computer-use and long-context positioning.

## 5. Contrast With Kimi K3

- Kimi K3 "**changes attention internals**" (KDA delta + gate), keeping the MLA base;
- MiniMax M3 "**changes attention's computation scope**" (sparsify), jumping straight from standard attention to MSA;
- common: both aim to **hold 1M context without exploding compute**; differ in where they cut.

> Next: DeepSeek V4, more aggressive than sparsity — compressing KV itself.
