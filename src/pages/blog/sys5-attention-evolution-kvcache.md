---
title: 推理系统基础设施手记（五）：从 MHA 到 HiCache——KV Cache 压缩方向的演进全景
description: 把 2026 年长上下文模型的 KV Cache 演进串成一条完整链：MHA（不压缩）→ GQA（减头）→ MLA（压特征维）→ DSA（沿 token 维选重要）→ CSA（先压 token 再稀疏）→ HCA（极度压 token 后稠密）→ HiCache（架构压缩后管放在哪）。关键是「压缩方向」从特征维转向了 token 维；而 HCA 与 HiCache 是上下游两级优化，不是竞品。
pubDate: 2026-08-26
series: 推理系统基础设施手记
lang: zh
altLang: en
altHref: /en/blog/sys5-attention-evolution-kvcache
layout: ../../layouts/BlogPost.astro
---

## 0. 一句话主线

> **MHA：不压缩 → GQA：减 KV 头数 → MLA：沿「特征维」压缩 → DSA：沿「token 维」做稀疏选择 → CSA：先沿 token 维压缩再做稀疏 → HCA：沿 token 维极度压缩后直接 Dense → HiCache：架构压缩后的 KV 还要管「放 GPU / CPU / SSD 哪层」。**

整条线最关键的变化，是 **KV Cache 的压缩方向发生了转移**——从「把每个 token 存得更小」（特征维）转向「把很长的历史本身压短」（token 维）。而最后一步 HiCache 已经不属于「压缩」，它是另一个维度的问题：**这些 KV 已经产生了，该放在哪**。

---

## 1. 完整演进链

<div class="fig">
<svg viewBox="0 0 680 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MHA 到 HiCache 的 KV Cache 演进全景">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="24" font-size="15" font-weight="700" fill="#1a1a1a">KV Cache 演进：压缩方向从「特征维」走向「token 维」</text>

  <!-- row template: box + annotation, 7 rows -->
  <!-- MHA -->
  <rect x="20" y="44" width="120" height="34" rx="6" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="80" y="66" font-size="13" font-weight="700" fill="#374151" text-anchor="middle">MHA</text>
  <text x="156" y="58" font-size="12" fill="#444">每个 token 完整存 K/V，不压缩</text>
  <text x="156" y="74" font-size="11" fill="#9ca3af">KV 长度 = 上下文长度 N</text>

  <!-- GQA -->
  <rect x="20" y="92" width="120" height="34" rx="6" fill="#ecfeff" stroke="#06b6d4"/>
  <text x="80" y="114" font-size="13" font-weight="700" fill="#0e7490" text-anchor="middle">GQA</text>
  <text x="156" y="106" font-size="12" fill="#444">多个 Q 头共享 KV 头 → KV 头数↓</text>
  <text x="156" y="122" font-size="11" fill="#9ca3af">Token 数仍是 N，只减了 head 数</text>

  <!-- MLA -->
  <rect x="20" y="140" width="120" height="34" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="80" y="162" font-size="13" font-weight="700" fill="#1e40af" text-anchor="middle">MLA</text>
  <text x="156" y="154" font-size="12" fill="#444">每个 token 压成低维 latent c（特征维）</text>
  <text x="156" y="170" font-size="11" fill="#9ca3af">N 个 entry，但每个更小 ≈ 1/64</text>

  <!-- DSA -->
  <rect x="20" y="188" width="120" height="34" rx="6" fill="#eef2ff" stroke="#6366f1"/>
  <text x="80" y="210" font-size="13" font-weight="700" fill="#4338ca" text-anchor="middle">DSA</text>
  <text x="156" y="202" font-size="12" fill="#444">Indexer 选 top-k 重要 token（token 维）</text>
  <text x="156" y="218" font-size="11" fill="#9ca3af">省算力，但 KV 仍要存（省算不省存）</text>

  <!-- CSA -->
  <rect x="20" y="236" width="120" height="34" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="80" y="258" font-size="13" font-weight="700" fill="#1e40af" text-anchor="middle">CSA</text>
  <text x="156" y="250" font-size="12" fill="#444">先 4 token→1 KV 块，再 top-k 稀疏</text>
  <text x="156" y="266" font-size="11" fill="#9ca3af">1M → 250K 压缩 KV，再选 1024</text>

  <!-- HCA -->
  <rect x="20" y="284" width="120" height="34" rx="6" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="80" y="306" font-size="13" font-weight="700" fill="#92400e" text-anchor="middle">HCA</text>
  <text x="156" y="298" font-size="12" fill="#444">128 token→1 条目，对全部做 Dense</text>
  <text x="156" y="314" font-size="11" fill="#9ca3af">1M → ~7800 条目，不再需要 top-k</text>

  <!-- HiCache -->
  <rect x="20" y="332" width="120" height="34" rx="6" fill="#faf5ff" stroke="#a855f7"/>
  <text x="80" y="354" font-size="13" font-weight="700" fill="#7e22ce" text-anchor="middle">HiCache</text>
  <text x="156" y="346" font-size="12" fill="#444">压缩后的 KV 分层存放（系统维）</text>
  <text x="156" y="362" font-size="11" fill="#9ca3af">GPU HBM → CPU DRAM → SSD NVMe</text>

  <!-- vertical connectors -->
  <line x1="80" y1="78" x2="80" y2="92" stroke="#9ca3af" marker-end="url(#arr)"/>
  <line x1="80" y1="126" x2="80" y2="140" stroke="#9ca3af" marker-end="url(#arr)"/>
  <line x1="80" y1="174" x2="80" y2="188" stroke="#9ca3af" marker-end="url(#arr)"/>
  <line x1="80" y1="222" x2="80" y2="236" stroke="#9ca3af" marker-end="url(#arr)"/>
  <line x1="80" y1="270" x2="80" y2="284" stroke="#9ca3af" marker-end="url(#arr)"/>
  <line x1="80" y1="318" x2="80" y2="332" stroke="#a855f7" marker-end="url(#arr)"/>
</svg>
<p class="cap">图：从 MHA 到 HiCache 的完整演进。颜色标识压缩所在的「维」——蓝=特征维（MLA）/紫=token 维（CSA/HCA）/灰=不压缩（MHA）/青=减头（GQA）/靛=token 维选择（DSA）/紫=系统层（HiCache）。</p>
</div>

---

## 2. 每一步到底在动什么刀

| 机制 | 动刀位置 | 压缩比 | 解决什么 | 不解决什么 |
|---|---|---|---|---|
| **MHA** | 无 | 1× | 基准：精确 | KV 随 N 线性膨胀 |
| **GQA** | KV 头数 | 4–8× | 减 KV 头，省显存 | Token 数仍是 N |
| **MLA** | 特征维（每 token 更小） | ~1/64 | 每 token 缓存变小 | entry 数仍是 N |
| **DSA** | token 维（选重要的读） | 算力↓ | 只算 top-k，省算力 | KV 容量没降（省算不省存） |
| **CSA** | token 维（先压再选） | 4× + top-k | KV 降到 1/4，再稀疏 | 仍需 Indexer 选块 |
| **HCA** | token 维（极压后 Dense） | 128× | KV 压到极致，稠密更快 | 单 token 信息高度抽象 |
| **HiCache** | 存储层级（放哪） | — | 多级存、降 GPU 压力 | 不做任何压缩 |

---

## 3. 最关键的一点：压缩方向变了

很多人把 MLA / DSA / CSA / HCA 混为一谈，其实它们的**压缩方向完全不同**：

- **MLA = 每个 token 存得更小**（特征维 / head 维）
  ```
  T1 → latent1      ← 还是 N 个 entry
  T2 → latent2
  ...
  TN → latentN
  ```
  只是「每个 entry 更小」，entry 数量没变。

- **HCA = token 数本身变少**（token 维）
  ```
  T1..T128 → C1      ← N/128 个 entry
  T129..T256 → C2
  ```
  这是「把很长的历史直接压成很短的摘要」。

- **DSA = 每次只读重要的 token**（token 维的「选择」而非「压缩」）
  ```
  1M token → Indexer → top-k=2048 → 只 attend 这些
  ```
  它不减少 KV 的存储量，只减少**计算量**——所以「验证/选择多少 token」与「KV Cache 本身存多少 token」是两个问题。

- **CSA = 先把 token 压小，再做稀疏读取**
  ```
  1M → 4× 压缩 → 250K 压缩 KV → top-k → 1024 → Attention
  ```
  即「先粗压，再挑重点」。

- **HCA = 把很长的历史直接压成很短的摘要后全读**
  ```
  1M → 128× 压缩 → 7800 条目 → Dense Attention
  ```
  用「更狠的压缩」换掉了「稀疏检索」——因为 7800 条已经不多，直接稠密反而比不规则稀疏更快。

**一句话区分**：MLA 在「特征维」动刀（每个 token 更小），DSA/CSA/HCA 在「token 维」动刀（历史变短或只挑重要的）。这是理解整条演进的钥匙。

---

## 4. MSA 在哪？

`sys3` 已经单独讲过 **MSA / CSA / HCA 三者的对比与协作**：MSA 只改「看哪里」（稀疏选择，不压 KV），CSA 先压再选，HCA 极压后稠密。本文的演进链把它们放进更大的上下文里——MSA 是 DSA 思路的「模型内原生」版本，而 DeepSeek V4 走的是 CSA/HCA 这条「先压后读」的路线。两者不是替代，是分层协作。

---

## 5. HCA ≠ HiCache：这是两级优化

这是最容易被混淆的一点。**HCA 和 HiCache 不是竞品，是上下游**。

```text
第一层：模型压缩（架构层）
MHA → MLA → CSA / HCA
             ↓
         KV 体积下降（1M → 几个 GB）

第二层：系统分层（系统层）
HiCache
             ↓
     GPU HBM / CPU DRAM / SSD NVMe 分层
             ↓
     GPU HBM 压力进一步下降
```

- **HCA 负责「减少 KV 表示量」**：1M token 的 KV 从 ~100GB 压到几个 GB（架构层面的降本）。
- **HiCache 负责「管理剩下的 KV」**：这几个 GB 里，热的放 GPU HBM、温的放 CPU DRAM、冷的放 SSD，需要时用 H2D/D2H 搬（系统层面的分层管理）。

<div class="fig">
<svg viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="HCA 与 HiCache 的两级优化关系">
  <defs>
    <marker id="arr2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4b5563"/>
    </marker>
  </defs>
  <text x="20" y="24" font-size="14" font-weight="700" fill="#1a1a1a">HCA（架构压缩）与 HiCache（系统分层）是上下游两级优化</text>

  <!-- layer 1 -->
  <rect x="20" y="48" width="640" height="56" rx="8" fill="#fff7ed" stroke="#f59e0b"/>
  <text x="35" y="72" font-size="13" font-weight="700" fill="#92400e">第一层 · 模型架构压缩</text>
  <text x="35" y="92" font-size="11.5" fill="#444">MHA → MLA → CSA / HCA：把 KV 表示量压下来，1M token 从 ~100GB 降到几个 GB</text>

  <!-- arrow down -->
  <line x1="340" y1="104" x2="340" y2="124" stroke="#9ca3af" marker-end="url(#arr2)"/>
  <text x="350" y="120" font-size="11" fill="#9ca3af">KV Cache</text>

  <!-- layer 2 -->
  <rect x="20" y="128" width="640" height="56" rx="8" fill="#faf5ff" stroke="#a855f7"/>
  <text x="35" y="152" font-size="13" font-weight="700" fill="#7e22ce">第二层 · 系统分层（HiCache）</text>
  <text x="35" y="172" font-size="11.5" fill="#444">GPU HBM（热）/ CPU DRAM（温）/ SSD NVMe（冷），按需 H2D/D2H 搬运，降 GPU 压力</text>

  <!-- storage tiers -->
  <rect x="40" y="200" width="170" height="44" rx="6" fill="#ecfdf5" stroke="#10b981"/>
  <text x="125" y="222" font-size="12" font-weight="700" fill="#047857" text-anchor="middle">GPU HBM</text>
  <text x="125" y="238" font-size="10.5" fill="#047857" text-anchor="middle">L1 · 热 KV</text>

  <rect x="255" y="200" width="170" height="44" rx="6" fill="#eff6ff" stroke="#2563eb"/>
  <text x="340" y="222" font-size="12" font-weight="700" fill="#1d4ed8" text-anchor="middle">CPU DRAM</text>
  <text x="340" y="238" font-size="10.5" fill="#1d4ed8" text-anchor="middle">L2 · 温 KV</text>

  <rect x="470" y="200" width="170" height="44" rx="6" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="555" y="222" font-size="12" font-weight="700" fill="#374151" text-anchor="middle">SSD NVMe</text>
  <text x="555" y="238" font-size="10.5" fill="#374151" text-anchor="middle">L3 · 冷 KV</text>

  <line x1="210" y1="222" x2="253" y2="222" stroke="#9ca3af" marker-end="url(#arr2)"/>
  <line x1="425" y1="222" x2="468" y2="222" stroke="#9ca3af" marker-end="url(#arr2)"/>
</svg>
<p class="cap">图：HCA 减少「KV 表示量」，HiCache 管理「KV 放哪层」。两者正交、可叠加，构成从架构到系统的两级优化。</p>
</div>

> 顺带一提：DeepSeek-V4 的 KV 管理确实已经针对「CSA/HCA 压缩 KV + SWA 未压缩 KV」分别设计了 cache 结构——这正说明 HCA 的压缩与 HiCache 这种系统层 KV 管理是可以衔接起来的。如果你在看 **GLM-5.2/5.3 的 DSA、HiCache 和 DeepSeek-V4 的 CSA/HCA 是不是同一套思想**，这个区别尤其关键：**DSA/CSA/HCA 解决「Attention 怎么读」，HiCache 解决「KV Cache 放哪里」**。

---

## 6. 一句话串起来

> **MLA 是「每个 token 存得更小」，DSA 是「每次只读重要 token」，CSA 是「先把 token 压小再做稀疏读取」，HCA 是「把很长的历史直接压成很短的摘要后全读」，而 HiCache 则是在这些 KV 已经产生之后，解决「放 GPU、CPU 还是更低层存储」的问题。**

---

## 7. 与已有文章的衔接

- **MLA 算子级实现** → 见 [`op1-mla-operator`](../op1-mla-operator)（含 MHA 基准与 c_kv 伪代码；GQA/MQA 算子对比待补）
- **CSA / HCA 三者对比与协作** → 见 [`sys3-attention-evolution`](../sys3-attention-evolution)（MSA/CSA/HCA 对比图 + V4 分层图）
- **DeepSeek V4 架构深读（四级演进 MLA→NSA→DSA→CSA+HCA）** → 见 [`mm5-efficiency-frontier`](../mm5-efficiency-frontier) §2
- **HiCache 与 NUMA/PCIe/NIC 拓扑** → 见 [`sys1-pcie-numa-nic-hicache`](../sys1-pcie-numa-nic-hicache)
- **长上下文训练（Ring Attention）** → 见 [`sys2-ring-attention`](../sys2-ring-attention)

本文是把上面散落各篇的「压缩方向」主线，第一次串成一条从 MHA 到 HiCache 的完整链。
