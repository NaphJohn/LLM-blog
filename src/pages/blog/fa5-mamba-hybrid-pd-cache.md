---
title: 前沿架构解码手记（五）：Mamba 混合状态 + PD 分离——缓存正确性为何要避免「静默命中错位」
description: 前面四篇讲前沿模型怎么"改注意力"，这一篇转到"改完注意力之后的混合架构怎么服务"。以 Jamba 这类 Mamba+Attention 混合模型为例，拆解 PD 分离部署时跨节点要搬运的"混合状态"（KV Cache + Mamba SSM 递归状态），解释为什么 Mamba 状态没有 token-id 寻址键、在 KV 卸载 + 多 connector 场景下会"静默命中错位"，以及用 fingerprint / 版本号 / connector 级 state-id 守住缓存正确性的做法。
pubDate: 2026-08-10
series: 前沿架构解码手记
lang: zh
altLang: en
altHref: /en/blog/fa5-mamba-hybrid-pd-cache
layout: ../../layouts/BlogPost.astro
---

## 0. 为什么单聊这一篇

前四篇（总览 → Kimi K3 → MiniMax M3 → DeepSeek V4）讲的是**怎么把注意力改到既撑得起 1M 上下文、又不让 KV 爆炸**。但从 fa3、fa4 起你已经能看出另一条暗线：**注意力正在被"非注意力"取代一部分**——MiniMax 的 MSA 是稀疏化，DeepSeek 的 CSA/HCA 是压缩，而更激进的一步，是直接把一部分层换成 **Mamba 这类状态空间模型（SSM）**。

于是出现一类"混合架构"：Attention 层 + Mamba 层交织（典型如 Jamba，每 8 层 Mamba 插 1 层 Attention）。架构变了，**部署 / 推理时要维护的"状态"也跟着变了**——这恰恰是工程上最容易踩坑、也最容易被忽略的点。本篇就把它讲透。

<figure class="arch-fig">
<svg viewBox="0 0 680 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="PD 分离下混合模型的状态搬运与静默命中错位风险">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#8B4513"/>
    </marker>
    <style>
      .box{fill:#fff7ed;stroke:#c2410c;stroke-width:1.5}
      .box2{fill:#eff6ff;stroke:#1d4ed8;stroke-width:1.5}
      .gate{fill:#fef9c3;stroke:#a16207;stroke-width:1.5}
      .lab{font:600 13px -apple-system,'PingFang SC',sans-serif;fill:#1f2937}
      .sub{font:11px -apple-system,'PingFang SC',sans-serif;fill:#475569}
      .warn{font:600 11px -apple-system,'PingFang SC',sans-serif;fill:#b91c1c}
      .conn{font:10px -apple-system,'PingFang SC',sans-serif;fill:#6b7280}
    </style>
  </defs>

  <rect class="box" x="20" y="50" width="190" height="150" rx="8"/>
  <text class="lab" x="115" y="74" text-anchor="middle">Prefill 节点</text>
  <text class="sub" x="115" y="98" text-anchor="middle">产出「混合状态」</text>
  <line x1="40" y1="112" x2="190" y2="112" stroke="#c2410c" stroke-dasharray="3 3"/>
  <text class="sub" x="115" y="132" text-anchor="middle">① KV Cache</text>
  <text class="sub" x="115" y="148" text-anchor="middle">（token-id 寻址·有 key）</text>
  <text class="sub" x="115" y="172" text-anchor="middle">② Mamba SSM 状态</text>
  <text class="sub" x="115" y="188" text-anchor="middle">（定长·无 token-id key）</text>

  <rect class="gate" x="250" y="95" width="180" height="60" rx="8"/>
  <text class="lab" x="340" y="118" text-anchor="middle">RDMA / Mooncake</text>
  <text class="sub" x="340" y="138" text-anchor="middle">按 ptr+index×item_len 裸字节</text>
  <line x1="210" y1="125" x2="248" y2="125" stroke="#8B4513" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="432" y1="125" x2="470" y2="125" stroke="#8B4513" stroke-width="2" marker-end="url(#arrow)"/>

  <rect class="box2" x="470" y="50" width="190" height="150" rx="8"/>
  <text class="lab" x="565" y="74" text-anchor="middle">Decode 节点</text>
  <text class="sub" x="565" y="98" text-anchor="middle">消费混合状态</text>
  <text class="sub" x="565" y="132" text-anchor="middle">按 token 自回归</text>
  <text class="sub" x="565" y="148" text-anchor="middle">生成下一段</text>
  <text class="sub" x="565" y="172" text-anchor="middle">若状态错位 →</text>
  <text class="warn" x="565" y="188" text-anchor="middle">静默吐错·不报错</text>

  <rect class="gate" x="250" y="252" width="380" height="70" rx="8"/>
  <text class="lab" x="440" y="276" text-anchor="middle">缓存正确性护栏（缺则静默命中错位）</text>
  <text class="sub" x="440" y="298" text-anchor="middle">指纹 fingerprint（层+请求id+步号+前缀哈希）</text>
  <text class="sub" x="440" y="314" text-anchor="middle">版本号 / 序列号 · connector 级 state-id（按 id 而非位置投递）</text>

  <text class="conn" x="340" y="240" text-anchor="middle">↓ 必须在交接点做 虚拟id→物理id 翻译；压缩时禁止在传输在飞时搬页</text>
  <path d="M340,155 C340,200 340,212 340,252" fill="none" stroke="#a16207" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>
</svg>
<figcaption>图：PD 分离下，混合模型要把「KV Cache + Mamba SSM 状态」整组跨节点搬运。SSM 状态没有 token-id 寻址键，一旦在路由中被错放 / 串号 / 读到陈旧块，Decode 侧“以为命中”却拿到错误历史，错误沿序列静默累积——不报错。护栏 = 给两类状态都打指纹 / 版本号 / state-id，使错位从“静默”变“可发现”。</figcaption>
</figure>

## 1. 什么是"Mamba 混合状态"

在混合模型里，序列向前推进所需的"记忆"由两路组成：

- **Attention 层 → KV Cache**：随序列长度增长，按 token 切块存储，天然带 token 序列这个"寻址键"。
- **Mamba 层 → SSM 递归状态（hidden state h）**：定长、与序列长度无关，按 `hₜ = Ā·hₜ₋₁ + B·xₜ`、`yₜ = C·hₜ` 递归更新。

所谓"**混合状态**"，就是 PD（Prefill-Decode）分离部署时，必须跨节点从 Prefill 搬到 Decode 的**这一整组「KV Cache + Mamba SSM 状态」**。叫"混合"而不叫"Mamba 状态"，是因为两类状态都得管对、缺一不可。

## 2. PD 分离：纯注意力 vs 混合模型，要带的东西不同

```
纯 Attention 模型（如 Llama）
  Prefill 产出 ──跨节点──▶ Decode 只搬：KV Cache（按 token-id 哈希寻址）
                          错块易暴露为明显退化 / 缺失 → 非"静默"

Mamba 混合模型（如 Jamba）
  Prefill 产出 ──跨节点──▶ Decode 要同时搬：
     A. KV Cache（Attention 层，随序列增长，token-id 寻址）
     B. Mamba SSM 状态（Mamba 层，定长、位置相关、无 token-id key）
```

纯注意力模型过了边界只带 KV；混合模型**两套都得带**，而且 B 这路没有 Attention 那种"天然 key"。

## 3. 为什么混合模型更容易"静默命中错位"

关键差异在**寻址键**：

- **KV Cache 有天然 key**：token 序列本身就是寻址键，vLLM 这类前缀缓存用块哈希（block hash）校验。块错了通常表现为明显退化或缺失，**容易被发现**。
- **Mamba SSM 状态没有 token-id key**：它是一坨定长连续 tensor，只能靠"位置 / 请求上下文"去对应。一旦在路由中被错放、串号、或读到陈旧块，Decode 侧"以为命中"了正确的状态，实际拿到的是别的前缀 / 别请求的状态——模型**照常生成，但内容已经错了，且不抛任何异常**。

这就是"**静默命中错位**"：不像 KV 错块那样"吵"，它**一声不响地吐错**。纯注意力模型即便 KV 错了也容易暴露；Mamba 状态错位更隐蔽，所以"缓存正确性"主要针对的就是它。

<div class="warnbox">
<strong>⚠️ 隐蔽性来源：</strong>SSM 状态错位不会触发 cache miss 或显式报错，它只是让后续每一步的 hₜ 基于错误的历史递推——错误会沿序列悄悄累积，但首 token 的概率分布往往"看起来正常"，人工抽查很难一眼发现。
</div>

## 4. KV 卸载 + 多 connector 把风险放大了

"开 KV 卸载 + Mamba 混合状态 + 多 connector 的 PD 分离部署"这句话，把三个放大因子叠在一起：

- **KV 卸载（offload 到 CPU / NVMe）**：状态不再只在 GPU 显存里，而是要在存储层级间搬进搬出，多了一道序列化 / 反序列化与寻址环节。
- **多 connector**：状态可能走不同通道在 Prefill 与 Decode 之间流转，路由更复杂，块被错配的概率上升。
- **混合状态**：上两条同时作用于 KV 和 SSM 两类状态，而 SSM 这路又没有 token-id 校验。

三者叠加，SSM 状态块若无强制校验，Decode 侧"命中"到错误 / 陈旧状态却无报错的概率显著升高。

## 5. 缓存正确性怎么修：给混合状态打指纹

解法不是"不卸载"或"不用多 connector"，而是**为混合状态（KV + SSM 一起）建立可校验的身份**：

- **指纹（fingerprint）**：用「层号 + 请求 id + 步号 + 前缀哈希」算出状态块的唯一指纹，传输与加载时比对。
- **版本号 / 序列号**：Prefill 产出的状态带版本，Decode 加载前验版本一致，拒绝陈旧块。
- **connector 级 state-id**：在多 connector 路由里，每个状态块绑定全局唯一 state-id，路由按 id 而非"位置"投递，避免串号。

核心思想一句话：**让 SSM 状态也拥有像 KV 那样可被校验的"身份"，使错位从"静默"变成"报错可发现"**。

<div class="keybox">
<strong>一句话总结：</strong>"Mamba 混合状态" = 混合模型跨 PD 边界必须携带的「KV Cache + Mamba SSM 递归状态」联合体；它的 SSM 部分因为没有 token-id 寻址键，在卸载 + 多 connector 场景下最容易静默错位，所以要用指纹 / 版本号 / connector 级 state-id 守住缓存正确性。
</div>

## 6. 工程视角：为什么这件事会越来越重要

- **混合架构在变多**：Jamba、Zamba、Griffon 等把 Mamba / SSM 与 Attention 交织，已成为前沿架构的一条主流路线。
- **PD 分离 + 卸载是降本标配**：长上下文、大模型推理要降本，PD 分离与 KV 卸载几乎必上，多 connector 也很常见。
- **两者交汇处的正确性，是"能不能放心上线"的底线**：一个静默错位的 SSM 状态，可能让整个服务的输出悄悄变差而不报警——对生产环境是隐性事故。

> 延伸：如果你在 vLLM / SGLang 上跑混合模型并做 PD 分离，重点检查 SSM 状态是否走了和 KV 同样的「前缀哈希 / 块校验」机制；很多实现还只给 KV 做了校验，SSM 状态仍是"裸搬"，这正是静默命中错位的温床。
