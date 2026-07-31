---
title: （二）vLLM 原理与结构：从 PagedAttention 到删掉 PagedAttention
description: 分页 KV Cache 的虚拟内存类比、连续批处理、V1 架构分层，以及 0.25 版本移除 PagedAttention、Model Runner V2 成为默认背后的逻辑。
pubDate: 2026-08-01
series: vLLM 与 SGLang 框架解码手记
lang: zh
altLang: en
altHref: /en/blog/fw2-vllm-internals
layout: ../../layouts/BlogPost.astro
---

## 1. PagedAttention：把操作系统的虚拟内存搬进 KV Cache

上一篇提到，朴素推理的 KV Cache 是"按最大长度连续预留"，利用率只有 20%–40%。vLLM 的第一个杀手锏就是解决这个问题，思路直接借鉴了操作系统。

### 1.1 类比

| 操作系统 | vLLM |
|---|---|
| 进程的虚拟地址空间 | 一条请求的逻辑 KV 序列 |
| 物理内存页（4 KB） | KV block（通常 16 个 token） |
| 页表（page table） | **块表（block table）** |
| 按需分页、写时复制 | 按需分块、前缀共享 |

一条请求的 KV **在逻辑上是连续的**（token 0,1,2,...），但**在物理显存里可以散落在任意位置**。中间由一张块表做映射。

<div class="fig">
<svg viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#374151">逻辑视图（请求看到的连续序列）</text>
  <g font-size="10" text-anchor="middle">
    <rect x="16" y="28" width="60" height="26" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="46" y="45" fill="#1e3a8a">tok 0-15</text>
    <rect x="80" y="28" width="60" height="26" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="110" y="45" fill="#1e3a8a">tok 16-31</text>
    <rect x="144" y="28" width="60" height="26" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="174" y="45" fill="#1e3a8a">tok 32-47</text>
    <rect x="208" y="28" width="60" height="26" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="238" y="45" fill="#1e3a8a">tok 48-63</text>
  </g>
  <text x="286" y="46" font-size="11" fill="#6b7280">请求 A（64 token）</text>

  <rect x="16" y="76" width="252" height="40" rx="5" fill="#fef9c3" stroke="#eab308"/>
  <text x="142" y="94" font-size="12" font-weight="700" fill="#854d0e" text-anchor="middle">块表 Block Table</text>
  <text x="142" y="110" font-size="10" fill="#854d0e" text-anchor="middle">[0]→#7  [1]→#2  [2]→#9  [3]→#4</text>

  <line x1="46" y1="56" x2="60" y2="74" stroke="#9ca3af" stroke-width="1"/>
  <line x1="110" y1="56" x2="110" y2="74" stroke="#9ca3af" stroke-width="1"/>
  <line x1="174" y1="56" x2="174" y2="74" stroke="#9ca3af" stroke-width="1"/>
  <line x1="238" y1="56" x2="224" y2="74" stroke="#9ca3af" stroke-width="1"/>

  <text x="16" y="146" font-size="12" font-weight="700" fill="#374151">物理显存（block 池，任意散落）</text>
  <g font-size="10" text-anchor="middle">
    <rect x="16" y="156" width="46" height="28" rx="3" fill="#f3f4f6" stroke="#d1d5db"/><text x="39" y="174" fill="#9ca3af">#0 空</text>
    <rect x="66" y="156" width="46" height="28" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="89" y="174" fill="#065f46">#1 B</text>
    <rect x="116" y="156" width="46" height="28" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="139" y="174" fill="#1e3a8a">#2 A1</text>
    <rect x="166" y="156" width="46" height="28" rx="3" fill="#f3f4f6" stroke="#d1d5db"/><text x="189" y="174" fill="#9ca3af">#3 空</text>
    <rect x="216" y="156" width="46" height="28" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="239" y="174" fill="#1e3a8a">#4 A3</text>
    <rect x="266" y="156" width="46" height="28" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="289" y="174" fill="#065f46">#5 B</text>
    <rect x="316" y="156" width="46" height="28" rx="3" fill="#f3f4f6" stroke="#d1d5db"/><text x="339" y="174" fill="#9ca3af">#6 空</text>
    <rect x="366" y="156" width="46" height="28" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="389" y="174" fill="#1e3a8a">#7 A0</text>
    <rect x="416" y="156" width="46" height="28" rx="3" fill="#f3f4f6" stroke="#d1d5db"/><text x="439" y="174" fill="#9ca3af">#8 空</text>
    <rect x="466" y="156" width="46" height="28" rx="3" fill="#dbeafe" stroke="#3b82f6"/><text x="489" y="174" fill="#1e3a8a">#9 A2</text>
    <rect x="516" y="156" width="46" height="28" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="539" y="174" fill="#065f46">#10 B</text>
    <rect x="566" y="156" width="46" height="28" rx="3" fill="#f3f4f6" stroke="#d1d5db"/><text x="589" y="174" fill="#9ca3af">#11 空</text>
  </g>

  <rect x="380" y="76" width="284" height="40" rx="5" fill="#ecfdf5" stroke="#10b981"/>
  <text x="522" y="94" font-size="12" font-weight="700" fill="#047857" text-anchor="middle">共享前缀：写时复制</text>
  <text x="522" y="110" font-size="10" fill="#047857" text-anchor="middle">A、B 若 prompt 前缀相同 → 指向同一物理块，引用计数 +1</text>

  <text x="16" y="212" font-size="11" fill="#6b7280">图 1：PagedAttention 的逻辑—物理映射。碎片被压缩到"最多浪费一个 block"，且相同前缀可跨请求共享同一物理块。</text>
  <text x="16" y="232" font-size="11" fill="#6b7280">结果：KV 有效利用率从 ~20-40% 提到 &gt;90%，同等显存下并发数提升数倍。</text>
</svg>
</div>

### 1.2 三个直接收益

1. **碎片几乎消失**：浪费的上限是"最后一个 block 里没填满的部分"，最多 15 个 token 的空间，而不是上千。
2. **共享前缀零成本**：系统提示词、few-shot 示例、多轮对话历史——只要前缀相同，多条请求指向同一批物理块，引用计数管理，写时复制。
3. **并行采样便宜**：一个 prompt 要采 n 个不同回答（best-of-n），prompt 部分的 KV 只存一份。

## 2. 连续批处理：从"等整批"到"等一步"

分页解决了显存，还要解决**队头阻塞**。

vLLM 的调度粒度不是"一批请求"，而是**一次迭代（iteration-level scheduling）**：

```
每一步 decode 结束后：
  ├─ 谁生成完了 → 立刻返回、释放它的 block
  ├─ 队列里有新请求 → 立刻塞进空出来的槽位
  └─ 显存不够 → 抢占（把某条请求的 KV 换出或重算）
```

于是批的组成是**动态流动**的，GPU 永远在满负荷跑活跃请求，而不是等最慢的那一条。这就是 **Continuous Batching**（也叫 in-flight batching）。

配合 **chunked prefill**（把长 prompt 的 prefill 切成小块，穿插进 decode 步）后，长 prompt 不再一次性霸占整步计算，正在流式输出的请求也不会被卡住。

## 3. V1 架构：进程分层

vLLM V1（2025 年重构）把整个引擎拆成清晰的层次：

<div class="fig">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <rect x="150" y="14" width="380" height="34" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
  <text x="340" y="36" font-size="13" font-weight="700" fill="#1d4ed8" text-anchor="middle">API Server（OpenAI 兼容 / 前端进程）</text>
  <text x="546" y="36" font-size="10" fill="#6b7280">tokenize · 请求校验</text>

  <path d="M340 50 L340 66" stroke="#6b7280" stroke-width="1.5" marker-end="url(#a2)"/>
  <defs><marker id="a2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6b7280"/></marker></defs>
  <text x="352" y="63" font-size="10" fill="#6b7280">ZMQ / IPC</text>

  <rect x="150" y="68" width="380" height="34" rx="6" fill="#f0fdf4" stroke="#86efac"/>
  <text x="340" y="90" font-size="13" font-weight="700" fill="#047857" text-anchor="middle">EngineCore（独立进程，busy-loop）</text>

  <rect x="60" y="116" width="180" height="60" rx="6" fill="#fefce8" stroke="#fde047"/>
  <text x="150" y="136" font-size="12" font-weight="700" fill="#854d0e" text-anchor="middle">Scheduler 调度器</text>
  <text x="150" y="154" font-size="10" fill="#854d0e" text-anchor="middle">迭代级调度 · 抢占</text>
  <text x="150" y="168" font-size="10" fill="#854d0e" text-anchor="middle">chunked prefill · 优先级</text>

  <rect x="256" y="116" width="168" height="60" rx="6" fill="#fefce8" stroke="#fde047"/>
  <text x="340" y="136" font-size="12" font-weight="700" fill="#854d0e" text-anchor="middle">KVCacheManager</text>
  <text x="340" y="154" font-size="10" fill="#854d0e" text-anchor="middle">block 池 · 块表</text>
  <text x="340" y="168" font-size="10" fill="#854d0e" text-anchor="middle">前缀缓存 · 卸载/分层</text>

  <rect x="440" y="116" width="180" height="60" rx="6" fill="#fefce8" stroke="#fde047"/>
  <text x="530" y="136" font-size="12" font-weight="700" fill="#854d0e" text-anchor="middle">Structured Output</text>
  <text x="530" y="154" font-size="10" fill="#854d0e" text-anchor="middle">grammar / JSON schema</text>
  <text x="530" y="168" font-size="10" fill="#854d0e" text-anchor="middle">流式解析引擎</text>

  <path d="M340 178 L340 194" stroke="#6b7280" stroke-width="1.5" marker-end="url(#a2)"/>

  <rect x="150" y="196" width="380" height="42" rx="6" fill="#fdf2f8" stroke="#f9a8d4"/>
  <text x="340" y="214" font-size="13" font-weight="700" fill="#9d174d" text-anchor="middle">Model Runner V2（MRv2）</text>
  <text x="340" y="230" font-size="10" fill="#9d174d" text-anchor="middle">async-first · 零 CPU-GPU 同步 · 整步 CUDA Graph 捕获</text>

  <rect x="60" y="252" width="140" height="32" rx="6" fill="#f3f4f6" stroke="#d1d5db"/>
  <text x="130" y="272" font-size="11" fill="#374151" text-anchor="middle">Attention 后端</text>
  <rect x="212" y="252" width="140" height="32" rx="6" fill="#f3f4f6" stroke="#d1d5db"/>
  <text x="282" y="272" font-size="11" fill="#374151" text-anchor="middle">FlashAttn / FlashInfer</text>
  <rect x="364" y="252" width="120" height="32" rx="6" fill="#f3f4f6" stroke="#d1d5db"/>
  <text x="424" y="272" font-size="11" fill="#374151" text-anchor="middle">量化 kernel</text>
  <rect x="496" y="252" width="124" height="32" rx="6" fill="#f3f4f6" stroke="#d1d5db"/>
  <text x="558" y="272" font-size="11" fill="#374151" text-anchor="middle">TP / PP / EP</text>
</svg>
</div>

关键设计：**API Server 与 EngineCore 分进程**。前端的 tokenize、HTTP 解析、JSON 序列化这些 CPU 密集活儿不再阻塞 GPU 调度循环，两边通过 ZMQ 通信。这也是后来 PD 分离能自然长出来的基础。

## 4. 转折：0.25 为什么把 PagedAttention 删了

这是 2026 年 7 月最值得说的一件事。

**vLLM v0.25.0（7/11-7/12 发布）** 做了一件反直觉的事：**移除 PagedAttention**（PR #47361），同时把 **Model Runner V2 变成所有稠密模型的默认路径**（#39337）。

看起来像是自废武功，其实是**抽象层过期了**。

### 4.1 为什么能删

PagedAttention 最初是一个"中间层"：因为当时的注意力 kernel 只会读**连续**的 KV，所以 vLLM 需要额外一层来做逻辑—物理转换。

但到了 2026 年，**现代注意力后端（FlashAttention 3/4、FlashInfer）已经在 kernel 内部原生支持读 block table 并 gather 分页 KV**。也就是说，"分页"这件事本身已经下沉进了 kernel。

那么 vLLM 里那层 PagedAttention 抽象就变成了**纯粹的冗余开销**——多一次 Python 侧的调度、多一层张量整形、多一批不必要的同步。删掉它，分页语义不变（块表还在、KVCacheManager 还在），只是执行路径变直了。

<div class="keybox">
<strong>要区分两件事：</strong>被删掉的是 vLLM 内部那个叫 <code>PagedAttention</code> 的<strong>算子抽象层</strong>；<strong>分页 KV Cache 这个机制本身依然存在</strong>，只是由注意力后端 kernel 直接完成。这是"抽象下沉"，不是"功能退化"。
</div>

### 4.2 MRv2：把整步 decode 录成一张图

同一版里 Model Runner V2 转正，解决的是上一篇说的**第三个瓶颈：GPU 等 CPU**。

MRv2 的核心是 **async-first + 零 CPU-GPU 同步**：

- 前向路径上所有需要"把结果拷回 CPU 判断一下"的地方全部消除，改成 GPU 上就地处理；
- 因为没有 host 同步点了，**整个 decode step（包括投机解码的 draft + verify）可以被捕获成一张完整的 CUDA Graph**；
- step N 和 step N+1 可以重叠——CPU 提前一整步准备下一步的输入。

效果有多大？单 token 的 kernel 启动开销从 **~300 µs 塌缩到 ~5 µs**。

<div class="fig">
<svg viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#b91c1c">MRv1：每步都有同步点</text>
  <rect x="16" y="26" width="70" height="24" rx="3" fill="#bfdbfe" stroke="#3b82f6"/><text x="51" y="42" font-size="10" fill="#1e3a8a" text-anchor="middle">CPU 准备</text>
  <rect x="90" y="26" width="90" height="24" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="135" y="42" font-size="10" fill="#065f46" text-anchor="middle">GPU 前向</text>
  <rect x="184" y="26" width="54" height="24" rx="3" fill="#fecaca" stroke="#ef4444"/><text x="211" y="42" font-size="10" fill="#7f1d1d" text-anchor="middle">D2H</text>
  <rect x="242" y="26" width="60" height="24" rx="3" fill="#fed7aa" stroke="#f97316"/><text x="272" y="42" font-size="10" fill="#7c2d12" text-anchor="middle">CPU 判断</text>
  <rect x="306" y="26" width="54" height="24" rx="3" fill="#fecaca" stroke="#ef4444"/><text x="333" y="42" font-size="10" fill="#7f1d1d" text-anchor="middle">H2D</text>
  <rect x="364" y="26" width="70" height="24" rx="3" fill="#bfdbfe" stroke="#3b82f6"/><text x="399" y="42" font-size="10" fill="#1e3a8a" text-anchor="middle">CPU 准备</text>
  <rect x="438" y="26" width="90" height="24" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="483" y="42" font-size="10" fill="#065f46" text-anchor="middle">GPU 前向</text>
  <text x="540" y="42" font-size="10" fill="#6b7280">…</text>
  <text x="16" y="66" font-size="10" fill="#b91c1c">GPU 空闲区间：CPU 准备 + D2H + 判断 + H2D ≈ 单步一半时间</text>
  <rect x="184" y="54" width="176" height="6" fill="#fca5a5"/>

  <line x1="16" y1="84" x2="664" y2="84" stroke="#e5e7eb"/>

  <text x="16" y="106" font-size="12" font-weight="700" fill="#047857">MRv2：整步录成 CUDA Graph，CPU 领先一步</text>
  <rect x="16" y="114" width="70" height="24" rx="3" fill="#bfdbfe" stroke="#3b82f6"/><text x="51" y="130" font-size="10" fill="#1e3a8a" text-anchor="middle">CPU 备 N</text>
  <rect x="90" y="114" width="70" height="24" rx="3" fill="#bfdbfe" stroke="#3b82f6"/><text x="125" y="130" font-size="10" fill="#1e3a8a" text-anchor="middle">CPU 备 N+1</text>
  <rect x="164" y="114" width="70" height="24" rx="3" fill="#bfdbfe" stroke="#3b82f6"/><text x="199" y="130" font-size="10" fill="#1e3a8a" text-anchor="middle">CPU 备 N+2</text>
  <text x="244" y="130" font-size="10" fill="#6b7280">…（CPU 始终领先）</text>

  <rect x="90" y="144" width="112" height="24" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="146" y="160" font-size="10" fill="#065f46" text-anchor="middle">GPU 图 N（一张图）</text>
  <rect x="206" y="144" width="112" height="24" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="262" y="160" font-size="10" fill="#065f46" text-anchor="middle">GPU 图 N+1</text>
  <rect x="322" y="144" width="112" height="24" rx="3" fill="#a7f3d0" stroke="#10b981"/><text x="378" y="160" font-size="10" fill="#065f46" text-anchor="middle">GPU 图 N+2</text>
  <text x="442" y="160" font-size="10" fill="#047857">GPU 连续无气泡</text>

  <text x="16" y="188" font-size="11" fill="#6b7280">图 3：kernel 启动开销 ~300 µs → ~5 µs。收益在中小 batch（真实 Agent 流量区间）最明显。</text>
</svg>
</div>

### 4.3 代价与坑

<div class="warnbox">
<strong>0.25 是一次大跨度换代，升级必须做完整回归：</strong>
<ul>
<li>自定义算子 / 老架构 GPU 会回退到 MRv1 路径，拿不到收益；</li>
<li><strong>Transformers v4 被弃用</strong>（#40389），需迁到 v5；</li>
<li>编译要求提到 <strong>C++20</strong>；</li>
<li>旧的 partial-prefill 参数被移除（#49244）。</li>
</ul>
</div>

紧接着的 **v0.25.1（7/14）** 是一个两 commit 的必打补丁：

- **#48330 混合 dtype 量化融合守卫**——修 NVFP4 模型**静默输出乱码**。根因很值得看：FlashInfer 的 `allreduce + RMSNorm + static-quant` 三合一融合核，在激活是 BF16、RMSNorm 权重是 FP32 时 dtype 不一致，把 4-bit NVFP4 读成了错误位模式，隐状态损坏，输出变成重复的 `!!!!!`。修复方式是加一个 dtype 匹配哨兵：dtype 不一致走安全路径，一致时保留融合。
- **#47888** TorchCodec 缺 FFmpeg 时不再阻塞启动。

<div class="keybox">
一句话看懂 #48330：<strong>融合核省 HBM 往返是真快，但"默认 dtype 一致"这个隐含假设会破。</strong>加一个 dtype 哨兵就能兼得速度与正确性——这是所有激进 kernel 融合都要面对的经典权衡。
</div>

## 5. vLLM 的差异化优势

跑了这么多版本跟踪下来，vLLM 最稳的护城河其实是**广度**：

- **模型广度**：`Transformers backend parity`（0.25 起 Transformers 后端速度追平原生实现），意味着**只要 HF 上有实现的新架构，Day-0 就能全速服务**，不用等 vLLM 写原生实现。
- **硬件广度**：CUDA / ROCm（AITER）/ Intel XPU（DeepSeek-V4 `fuse_index_q` SYCL 路径）/ TPU / CPU，是所有引擎里最全的。
- **量化广度**：FP8 / INT4 / AWQ / GPTQ / NVFP4 / MXFP4 / compressed-tensors 全支持。
- **生态广度**：KV Connector 接口（可对接 Mooncake / NIXL / LMCache）、Rust router、二级 KV 缓存 TieringManager（PR #42285）。

## 6. 小结

| 机制 | 解决什么 | 现状 |
|---|---|---|
| 分页 KV + 块表 | 显存碎片、前缀共享 | 保留；算子层下沉到 attention 后端 |
| 连续批处理 + chunked prefill | 队头阻塞 | 稳定基石 |
| API/Engine 分进程（V1） | CPU 前端阻塞 GPU 循环 | 稳定；PD 分离的基础 |
| **Model Runner V2** | GPU 等 CPU 的空转 | **0.25 起所有稠密模型默认** |
| 全 CUDA Graph 捕获 | kernel 启动开销 | 300 µs → 5 µs |
| Transformers backend parity | 新模型上线速度 | Day-0 全速服务 |

下一篇看 SGLang——它从完全不同的角度切入：不是从"显存怎么管"，而是从"**LLM 程序的结构长什么样**"出发。
