---
title: （七）图模式：CUDA Graph 原理与 vLLM / SGLang 的实现
description: 为什么 decode 是 CPU bound、CUDA Graph 的 capture/replay 机制与三条硬约束，以及 vLLM 的分段图+全图双模式分发和 SGLang 的 CudaGraphRunner + 默认开启的 PCG。
pubDate: 2026-08-04
series: vLLM 与 SGLang 框架解码手记
lang: zh
altLang: en
altHref: /en/blog/fw7-cuda-graph
layout: ../../layouts/BlogPost.astro
---

## 1. 为什么需要图模式：decode 是 CPU bound

前面几篇讲的 PagedAttention、前缀复用、连续批处理，优化目标都是**显存与计算量**。但把这两样都优化完之后，推理引擎会撞上第三堵墙：**CPU 的 kernel 发射开销**。

decode 阶段有个特点：每步只算 `batch × 1` 个 token，矩阵又扁又小，每个 kernel 在 GPU 上**实际执行只有几微秒**。而 CPU 端发射一个 kernel 的固定开销——参数装配、driver 调用、stream 提交——大约是 **5–10 μs**。一次 forward 有多少个 kernel？一个 60 层的模型，每层 RMSNorm、QKV 投影、RoPE、attention、残差加、MLP 三四个 GEMM……加起来**几百到上千个**。

算一笔账：1000 个 kernel × 7 μs 发射开销 ≈ **7 ms 的 CPU 时间**；而 GPU 执行同样这些 kernel 可能只要 2–3 ms。结论很反直觉：**decode 的瓶颈不在 GPU，而在 CPU 发射 kernel 的速度**。GPU 大部分时间在等 CPU 喂活，利用率曲线像锯齿，大量"空转气泡"。

<div class="fig">
<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="18" font-size="12" font-weight="700" fill="#374151">Eager：CPU 逐个发射，GPU 空转</text>
  <text x="16" y="52" font-size="11" fill="#6b7280">CPU</text>
  <g fill="#dbeafe" stroke="#3b82f6">
    <rect x="52" y="38" width="30" height="18" rx="3"/><rect x="102" y="38" width="30" height="18" rx="3"/><rect x="152" y="38" width="30" height="18" rx="3"/><rect x="202" y="38" width="30" height="18" rx="3"/><rect x="252" y="38" width="30" height="18" rx="3"/><rect x="302" y="38" width="30" height="18" rx="3"/><rect x="352" y="38" width="30" height="18" rx="3"/><rect x="402" y="38" width="30" height="18" rx="3"/><rect x="452" y="38" width="30" height="18" rx="3"/><rect x="502" y="38" width="30" height="18" rx="3"/><rect x="552" y="38" width="30" height="18" rx="3"/><rect x="602" y="38" width="30" height="18" rx="3"/>
  </g>
  <text x="16" y="90" font-size="11" fill="#6b7280">GPU</text>
  <rect x="52" y="74" width="580" height="18" rx="3" fill="none" stroke="#d1d5db" stroke-dasharray="4 3"/>
  <g fill="#a7f3d0" stroke="#10b981">
    <rect x="52" y="74" width="12" height="18" rx="2"/><rect x="102" y="74" width="12" height="18" rx="2"/><rect x="152" y="74" width="12" height="18" rx="2"/><rect x="202" y="74" width="12" height="18" rx="2"/><rect x="252" y="74" width="12" height="18" rx="2"/><rect x="302" y="74" width="12" height="18" rx="2"/><rect x="352" y="74" width="12" height="18" rx="2"/><rect x="402" y="74" width="12" height="18" rx="2"/><rect x="452" y="74" width="12" height="18" rx="2"/><rect x="502" y="74" width="12" height="18" rx="2"/><rect x="552" y="74" width="12" height="18" rx="2"/><rect x="602" y="74" width="12" height="18" rx="2"/>
  </g>
  <text x="52" y="112" font-size="10" fill="#9ca3af">虚线 = GPU 空转气泡：kernel 执行几 μs，launch 开销 5–10 μs，GPU 在等 CPU</text>

  <text x="16" y="148" font-size="12" font-weight="700" fill="#374151">图模式：一次 launch 重放整图</text>
  <text x="16" y="182" font-size="11" fill="#6b7280">CPU</text>
  <rect x="52" y="166" width="46" height="18" rx="3" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="75" y="179" font-size="10" fill="#1e3a8a" text-anchor="middle">launch</text>
  <text x="112" y="179" font-size="10" fill="#9ca3af">← 之后 CPU 空闲，去调度下一批请求</text>
  <text x="16" y="220" font-size="11" fill="#6b7280">GPU</text>
  <rect x="52" y="204" width="580" height="18" rx="3" fill="#a7f3d0" stroke="#10b981"/>
  <text x="342" y="217" font-size="10" fill="#065f46" text-anchor="middle">整张图的 kernel 背靠背执行，无气泡</text>
</svg>
</div>

## 2. CUDA Graph：录制一次，重放千次

CUDA Graph 是 NVIDIA 在 CUDA 10 引入的机制，思路非常直接：

- **Capture（捕获）**：把一段 kernel 序列连同每个 kernel 的参数、显存地址**录制成一张有向无环图**，存成图对象；
- **Replay（重放）**：之后每次执行，CPU 只需**一次 launch** 把整张图提交给 GPU，图里的 kernel 由 GPU 驱动器背靠背调度，CPU 开销从 O(kernel 数) 降到 **O(1)**；
- 图还可以被**更新**（`cudaGraphExecUpdate`）局部改参数，但推理引擎一般不用，直接靠静态 buffer 换数据。

天下没有免费的午餐，图模式有三条硬约束：

1. **形状固定**：图里每个 kernel 的 grid/block、tensor shape 在录制时就定死了；
2. **显存地址固定**：重放时用的是录制时录下来的指针，输入输出必须放在固定地址的 buffer 里；
3. **无 CPU↔GPU 同步、无动态控制流**：图里没有 `if batch > 32` 这种东西，也不能在中途 `.item()` 把数据读回 CPU。

所以所有推理引擎的图模式都收敛到同一套工程配方：

| 配方 | 解决的问题 |
|---|---|
| **静态输入/输出 buffer** | 地址固定：replay 前把真实输入 copy 进静态 buffer，算完从静态输出 buffer 读回 |
| **按离散尺寸捕获一组图** | 形状固定：为 bs = 1, 2, 4, 8, … 各录一张图 |
| **运行时向上 padding** | 实际 bs = 100 → 用 bs = 128 的图，多算 28 行再丢掉 |
| **超出最大捕获尺寸回退 eager** | 图不可能无限多，大 batch 本身 GPU 利用率高，不缺这点开销 |

padding 听起来浪费，但 decode 的 kernel 是访存 bound 的——多算几行 padding token 的代价远小于 CPU launch 开销，这笔交换几乎总是赚的。

## 3. vLLM：分段图打底，全图收口，dispatcher 动态分发

vLLM V1 把图模式和 torch.compile 的分段编译绑在了一起，形成一套统一框架。

### 3.1 PIECEWISE：在 attention 处切图

V1 的默认编译档位会把整个 forward 的 FX 图**在每个 attention 算子处切开**，只对 **attention 之间的片段**（embedding、RMSNorm、QKV 投影、MoE/MLP、残差……基本都是逐 token 的计算）捕获 CUDA Graph，**attention 本身保持 eager**。

为什么这么切？因为 attention 是图兼容的"老大难"：序列长度可变、分页 KV 的块表索引、varlen 的 cu_seqlens……让它进图需要后端专门适配。而 attention 之间的计算是纯粹的逐 token 算子，形状只由 token 数决定，天然适合录图。为了让分段图的显存可以被细粒度复用，V1 还把 attention 改写成**输出张量作为输入传入**的形式——attention 不自己分配输出，而是写到图管理好的 buffer 里。

### 3.2 FULL：连 attention 也录进图

对于**形状完全均匀的 pure decode batch**（每条请求都恰好生成 1 个 token，没有 prefill 混入），如果使用图兼容的 attention 后端（FlashAttention / FlashInfer 的 decode kernel 等，它们把 seq_lens 放在 GPU buffer 里、kernel 内部按 buffer 内容寻址），就可以把**整个 forward 含 attention** 录成一张图，连 attention 的 launch 开销也消掉。对小模型和 MoE 的低延迟场景收益明显。

### 3.3 五级 cudagraph_mode 与运行时分发

vLLM 把上面的能力组合成五档：

| 模式 | 行为 | 适用 |
|---|---|---|
| `NONE` | 全程 eager | 调试、图不兼容的模型 |
| `PIECEWISE` | 只用分段图 | pooling 模型等 |
| `FULL` | 只用全图 | 纯 decode 的小模型/短 prompt 负载 |
| `FULL_DECODE_ONLY` | 均匀 decode 走全图，其余 eager | PD 分离里的 decode 实例，省分段图的显存 |
| `FULL_AND_PIECEWISE`（默认） | 均匀 decode 走全图，其余走分段图 | 通用最强，显存占用也最高 |

后两档是**双模式**：运行时由一个 **dispatcher 按 batch 组成动态分发**——这个 batch 是均匀 decode 就用全图，混入 prefill/extend 就切分段图，cascade attention 这类特殊情况强制走 PIECEWISE，都不匹配就降级 eager。如果某个 attention 后端不支持当前档位，框架会自动降级到最接近的可用模式。

### 3.4 捕获尺寸

捕获哪些 batch 尺寸由 `cudagraph_capture_sizes` 控制（默认自动生成一组档位，上限到 `max_num_seqs`）：

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 48]}'
```

调优的经验法则是：**让你的典型并发档位出现在捕获列表里**。有实测案例仅把并发 16 加进捕获列表，吞吐就提升了 8%；切到 `FULL_DECODE_ONLY` 后再提升 25%。代价是每多一个尺寸就多占一份显存和捕获时间，V1 的图占用比 V0 明显更多，显存紧张时需要裁剪列表或用 `enforce_eager` 全关。

## 4. SGLang：decode 全图 + prefill 分段图，两条独立路径

SGLang 没有走"统一 dispatcher"路线，而是给 decode 和 prefill 各做了一个 runner。

### 4.1 Decode：CudaGraphRunner 全图捕获

SGLang 的 decode attention kernel **天生就是图兼容的**——序列长度、KV 写入位置等动态信息全部放在 GPU 上的静态 buffer 里，kernel 启动后自己读 buffer 寻址，不需要 CPU 侧的任何动态决策。所以 SGLang 的 decode 直接把**整个 forward 录成一张图**，没有 vLLM 那种"attention 留在外面"的妥协。

核心组件：

- **GraphInputBuffers**：预分配的静态输入 buffer——`input_ids`、`positions`、`seq_lens`、`out_cache_loc`（KV cache 写入位置）等，地址终身不变；
- **get_batch_sizes_to_capture()**：生成捕获尺寸表，1 到 `--cuda-graph-max-bs`（小 bs 密、大 bs 疏），也可用 `--cuda-graph-bs` 显式指定；
- **从大到小捕获 + 全局共享显存池**：先录最大的图，小图捕获时复用大图池里的显存，显著压低总占用；
- **运行时三分**：（1）`bisect` 找到 ≥ 实际 bs 的最小已捕获尺寸；（2）把真实 batch 数据 copy 进静态 buffer，空位 padding；（3）`graph.replay()`，从静态输出 buffer 取回结果。

如果实际 bs 超过 `cuda_graph_max_bs`，或者 batch 里有图不兼容的特性（如某些投机解码路径），就回退普通 eager 前向。

### 4.2 Prefill / Extend：PiecewiseCudaGraphRunner（PCG）

prefill 的麻烦在于 **token 数每步都变**，没法像 decode 那样按 batch size 枚举。SGLang 的 PCG（Piecewise CUDA Graph，**2026 年的版本里已默认开启**，旧的 `--enable-piecewise-cuda-graph` 标志已废弃）解法和 vLLM 分段图同源，但实现路径不同：

1. **追踪**：用 `torch.compile` + 自定义 `SGLangBackend` 拿到模型 forward 的 FX 图；
2. **切图**：在注册的 **split ops**（attention、all-reduce、MoE dispatch 等动态/通信算子）处把图切成若干段，split ops 保持 eager；
3. **分段捕获**：每一段按一组 **token 数档位**分别录图。默认档位是渐疏的——4–32 步长 4、48–256 步长 16、288–512 步长 32、576–1024 步长 64、1280–4096 步长 256、4096 以上步长 512，上限取 `--piecewise-cuda-graph-max-tokens`（非 MLA 默认等于 `chunked_prefill_size`，MLA 默认 2048）；
4. **运行时**：实际 token 数**二分查找向上取整**到最近档位，copy 进静态 buffer、零填充，逐段 replay，输出再切回真实长度；超过最大档回退普通路径。

显存优化上，PCG 同样是**全局共享显存池 + 从大到小捕获**，并且把最后一段的输出张量用弱引用持有，最大化复用。

一个对 kernel 开发者的重要细节：PCG 依赖 torch.compile 追踪，**新写的 CUDA kernel 默认不可追踪**（JIT 编译、文件 IO、动态加载都会炸），需要用 `register_custom_op` 包装成不透明节点才能进分段图。

### 4.3 与 torch.compile 叠加

SGLang 还有独立的 `--enable-torch-compile`（默认对小 bs ≤ 32 生效）：在 CUDA Graph 之内再做一层 Inductor 的 kernel 融合，图内每个 kernel 更小更少。两者是叠加关系，不是替代关系。

## 5. 同台对比

| 维度 | vLLM V1 | SGLang |
|---|---|---|
| decode 图粒度 | 双模式：均匀 decode 全图，其余分段图，dispatcher 按 batch 动态切换 | CudaGraphRunner 固定全图（decode kernel 原生图兼容） |
| prefill 图 | 分段图（PIECEWISE，与 torch.compile 分段编译一体） | PCG 分段图（独立 runner，已默认开启） |
| 切图点 | attention 算子 | split ops：attention / all-reduce / MoE dispatch |
| 捕获维度 | batch 尺寸列表 `cudagraph_capture_sizes` | decode 按 bs（`--cuda-graph-max-bs`）；prefill 按 token 数档位 |
| 显存策略 | 编译器后端管理中间 buffer；显存占用偏高，可裁尺寸列表 | 全局共享显存池 + 从大到小捕获 + 弱引用输出 |
| 回退 | 模式自动降级（FULL → PIECEWISE → NONE） | 超最大档 / 图不兼容 → eager |
| 关掉它 | `enforce_eager=True` 或 `cudagraph_mode: NONE` | `--disable-cuda-graph` / `--disable-piecewise-cuda-graph` |

设计理念的差异值得玩味：vLLM 选择了**一套框架、运行时分发**——表达力强，batch 组成什么样就用什么图，但复杂度和显存占用也最高；SGLang 选择了**两条独立、各自极致的路径**——decode 全图简单直接、PCG 专攻 prefill，工程上更"憨"但每块都好维护。两者在 padding、静态 buffer、共享显存池这些基本功上完全一致。

## 6. 工程权衡与常见坑

1. **显存换 CPU**：图模式是典型的显存换开销。SGLang 社区的经验数据是每个捕获尺寸约 100–500 MB 输入 buffer + 10–50 MB 图元数据 + 1–2 GB 一次性共享池；vLLM V1 的图占用比 V0 更多。OOM 时第一反应应该是裁剪捕获尺寸或调大 `--cuda-graph-max-bs` 的反方向。
2. **捕获时间计入启动**：几十个尺寸 × 每尺寸一次 warmup + 一次捕获，大模型启动会慢几十秒到几分钟。生产镜像可以预热后固化。
3. **padding 不是免费的**：bs=100 pad 到 128 浪费 28% 的 decode 计算，但通常仍远小于 launch 开销；真正要避的是"档位太疏导致长期 pad 一倍"的捕获表。
4. **调试期先关图**：图会掩盖真实的 kernel 调用栈，报错信息变得难读。排查精度或崩溃问题时，`enforce_eager` / `--disable-cuda-graph` 应该是第一反应。
5. **投机解码的交互**：草稿-验证结构下 batch 形状更复杂，两家都对投机路径的图兼容做了专门处理（或限制图只在特定阶段生效），开投机解码时留意图是否真的命中。

## 7. 小结

- 图模式解决的是 decode 的 **CPU launch bound**：录制一次、重放千次，CPU 开销 O(n) → O(1)；
- 三条硬约束（形状/地址固定、无同步）推导出通用配方：**静态 buffer + 按尺寸捕获 + 向上 padding**；
- **vLLM**：分段图与 torch.compile 一体，FULL / PIECEWISE 双模式由 dispatcher 按 batch 动态分发；
- **SGLang**：decode 走 CudaGraphRunner 全图，prefill 走默认开启的 PCG 分段图，两条独立路径；
- 代价是显存和启动时间——图模式是推理引擎里最典型的"用资源换确定性延迟"的设计。

下一篇我们回到调度层，看 PD 分离（Prefill/Decode Disaggregation）如何把两种完全不同负载特征的阶段拆到不同 GPU 上，以及它和图模式、连续批处理如何叠加。

## 延伸阅读

- [vLLM 官方文档：CUDA Graphs 设计](https://docs.vllm.ai/en/latest/design/cuda_graphs.html) 与 [torch.compile 整合](https://docs.vllm.ai/en/latest/design/torch_compile.html)
- [SGLang 官方文档：Piecewise CUDA Graph](https://sgl-project.github.io/advanced_features/piecewise_cuda_graph.html)
- SGLang 源码：`python/sglang/srt/model_executor/cuda_graph_runner.py`、`piecewise_cuda_graph_runner.py`
- [NVIDIA CUDA Graphs 官方指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
