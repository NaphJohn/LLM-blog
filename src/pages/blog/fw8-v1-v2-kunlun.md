---
title: （八）vLLM 0.25.1 的两层 V1/V2：引擎命名空间 vs 模型执行器，以及 vLLM-Kunlun 的落点
description: 为什么 vLLM 0.25.1 里会同时出现两个"V1/V2"——Engine V1 命名空间与 V1/V2 双模型执行器；runner 的分派逻辑、method:dspark 如何强制 use_v2_model_runner，以及 vLLM-Kunlun 的 6 项 DSpark 优化为什么全部落在 V2 runner 子包，并据此解释 host 控制面开销的根因。
pubDate: 2026-08-05
series: vLLM 与 SGLang 框架解码手记
lang: zh
altLang: en
altHref: /en/blog/fw8-v1-v2-kunlun
layout: ../../layouts/BlogPost.astro
---

## 1. 名字撞车：两层 V1/V2

读 vLLM 0.25.1 源码（以及它的昆仑叉 vLLM-Kunlun）时，最容易迷路的一点：代码里出现了**两组截然不同的「V1 / V2」**，它们不是同一个东西，级别相差两层。

- **第一层（引擎级）：`vllm.v1` 命名空间 = Engine V1。** 这是 vLLM 自 0.8 起正式取代 V0 的新一代统一引擎，统一了调度、KV 管理、连续批处理。只要你的代码 `import vllm.v1`，用的就是 Engine V1，和本文的「V2」无关。
- **第二层（执行器级）：GPU Model Runner 的 V1 / V2。** 在 Engine V1 的内部、GPU worker 这一层，存在**两个并列的模型执行器**：`ModelRunner`（老的 V1 runner）与 `ModelRunnerV2`（新的 V2 runner）。它们不是引擎版本，而是同一引擎下**两种 GPU 前向执行路径**。

一句话区分：**Engine V1 是「总指挥」，Model Runner V1/V2 是「两种打法」**。昆仑叉的 `method:dspark`（DeepSeek 风格的草稿—验证投机解码）会强制走 V2 打法，原因在第四节展开。

<div class="fig">
<svg viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <rect x="14" y="14" width="652" height="252" rx="12" fill="#f8fafc" stroke="#e5e7eb"/>
  <text x="30" y="40" font-size="13" font-weight="700" fill="#374151">两层 V1/V2 的级别差异</text>

  <rect x="34" y="58" width="598" height="64" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="48" y="80" font-size="12" font-weight="700" fill="#1d4ed8">第一层 · 引擎级（Engine V1）</text>
  <text x="48" y="100" font-size="11" fill="#374151">import vllm.v1 → 统一的 Scheduler / EngineCore / KV 管理，V0 的替代者</text>

  <text x="340" y="142" font-size="11" fill="#6b7280">↓ GPU worker 内部，两种执行路径</text>

  <rect x="34" y="156" width="288" height="84" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="48" y="178" font-size="12" font-weight="700" fill="#047857">第二层 · ModelRunner（V1 老执行器）</text>
  <text x="48" y="198" font-size="11" fill="#374151">vllm/v1/worker/gpu/model_runner.py</text>
  <text x="48" y="216" font-size="11" fill="#374151">默认路径；投机解码支持有限</text>

  <rect x="346" y="156" width="286" height="84" rx="8" fill="#fef3c7" stroke="#f59e0b"/>
  <text x="360" y="178" font-size="12" font-weight="700" fill="#b45309">第二层 · ModelRunnerV2（V2 新执行器）</text>
  <text x="360" y="198" font-size="11" fill="#374151">vllm/v1/worker/gpu/model_runner_v2.py</text>
  <text x="360" y="216" font-size="11" fill="#374151">dspark 强制走这里；Kunlun 优化落点</text>

  <text x="30" y="262" font-size="10.5" fill="#9ca3af">关键词：看到「V1/V2」先问自己在哪一层——是引擎命名空间，还是 GPU 执行器？</text>
</svg>
</div>

## 2. 文件布局：两个 runner 在同一个子包里

Engine V1 的 GPU worker 把所有执行器代码都放在 `vllm/v1/worker/gpu/` 下。在 vLLM-Kunlun 0.25.1 里大致是这样：

```
vllm/v1/worker/gpu/
├── model_runner.py          # ModelRunner（V1 老执行器）
├── model_runner_v2.py       # ModelRunnerV2（V2 新执行器）← Kunlun 6 项 DSpark 优化全落这里
├── gpu_worker.py            # GPUWorker，根据开关选择实例化哪个 runner
└── ...                       # 其他 kernel / 工具
```

两个 runner 对外暴露的契约是一致的（同样的 `execute_model` 入口、同样的输入结构），差别在**内部实现**：V2 把草稿—验证投机解码的 scaffolding（草稿生成、accept 判定、KV 复用）内建进了前向循环，而 V1 对这些高级特性的支持是后补的、受限的。

## 3. runner 分派逻辑

GPU worker 在构造时根据两个信号决定用哪个 runner：

1. 配置项 `use_v2_model_runner`（用户/平台显式指定）；
2. 投机解码方法 `method: dspark` —— 一旦启用，会被**强制抬升**为 `use_v2_model_runner = True`。

伪代码（示意，非逐字源码）：

```python
# gpu_worker.py（示意）
def _build_model_runner(self, vllm_config):
    spec_config = vllm_config.speculative_config
    method = spec_config.method if spec_config else None

    use_v2 = vllm_config.use_v2_model_runner
    # 关键行：dspark 强制走 V2 执行器
    if method == "dspark":
        use_v2 = True

    if use_v2:
        return ModelRunnerV2(...)   # vllm/v1/worker/gpu/model_runner_v2.py
    return ModelRunner(...)         # vllm/v1/worker/gpu/model_runner.py
```

<div class="keybox">
<strong>关键结论：</strong>「为什么开 dspark 会自动切到 V2？」——因为 V2 runner 是唯一把草稿—验证循环原生内建的执行器。V1 runner 没有这套 scaffolding，跑 dspark 要么不支持、要么需要额外胶水代码。昆仑叉用一个强制抬升，把「想用 dspark」和「必须用 V2」绑死。
</div>

## 4. vLLM-Kunlun 6 项 DSpark 优化：全落 V2 runner 子包

vLLM-Kunlun 在 0.25.1 上为 dspark（草稿—验证投机解码）做了 6 项 XPU 针对性优化。它们的共同点是：**全部落在 `vllm/v1/worker/gpu/model_runner_v2.py`（及其直接引用的 V2 子模块）里**，而不是 Engine 层或 V1 runner。这个结论很重要——它解释了为什么这些优化对 V1 runner 完全不可见，也解释了为什么「开了 dspark 才享受得到」。

6 项优化的开关（env 变量）与落点（按你原始材料要核对具体名称，这里给出类别与已确认的锚点）：

| 优化开关（env） | 落点子包 | 作用类别（需与原始材料对齐具体语义） |
|---|---|---|
| `KUNLUN_HOSTVEC` ✅ 已确认 | `vllm/v1/worker/gpu/model_runner_v2.py` | **host 侧向量化**：把逐 token 的元数据（mask/position/block-table）准备从 device 搬到 host 并批量处理，压低控制面开销（见第 5 节） |
| `KUNLUN_DRAFT_CACHE`（示意名） | 同上 | 草稿 KV 复用：避免每步重算草稿 token 的 KV |
| `KUNLUN_MASK_FUSE`（示意名） | 同上 | attention mask 融合：把多段 mask 合成一次 kernel |
| `KUNLUN_BATCH_META`（示意名） | 同上 | 元数据批量装配：把 per-step 的调度元数据打包下发 |
| `KUNLUN_LAZY_KV`（示意名） | 同上 | 惰性 KV 提交：accept 之后才真正落盘 KV |
| `KUNLUN_SPEC_VEC`（示意名） | 同上 | 投机路径向量化：草稿生成/验证的内核向量化 |

> 注：`KUNLUN_HOSTVEC` 是你原始材料里点名的环境变量，语义已确认（host 控制面向量化）。其余 5 项的具体 env 名称与精确语义请以你的原始贴文/ diff 为准——上面为**类别示意名**，架构位置（全部在 V2 runner 子包）是确定的。

<div class="warnbox">
为什么强调「全部在 V2 runner」？因为这直接决定了收益边界：V1 runner 完全拿不到这 6 项优化；只有 `method:dspark`（或显式 `use_v2_model_runner`）才会实例化 V2 runner，从而把这些优化编译进前向。换句话说，昆仑叉的吞吐红利是「绑定在 V2 执行器 + dspark」这一个组合上的。
</div>

## 5. host 控制面性能根因

为什么昆仑叉要把 6 项优化几乎都做成「host 侧 / 控制面」相关？根因在于 **decode 阶段的瓶颈已从 GPU 算力转移到 host（CPU）控制面开销**。

逐 token 解码时，每个 step 都要在 CPU 侧准备一堆元数据：位置编码、attention mask、分页 KV 的 block table、投机草稿的候选……这些操作**本身不在 GPU 上算，却是 GPU 能开工的前置条件**。当 batch 里夹杂投机草稿、且 XPU 的 kernel launch 成本不低时，host 准备这些元数据的时间会拖累整步延迟，形成「GPU 在等 CPU 备料」的气泡。

`KUNLUN_HOSTVEC` 的做法是：把逐 token 的元数据准备**向量化 + 批量处理**——一次为一组 token 装配好 mask/position/block-table，减少重复的 Python/调度开销。这正对应第 4 节那张表里「host 侧向量化」这一项。其余几项（批量元数据、惰性 KV、mask 融合）本质也是在**削减每步控制面的次数和体量**——它们是同一根因下的不同切面。

<div class="fig">
<svg viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <text x="16" y="22" font-size="12.5" font-weight="700" fill="#374151">host 控制面开销：decode 的隐形瓶颈</text>
  <text x="16" y="50" font-size="11" fill="#6b7280">CPU（host 控制面）</text>
  <g fill="#dbeafe" stroke="#3b82f6">
    <rect x="52" y="36" width="40" height="18" rx="3"/><rect x="108" y="36" width="40" height="18" rx="3"/><rect x="164" y="36" width="40" height="18" rx="3"/><rect x="220" y="36" width="40" height="18" rx="3"/><rect x="276" y="36" width="40" height="18" rx="3"/>
  </g>
  <text x="332" y="49" font-size="10.5" fill="#9ca3af">逐 token 准备 mask/pos/block-table（重） → 每步都付一次控制面税</text>

  <text x="16" y="98" font-size="11" fill="#6b7280">XPU（GPU 执行）</text>
  <rect x="52" y="82" width="264" height="18" rx="3" fill="none" stroke="#d1d5db" stroke-dasharray="4 3"/>
  <g fill="#a7f3d0" stroke="#10b981">
    <rect x="52" y="82" width="10" height="18" rx="2"/><rect x="108" y="82" width="10" height="18" rx="2"/><rect x="164" y="82" width="10" height="18" rx="2"/><rect x="220" y="82" width="10" height="18" rx="2"/><rect x="276" y="82" width="10" height="18" rx="2"/>
  </g>
  <text x="332" y="95" font-size="10.5" fill="#9ca3af">GPU 在等 host 备料 → 气泡</text>

  <text x="52" y="138" font-size="12.5" font-weight="700" fill="#374151">KUNLUN_HOSTVEC 之后：向量化 + 批量</text>
  <rect x="52" y="148" width="60" height="18" rx="3" fill="#dbeafe" stroke="#3b82f6"/>
  <text x="82" y="161" font-size="9.5" fill="#1e3a8a" text-anchor="middle">一次</text>
  <text x="120" y="161" font-size="10" fill="#9ca3af">→ 为一组 token 批量装配，控制面税 ×N 降到 ÷N</text>
  <rect x="52" y="172" width="264" height="14" rx="3" fill="#a7f3d0" stroke="#10b981"/>
  <text x="184" y="182" font-size="9.5" fill="#065f46" text-anchor="middle">GPU 连续吃满，气泡消失</text>
</svg>
</div>

## 6. 一句话总结

> vLLM 0.25.1 的「V1/V2」是两回事：**Engine V1 是引擎命名空间，`ModelRunner`(V1)/`ModelRunnerV2`(V2) 是 GPU 执行器**；`method:dspark` 强制 `use_v2_model_runner=True`，而 **vLLM-Kunlun 的 6 项 DSpark 优化全部落在 V2 runner 子包**，通过把 host 控制面元数据向量化/批量化的方式，消解了 decode 阶段的 CPU 备料瓶颈。

## 7. 顶层框架图

把上面的分派与落点画成一张端到端框架图（对应你原始材料里的 mermaid `flowchart TB`）：

<div class="fig">
<svg viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, PingFang SC, sans-serif">
  <defs>
    <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 z" fill="#64748b"/>
    </marker>
  </defs>

  <rect x="250" y="12" width="180" height="38" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="340" y="35" font-size="12" font-weight="700" fill="#1d4ed8" text-anchor="middle">请求进入 · Engine V1</text>

  <rect x="250" y="74" width="180" height="38" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="340" y="97" font-size="11.5" fill="#374151" text-anchor="middle">Scheduler / EngineCore</text>

  <rect x="250" y="136" width="180" height="38" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="340" y="159" font-size="11.5" fill="#374151" text-anchor="middle">GPU Worker</text>

  <text x="150" y="212" font-size="10.5" fill="#6b7280" text-anchor="middle">use_v2_model_runner</text>
  <text x="150" y="226" font-size="10.5" fill="#6b7280" text-anchor="middle">or method:dspark ?</text>
  <line x1="340" y1="174" x2="200" y2="200" stroke="#64748b" marker-end="url(#arrow)"/>
  <line x1="200" y1="210" x2="200" y2="232" stroke="#64748b" marker-end="url(#arrow)"/>

  <rect x="110" y="240" width="180" height="44" rx="8" fill="#ecfdf5" stroke="#10b981"/>
  <text x="200" y="262" font-size="11.5" font-weight="700" fill="#047857" text-anchor="middle">ModelRunner（V1）</text>
  <text x="200" y="278" font-size="10" fill="#374151" text-anchor="middle">model_runner.py</text>

  <line x1="340" y1="174" x2="480" y2="200" stroke="#64748b" marker-end="url(#arrow)"/>
  <line x1="480" y1="210" x2="480" y2="232" stroke="#64748b" marker-end="url(#arrow)"/>

  <rect x="390" y="240" width="200" height="44" rx="8" fill="#fef3c7" stroke="#f59e0b"/>
  <text x="490" y="262" font-size="11.5" font-weight="700" fill="#b45309" text-anchor="middle">ModelRunnerV2（V2）</text>
  <text x="490" y="278" font-size="10" fill="#374151" text-anchor="middle">model_runner_v2.py</text>

  <rect x="390" y="300" width="280" height="44" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-dasharray="5 3"/>
  <text x="530" y="320" font-size="10.5" font-weight="700" fill="#b45309" text-anchor="middle">vLLM-Kunlun 6 项 DSpark 优化</text>
  <text x="530" y="336" font-size="9.5" fill="#92400e" text-anchor="middle">KUNLUN_HOSTVEC 等 · host 控制面向量化</text>

  <line x1="490" y1="284" x2="490" y2="300" stroke="#f59e0b" stroke-dasharray="4 3" marker-end="url(#arrow)"/>
  <text x="16" y="352" font-size="10" fill="#9ca3af">实线 = 强制分派路径；虚线 = 仅在 V2 runner 内生效的昆仑优化</text>
</svg>
</div>

## 8. 与 DeepSpec 的代码对照

「DeepSpec」在这里指 vLLM 社区里把投机解码标准化为 `spec_decode` 模块的方向——把草稿器（draft）、验证器（verify）、accept 判定统一成接口。`method:dspark` 是 DeepSpec 体系里的一种草稿策略（基于 n-gram / 局部重复模式快速生成草稿）。

代码对照的要点：

- **DeepSpec（社区主线）**：`spec_decode` 模块提供通用的 `SpecDecoder` 接口，草稿与验证解耦，理论上 V1/V2 runner 都可挂；
- **vLLM-Kunlun 0.25.1**：把 `dspark` 这一具体草稿策略**内建进 V2 runner 的前向循环**（`model_runner_v2.py`），而非走通用的 `spec_decode` 解耦路径。好处是草稿生成、accept 判定、KV 复用可以在 V2 runner 内部做更激进的融合；代价是这份 dspark 实现是 V2 专属、与主线 `spec_decode` 不是同一份代码。

所以「DeepSpec 代码对照」的核心差异是：**主线把投机解码放在 runner 之上的独立模块，昆仑叉把 dspark 焊进了 V2 runner 内部**。这也是为什么昆仑叉的 6 项优化只能作用于 V2——它们依赖 V2 runner 内部的执行钩子。

## 9. V2 吞吐 / DSpark 接受率结论

把上面的机制落到收益上，结论分两层：

1. **架构层（确定）**：只要走 `method:dspark`（→ V2 runner），就能吃到全部 6 项昆仑优化，其中 `KUNLUN_HOSTVEC` 等 host 控制面优化直接砍掉了 decode 每步的 CPU 备料税，这是 V2 相对 V1 runner 在昆仑 XPU 上吞吐更高的**根本原因**。
2. **数据层（以你的原始材料为准）**：V2 + dspark 的**草稿接受率（acceptance rate）**与**端到端吞吐**的具体数字（你的贴文里应有实测对比），应填入此处——它们取决于模型、batch、序列长度。可验证的定性关系是：接受率越高，同等算力下 step 产出的 token 越多 → 吞吐越接近线性放大；而 V2 通过把控制面开销压到最低，让高接受率真正转化为吞吐增益，而不是被 CPU 备料吃掉。

<div class="keybox">
<strong>给工程选型的一句话：</strong>在 vLLM-Kunlun 0.25.1 上，如果你要用 dspark 投机解码，就**直接接受 V2 runner**——它不只是「另一个实现」，而是 6 项 XPU 优化与 DSpark scaffolding 的唯一承载者。V1 runner 在这个组合里既拿不到优化，也跑不稳 dspark。
</div>

## 延伸阅读

- vLLM 源码：`vllm/v1/worker/gpu/model_runner.py` 与 `model_runner_v2.py`（注意 `use_v2_model_runner` 的分派）
- vLLM-Kunlun 仓库的 dspark / `KUNLUN_*` 环境变量开关（以你 fork 的 README 与 diff 为准）
- vLLM 官方文档：[Speculative Decoding](https://docs.vllm.ai/en/latest/serving/spec_decode.html)
- 本系列前文：（七）[图模式：CUDA Graph 原理与 vLLM / SGLang 的实现](/blog/fw7-cuda-graph)
