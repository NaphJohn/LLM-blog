---
title: VLA 解码手记（三）：π 系列演进——Physical Intelligence 的 VLA 路线
description: 从 π0 到 π0.7，Physical Intelligence 如何把 VLA 一步步推到"超越专精策略"。核心是一张 π0 → π0.5 的结构演进时间轴（单头连续 → 离散 token → 双头分层）加结构对照表，再拆底座升级、跨本体、RL 专精与 π0.7 的记忆和组合泛化。
pubDate: 2026-08-04
series: VLA 解码手记
lang: zh
altLang: en
altHref: /en/blog/vla3-pi-family
layout: ../../layouts/BlogPost.astro
---

## 1. π0：把 VLA 做成"通用接口"（2024-10）

Physical Intelligence 的 **π0** 定了基线：

- **底座**：PaliGemma（约 3B VLM）+ 一个独立的**动作专家（action expert）**，动作专家用 **Flow Matching** 生成连续动作，输出 **50Hz** 高频控制。
- **数据**：在 OXE + 自有演示上训练，覆盖 **7 种机器人平台、68 项任务**。
- **统一接口**：语言/图像进，连续动作出。证明"一个通用 VLA 能在多本体上干活"。

## 2. π0-Fast：离散化提速

在 π0 上加**动作离散化 + 自回归**，换取推理速度。是"连续 vs 离散"两条路线的工程折中，适合延迟敏感场景。

## 3. π0.5：跨本体 + 任务分解（2025-04）

- **跨本体协同训练（cross-embodiment co-training）**：多台不同机器人共享一个"通用大脑"，学到与硬件无关的操作常识。
- **任务分解（task decomposition）**：高层把复杂指令拆成子目标，低层策略依次执行——让长时程任务可控。

## 4. 结构演进时间轴：π0 → π0.5 到底改了什么

前三节讲"能力涨了"，这一节讲"零件换了"。π 系列真正动**结构**的地方，集中在 π0 → π0.5 这一段，而且**参数量几乎没动**——改的是三件事的顺序：

1. **动作表示**：π0 是连续向量走流匹配；π0-Fast 换成离散 token 走自回归。
2. **动作头数量**：π0 / π0-Fast 只有一条头；π0.5 一头变两头（FAST 离散头 + flow 连续头）。
3. **梯度通路**：π0.5 后 PaliGemma 系发现双头联合训练会互相拉扯，于是加 Knowledge Insulation——只用 stop-grad 切断专家到骨干的梯度，模块本身不动。

<div class="fig">
<svg id="fig-pi-tl" viewBox="0 0 680 768" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" aria-labelledby="pi-tl-t pi-tl-d">
<title id="pi-tl-t">π0 到 π0.5 的结构演进时间轴</title>
<desc id="pi-tl-d">六个节点的纵向时间轴：π0 单头连续动作、π0-Fast 离散 token、π0.5 双动作头与分层推理、Knowledge Insulation 梯度阻断、π0.6 扩容与强化学习、π0.7 记忆与多模态提示。</desc>
<style>
#fig-pi-tl{--t1:#2C2C2A;--t2:#5F5E5A;--ln:#DDD9CE;
--b-p:#E6F1FB;--b-s:#185FA5;--b-t:#0C447C;
--t-p:#E1F5EE;--t-s:#1D9E75;--t-t:#04342C;
--g-p:#F1EFE8;--g-s:#CDC9BC;--g-t:#444441;}
:root[data-theme="dark"] #fig-pi-tl{--t1:#F1EFE8;--t2:#D3D1C7;--ln:#3A4046;
--b-p:#042C53;--b-s:#85B7EB;--b-t:#B5D4F4;
--t-p:#04342C;--t-s:#5DCAA5;--t-t:#9FE1CB;
--g-p:#2C2C2A;--g-s:#5F5E5A;--g-t:#D3D1C7;}
#fig-pi-tl text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
#fig-pi-tl .h1{font-size:14px;font-weight:500;fill:var(--t1)}
#fig-pi-tl .sub{font-size:11.5px;fill:var(--t2)}
#fig-pi-tl .dt{font-size:11px;fill:var(--t2);text-anchor:end;dominant-baseline:central}
#fig-pi-tl .tt{font-size:12.5px;font-weight:500;dominant-baseline:central}
#fig-pi-tl .bd{font-size:11.5px;fill:var(--t2);dominant-baseline:central}
#fig-pi-tl .c-b{fill:var(--b-p);stroke:var(--b-s);stroke-width:.5}
#fig-pi-tl .c-t{fill:var(--t-p);stroke:var(--t-s);stroke-width:.5}
#fig-pi-tl .c-g{fill:var(--g-p);stroke:var(--g-s);stroke-width:.5}
#fig-pi-tl .k-b{fill:var(--b-t)}#fig-pi-tl .k-t{fill:var(--t-t)}#fig-pi-tl .k-g{fill:var(--g-t)}
#fig-pi-tl .lk{stroke:var(--ln);stroke-width:1;fill:none}
</style>
<text class="h1" x="40" y="30">π0 → π0.5：结构演进时间轴</text>
<text class="sub" x="40" y="50">参数量几乎没动，改的是「动作表示 → 动作头数量 → 梯度通路」</text>
<line x1="118" y1="66" x2="118" y2="746" stroke="var(--ln)" stroke-width="1.5"/>

<text class="dt" x="104" y="90">2024-10</text>
<circle cx="118" cy="90" r="5" fill="#185FA5"/>
<line class="lk" x1="123" y1="90" x2="147" y2="90"/>
<rect class="c-b" x="150" y="72" width="490" height="108" rx="8"/>
<text class="tt k-b" x="166" y="96">π0 · 单头连续动作：VLA 有了统一接口</text>
<text class="bd" x="166" y="118">底座 PaliGemma 3B + 动作专家 300M，共享同一个注意力栈</text>
<text class="bd" x="166" y="135">动作走 Flow Matching：50 步 × 14 维 = 1 秒 @ 50 Hz</text>
<text class="bd" x="166" y="152">7 种本体 / 68 项任务；语言图像进，连续动作出</text>
<text class="bd" x="166" y="169">结构：单头，动作是稠密向量，中间没有语义层</text>

<text class="dt" x="104" y="214">2025-01</text>
<circle cx="118" cy="214" r="5" fill="#185FA5"/>
<line class="lk" x1="123" y1="214" x2="147" y2="214"/>
<rect class="c-b" x="150" y="196" width="490" height="92" rx="8"/>
<text class="tt k-b" x="166" y="220">π0-Fast · 把动作换成离散 token</text>
<text class="bd" x="166" y="242">FAST tokenizer 把整块动作压成离散 token，改走自回归</text>
<text class="bd" x="166" y="259">换来：可混 web 数据、单阶段训练、推理更快；代价是串行解码</text>
<text class="bd" x="166" y="276">结构：动作头退化成一张词表，连续控制变成逐 token 生成</text>

<text class="dt" x="104" y="320">2025-04</text>
<circle cx="118" cy="320" r="6" fill="#1D9E75"/>
<line class="lk" x1="123" y1="320" x2="147" y2="320"/>
<rect class="c-t" x="150" y="302" width="490" height="126" rx="8"/>
<text class="tt k-t" x="166" y="326">π0.5 · 双动作头 + 分层推理（结构拐点）</text>
<text class="bd" x="166" y="348">一头变两头：FAST 离散头（交叉熵）+ flow 连续头（流匹配）</text>
<text class="bd" x="166" y="365">损失 = CE(文本 &amp; FAST) + α·FM；预训练 α=0，后训练 α=10</text>
<text class="bd" x="166" y="382">α=0 时退化为纯 VLM，用同一个目标吃下 6 类异构数据</text>
<text class="bd" x="166" y="399">分层推理：先出子任务，再以它为条件出 50 步动作块</text>
<text class="bd" x="166" y="416">结构：多一个高层文本出口；两套动作表示用掩码互相隔离</text>

<text class="dt" x="104" y="462">2025-05</text>
<circle cx="118" cy="462" r="5" fill="#1D9E75"/>
<line class="lk" x1="123" y1="462" x2="147" y2="462"/>
<rect class="c-t" x="150" y="444" width="490" height="92" rx="8"/>
<text class="tt k-t" x="166" y="468">Knowledge Insulation · 只改梯度，不改模块</text>
<text class="bd" x="166" y="490">stop-grad：动作专家梯度不回传 VLM，骨干不被动作信号带偏</text>
<text class="bd" x="166" y="507">骨干改用 FAST token 自己学动作；收拾餐桌收敛约 7.5× 快</text>
<text class="bd" x="166" y="524">结构：改的是梯度通路与注意力掩码，网络拓扑不变</text>

<text class="dt" x="104" y="570">2025-11</text>
<circle cx="118" cy="570" r="5" fill="#B4B2A9"/>
<line class="lk" x1="123" y1="570" x2="147" y2="570"/>
<rect class="c-g" x="150" y="552" width="490" height="76" rx="8"/>
<text class="tt k-g" x="166" y="576">π0.6* · 扩容 + RECAP 强化学习</text>
<text class="bd" x="166" y="598">Gemma3 4B + 860M 动作专家；在已训 VLA 上做离线 RL 专精</text>
<text class="bd" x="166" y="615">结构：双头不变，变的是规模与后训练配方</text>

<text class="dt" x="104" y="660">2026-04</text>
<circle cx="118" cy="660" r="5" fill="#B4B2A9"/>
<line class="lk" x1="123" y1="660" x2="147" y2="660"/>
<rect class="c-g" x="150" y="642" width="490" height="94" rx="8"/>
<text class="tt k-g" x="166" y="666">π0.7 · 记忆 + 多模态提示</text>
<text class="bd" x="166" y="688">MEM 记忆机制：从过往经验里取上下文，长程多步更稳</text>
<text class="bd" x="166" y="705">图像 / 语言 / 演示混合提示，会「照着示范做」；组合泛化涌现</text>
<text class="bd" x="166" y="722">结构：双头之上再挂一条记忆与多模态提示通道</text>
</svg>
<p class="fig-cap">图 1：π 系列的结构演进时间轴——蓝框是「换表示 / 加头」的硬结构改动，绿框是梯度与训练配方的软改动，灰绿是规模与记忆的叠加</p>
</div>

关于这张图的三个读法：

- **蓝 → 绿 → 灰绿** 是一条"越来越软的改动"曲线：先换表示（硬改结构），再改损失与梯度（软改配方），最后改规模与记忆（叠加能力）。
- **π0-Fast 不是 π0 的升级版，是旁支**。它把动作头降级成一张词表换速度；π0.5 又把连续头请回来了，两者并存——这正是 π0.5 最关键的"两只脚站两条路"。
- **π0.5 的分水岭在 α**。训练时损失是 `CE(文本 & FAST) + α·FM`：预训练 α=0，整个模型退化成标准自回归 VLM，于是网页 QA、字幕、子任务标注、FAST 动作 token 能用同一个 next-token 目标一起吃；后训练 α=10，连续头才被激活学细粒度动作。**一个模型，两种训练节奏，各拿各要的东西。**

<div class="fig">
<svg id="fig-pi-cmp" viewBox="0 0 680 482" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" aria-labelledby="pi-cmp-t pi-cmp-d">
<title id="pi-cmp-t">π0 与 π0.5 的结构对照</title>
<desc id="pi-cmp-d">按输入、骨干、中间层、动作生成、输出、训练目标六行逐项对照 π0 与 π0.5。π0.5 新增 FAST 离散头与高层子任务出口，动作专家改为以子任务为条件。</desc>
<style>
#fig-pi-cmp{--t1:#2C2C2A;--t2:#5F5E5A;--ln:#DDD9CE;
--b-p:#E6F1FB;--b-s:#185FA5;--b-t:#0C447C;
--t-p:#E1F5EE;--t-s:#1D9E75;--t-t:#04342C;
--g-p:#F1EFE8;--g-s:#CDC9BC;--g-t:#444441;}
:root[data-theme="dark"] #fig-pi-cmp{--t1:#F1EFE8;--t2:#D3D1C7;--ln:#3A4046;
--b-p:#042C53;--b-s:#85B7EB;--b-t:#B5D4F4;
--t-p:#04342C;--t-s:#5DCAA5;--t-t:#9FE1CB;
--g-p:#2C2C2A;--g-s:#5F5E5A;--g-t:#D3D1C7;}
#fig-pi-cmp text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
#fig-pi-cmp .h1{font-size:14px;font-weight:500;fill:var(--t1)}
#fig-pi-cmp .sub{font-size:11.5px;fill:var(--t2)}
#fig-pi-cmp .rl{font-size:11px;fill:var(--t2);dominant-baseline:central}
#fig-pi-cmp .tt{font-size:12.5px;font-weight:500;text-anchor:middle}
#fig-pi-cmp .bd{font-size:11.5px;fill:var(--t2);dominant-baseline:central}
#fig-pi-cmp .c-b{fill:var(--b-p);stroke:var(--b-s);stroke-width:.5}
#fig-pi-cmp .c-t{fill:var(--t-p);stroke:var(--t-s);stroke-width:.5}
#fig-pi-cmp .c-g{fill:var(--g-p);stroke:var(--g-s);stroke-width:.5}
#fig-pi-cmp .k-b{fill:var(--b-t)}#fig-pi-cmp .k-t{fill:var(--t-t)}#fig-pi-cmp .k-g{fill:var(--g-t)}
</style>
<text class="h1" x="40" y="30">π0 与 π0.5 的结构对照</text>
<text class="sub" x="40" y="50">骨架没换，多了两条出口；动作头从「独立生成」变成「受子任务条件约束」</text>
<text class="tt k-b" x="248" y="82">π0 · 2024-10</text>
<text class="tt k-t" x="510" y="82">π0.5 · 2025-04</text>
<line x1="40" y1="96" x2="640" y2="96" stroke="var(--ln)" stroke-width="1"/>

<text class="rl" x="40" y="125">输入</text>
<rect class="c-b" x="128" y="102" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="125">图像 + 语言指令</text>
<rect class="c-b" x="380" y="102" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="125">图像 + 语言 + 本体状态（离散成文本）</text>

<text class="rl" x="40" y="181">骨干</text>
<rect class="c-b" x="128" y="158" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="181">PaliGemma 3B</text>
<rect class="c-b" x="380" y="158" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="181">同一骨架（语言主干 2B）</text>

<text class="rl" x="40" y="237">中间层</text>
<rect class="c-g" x="128" y="214" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="237">无中间语义层</text>
<rect class="c-t" x="380" y="214" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="237">FAST 离散头 + 子任务文本出口</text>

<text class="rl" x="40" y="293">动作生成</text>
<rect class="c-b" x="128" y="270" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="293">动作专家 300M · flow matching</text>
<rect class="c-t" x="380" y="270" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="293">同一专家 · 以子任务为条件</text>

<text class="rl" x="40" y="349">输出</text>
<rect class="c-b" x="128" y="326" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="349">连续动作块 50 × 14 · 50 Hz</text>
<rect class="c-b" x="380" y="326" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="349">同一动作块，先过一层语义</text>

<text class="rl" x="40" y="405">训练目标</text>
<rect class="c-g" x="128" y="382" width="240" height="46" rx="6"/>
<text class="bd" x="140" y="405">纯流匹配，无 α</text>
<rect class="c-g" x="380" y="382" width="260" height="46" rx="6"/>
<text class="bd" x="392" y="405">CE + α·FM，α 从 0 升到 10</text>

<text class="bd" x="40" y="443">读法：同一副骨架、同一个动作块形状。</text>
<text class="bd" x="40" y="460">差别全在「中间那层」与「损失里的 α」：前者决定能否做长任务，后者决定训练时是 VLM 还是策略。</text>
</svg>
<p class="fig-cap">图 2：π0 与 π0.5 的逐项对照——π0.5 把「一条连续出口」拆成「离散头 + 子任务 + 连续头」三条</p>
</div>

一句话总结结构差异：**π0 是"看得懂就直接动手"；π0.5 是"看得懂 → 先说一句要干什么 → 再动手"**。多出来的那句中间文本，就是长时程任务能撑住的全部秘密——它把 15 分钟的清理任务切成一段段可校验的子目标，出错时能从子目标层回退，而不是从头再来。

## 5. π0.6*：RL 专精（RECAP）

用**强化学习**在已训 VLA 上做专精打磨（RECAP 思路），把特定任务成功率再往上推。是"通用预训练 + 专精后训练"的典型后半段。

## 6. π0.7：记忆 + 组合泛化（2026-04）

π0.7 是系列目前的顶点，几个关键升级：

- **底座升级**：Gemma3 4B + 860M 动作专家。
- **MEM 记忆机制**：从过往经验中学习、携带上下文记忆，处理长程多步任务更稳。
- **多模态提示**：支持图像 / 语言 / 演示混合提示，会"照着示范做"。
- **涌现的组合泛化（compositional generalization）**：未见过的指令组合也能泛化——这是迈向"真正通用"的信号。
- **效果**：在多项基准上**超越专精策略（specialist policies）**，而不只是追平。

## 7. 演进主线（一句话）

> VLM 底座升级 → 跨本体复用 → RL 专精打磨 → 记忆 + 组合泛化。**通用 VLA 正从"能干活"走向"比专精策略更强"。**

## 8. 承上启下

π 系列是海外标杆。下一篇看国内：蚂蚁灵波 LingBot-VLA 2.0 的"一脑多机"、小米的开源实时 VLA、腾讯 HyVLA 的 FlowPRO——以及一处常见混淆的澄清。
