# DCP、PCP 并行策略原理和流程
## 1. 先记结论

- **PCP 拆“本轮新输入”**：把长 Prompt 的 token 分给多张 GPU 并行计算，核心目标是降低 **TTFT（Time To First Token，首 token 延迟）**。
- **DCP 拆“历史上下文”**：把 Decode 要读取的 KV Cache 沿序列维切到多张 GPU，核心目标是降低单卡 KV Cache 占用、扩大 batch，并改善长上下文 Decode 的可扩展性。
- 两者都沿 **sequence/context 维**做文章，但服务于两个计算特征完全不同的阶段。

一句话记忆：**PCP 分新题，DCP 分历史资料。**

---

## 2. 为什么要分成两套并行策略（[[并行策略概览]]）

一次生成式推理分为两个阶段：

### 2.1 Prefill：一次处理整个 Prompt

设输入长度为 `S`：

- 一次产生 `S` 个位置的 Q/K/V；
- Dense Attention 的主要计算量随 `S²` 增长；
- 通常更偏**计算密集型**；
- 用户最直接的体验指标是 TTFT。

### 2.2 Decode：每步只生成一个或少量 token

每一步只有很少的新 Query，却要读取长度接近 `S` 的历史 K/V：

- 单步 Attention 计算量大致随上下文长度 `S` 线性增长；
- KV Cache 会随 batch、层数和上下文长度增长；
- 通常更偏**显存容量和显存带宽受限**；
- 关键指标是 TPOT/ITL（每 token/相邻 token 延迟）、最大并发和吞吐。

因此，不能只做一个笼统的“Context Parallel”：

| 对比项 | PCP | DCP |
|---|---|---|
| 全称 | Prefill Context Parallelism | Decode Context Parallelism |
| 主要阶段 | Prefill | Decode |
| 切分对象 | Prompt 中待计算的 token / Query | 已存在的 KV Cache |
| 主要目标 | 降低长 Prompt 的 TTFT | 降低单卡 KV 占用、扩大 batch/吞吐 |
| 主要瓶颈 | `O(S²)` Attention 计算 | 长 KV 的容量与读取带宽 |
| 典型通信 | AllGather K/V 或 Ring Attention | 汇集 Query；合并局部 Attention 输出与 LSE |
| 是否天然增加设备数（vLLM 语义） | 是，PCP 是额外并行维 | 否，通常复用 TP ranks |
| 主要代价 | 更多 GPU、跨卡 K/V 通信、因果负载均衡 | 每层每步都有通信，可能抬高 TPOT |

---

## 3. PCP：怎样并行长 Prompt

假设 Prompt 长度为 `S`，PCP 大小为 `P`。总体思路是每个 rank 只负责大约 `S/P` 个 Query 位置；非 Attention 层大多是逐 token 计算，也能直接处理本地 token 切片。真正困难的是：一个 Query 仍需看到符合因果掩码的全局历史 K/V。

```text
长 Prompt:  t0  t1  t2  t3  t4  t5  t6  t7
                         │ 按序列维切分
PCP rank 0: 处理一部分 Query/token
PCP rank 1: 处理另一部分 Query/token
                         │
                 Attention 需要全局上下文
```

### 3.1 策略 A：Partial Q + Full KV

1. 每个 rank 计算自己的 Q/K/V 切片；
2. 在 PCP 组内 AllGather K/V，使每个 rank 得到完整 K/V；
3. 每个 rank 仅用自己的 Q 对完整 K/V 做带因果 Mask 的 Attention；
4. 各 rank 保留自己的输出 token 切片，继续执行后续逐 token 层。

特点：

- 实现相对直观，适合“上下文较长，但完整 K/V 仍放得下”的场景；
- Query 侧计算约分摊到 `1/P`；
- K/V 在各 rank 上复制，因此主要优化计算延迟，不充分节省峰值 K/V 内存；
- 每层 AllGather 的通信可能成为瓶颈。

### 3.2 策略 B：Partial Q + Partial KV（Ring Attention）

1. 每个 rank 只保存本地 Q/K/V；
2. K/V 块沿环逐步发送；
3. 每收到一个 K/V 块，就计算一份局部 Attention；
4. 使用 online softmax 的统计量合并各块结果；
5. 通信与计算尽量重叠。

特点：

- K/V 不需要在每张卡上完整展开，适合极长上下文；
- 内存扩展性更好；
- 环通信、因果 Mask 和调度更复杂；
- 仍可得到与完整 Attention 等价的精确结果，不必是近似 Attention。

### 3.3 为什么不能总是连续等分

因果 Attention 是下三角计算：越靠后的 Query，可见的 Key 越多。若简单地把前半段给 rank 0、后半段给 rank 1，rank 1 的工作量会明显更大。

工程上常用 round-robin、zigzag 或更细粒度分块，让每个 rank 同时拿到较早和较晚的位置，从而平衡有效 Attention 计算量。它们解决的是**计算负载均衡**，不是改变模型语义。

### 3.4 PCP 真正改善什么

理想情况下，每个 rank 的主要 Attention 计算会接近原来的 `1/P`；但实际延迟可以粗略理解为：

```text
T_pcp ≈ T_compute / P + T_communication + T_imbalance + T_sync
```

所以 PCP 并非越大越好。Prompt 太短时，节省的计算可能覆盖不了通信和同步成本。

---

## 4. DCP：怎样并行历史 KV Cache

假设 Decode 时上下文已有 `S` 个 token，DCP 大小为 `D`。DCP 将每个 KV head 的缓存继续沿 token/sequence 维切分：

```text
完整 KV 序列: K0 K1 K2 K3 K4 K5 K6 K7 ...

DCP rank 0:   K0       K4       ...
DCP rank 1:      K1       K5    ...
DCP rank 2:         K2       K6 ...
DCP rank 3:            K3       K7 ...
```

图中是 token 级轮转示意；真实系统也可能按若干 token 或 KV block 交错存放。

### 4.1 一次 Decode step 的数据流

1. 新 token 产生 Query；根据 TP/DCP 的布局，可能先 AllGather Query heads，或在各 DCP rank 上冗余计算一份较小的 Query 投影；
2. 每个 rank 只用本地 KV shard 计算局部 Attention；
3. 每个 rank 得到局部输出，以及局部 softmax 的归一化统计量（常表示为 LSE，log-sum-exp）；
4. 通过 AllGather + ReduceScatter 或 All-to-All 等集合通信，精确合并局部结果；
5. 新 token 的 K/V 按交错规则写入某个 shard，KV Cache 继续均匀增长。

### 4.2 为什么不能直接平均局部 Attention 输出

Softmax 的分母跨越全部 KV shard，局部结果的权重并不相等。

对第 `r` 个 shard，令：

```text
m_r = max(scores_r)
l_r = Σ exp(scores_r - m_r)
z_r = Σ exp(scores_r - m_r) · V_r
```

先得到全局最大值：

```text
m = max_r(m_r)
```

再稳定地合并：

```text
L = Σ exp(m_r - m) · l_r
O = [Σ exp(m_r - m) · z_r] / L
```

因此只要通信局部输出及 LSE/归一化统计量，就能得到与在一张卡上读取完整 KV 相同的 Attention 结果。

### 4.3 为什么使用交错布局

Decode 时 KV Cache 每一步都会增长。如果采用固定的连续区间切分，新增 token 容易持续落到最后一个 rank，造成容量和读取负载倾斜。

交错布局可写成近似规则：

```text
target_rank = floor(token_index / interleave_size) mod D
```

这样新 token 会轮流进入各 rank，任意时刻的 shard 长度差通常很小。

### 4.4 DCP 与少 KV head 模型

普通 TP 先沿 KV head 维切分。但当 `TP size > KV head 数 H` 时，head 维已经无法继续切，一些 rank 会保存重复 KV Cache。MLA 常可等效为极少量 KV head，因此这种重复尤其明显。

在当前 vLLM 的典型约束下，可把 DCP 理解为在这些原本复制 KV 的 TP ranks 之间改为沿序列维切分：

- DeepSeek-R1（MLA 可视作 1 个 KV head），`TP=8` 时可用 `DCP=8` 消除 8 份 KV 重复；
- Qwen3-235B-A22B 有 4 个 KV heads，`TP=8` 时存在约 2 份重复，可用 `DCP=2` 消除。

序列维切为 `D` 份后，相关 KV 部分的单 rank 占用理想上约降为 `1/D`。但 DCP 每层、每个 Decode step 都要通信，所以它常常首先是**容量与吞吐优化**，并不保证单请求 TPOT 一定降低。

---

## 5. 用 Tensor shape 看 PCP 与 DCP

### 5.1 先区分“全局 shape”和“单 rank 本地 shape”

并行不会改变模型在数学上的全局 Tensor shape，改变的是**每个 rank 实际持有和计算的局部 shape**。

记号：

```text
B    = batch_size
S    = Prefill 的 seq_len，或 Decode 时已有的 cache_len
M    = 本次 Decode 的新 token 数，通常 M=1
H    = hidden_size
Nq   = num_query_heads
Nkv  = num_kv_heads
Dh   = head_dim
H    = Nq × Dh
P    = PCP size
D    = DCP size
```

所以 `[B, S, H]` 与 `[B, S, Nq, Dh]` 不是逐维相等，而是一次 reshape：`H = Nq × Dh`。

为方便说明，先假设各维能整除并行度，且只有 PCP 或 DCP、暂不叠加 TP。这里的“DCP-only”是为了看清切分维度的逻辑模型；vLLM 的实际 DCP 通常复用 TP ranks，组合 shape 见 5.5。没有上下文并行时：

```text
输入 X:       [B, S, H]
Q:            [B, S, Nq,  Dh]
K/V:          [B, S, Nkv, Dh]
Attention分数: [B, Nq, S, S]      # 逻辑 shape，GQA 会把多个 Q head 映射到同一 KV head
输出 O:       [B, S, Nq, Dh] -> [B, S, H]
```

### 5.2 开启 PCP：切的是 Prefill 的 `S` 维

PCP 大小为 `P` 时，每个 rank 只处理约 `S/P` 个 Prompt token。连续切、round-robin 或 zigzag 只会改变“哪些 token 属于该 rank”，不会改变下面的局部 shape。

每个 PCP rank 首先得到：

```text
X_r:       [B, S/P, H]
Q_r:       [B, S/P, Nq,  Dh]
K_r/V_r:   [B, S/P, Nkv, Dh]
```

若采用 **Partial Q + Full KV**：

```text
AllGather 前 K_r/V_r: [B, S/P, Nkv, Dh]
AllGather 后 K/V:     [B, S,   Nkv, Dh]   # 每个 PCP rank 都有完整 K/V

本地 Attention 分数:  [B, Nq, S/P, S]
本地输出 O_r:          [B, S/P, Nq, Dh] -> [B, S/P, H]
```

最后把各 rank 的输出按 token 位置拼接/重排，逻辑上仍是 `[B, S, H]`。关键点是：**Query 长度从 `S` 变成了 `S/P`，Key/Value 长度是否仍为 `S`，取决于 PCP 的 Attention 实现。**

若采用 **Partial Q + Partial KV / Ring Attention**，不会一次物化 `[B, Nq, S/P, S]`：

```text
每个 ring step 收到的 K/V block: [B, S/P, Nkv, Dh]
该 step 的局部分数:              [B, Nq, S/P, S/P]
经过 P 个 step 后:               覆盖全局 S 个 K/V 位置
最终本地输出:                    [B, S/P, Nq, Dh]
```

因此，PCP 严格地说是**在 Prefill 阶段沿 `seq_len` 切分待计算 token/Query**。它是否同时让持久化 KV Cache 也缩小到 `1/P`，取决于框架采用 Full-KV 还是 Partial-KV 路径；不能仅从 `PCP=P` 推断 KV Cache 一定除以 `P`。

### 5.3 开启 DCP：切的是 Decode KV Cache 的 `S` 维

Decode 时本轮 Query 只有 `M` 个新 token。DCP 不把这个 `M` 切成 `M/D`，而是把历史 K/V 的 `S` 切成 `S/D`：

```text
全局 Decode Query:       [B, M, Nq,  Dh]
全局逻辑 KV Cache:       [B, S, Nkv, Dh]

每个 DCP rank 的 Query:  [B, M, Nq,  Dh]   # 复制，或通过通信获得所需 Q heads
每个 rank 的 KV shard:   [B, S/D, Nkv, Dh]
本地 Attention 分数:     [B, Nq, M, S/D]
本地 Attention 输出:     [B, M, Nq, Dh]
本地 LSE:                [B, Nq, M]
```

各 rank 用 LSE 合并局部输出后：

```text
全局 Attention 输出: [B, M, Nq, Dh] -> [B, M, H]
```

新 token 产生的 K/V shape 是 `[B, M, Nkv, Dh]`，但不会在所有 DCP rank 都追加一份；它按交错/分块规则写入负责该 token 的 shard。于是各 rank 的 cache_len 始终约为全局长度的 `1/D`。

### 5.4 通信次数分析

下面按**一个 Attention layer 的一次 Prefill 或一次 Decode step**计数，只统计 PCP/DCP 引入的通信，不包含 TP、EP、PP 本身的通信。逻辑通信阶段和底层 NCCL 调用数不一定相同，例如 K/V 可以打包后用一次 collective，也可以分开调用两次。

| 路径 | 每个 Attention layer 的通信 | 原语 | 传输内容 |
|---|---:|---|---|
| PCP：Partial Q + Full KV | 1 个 AllGather 阶段 | `AllGather` | 沿序列维收集 K/V |
| PCP：Partial Q + Partial KV | `P-1` 轮 | `Send/Recv`，组成 Ring | 每轮交换一个 K/V block |
| DCP：AG+RS，Q 未复制 | 通常 3 次 collective | `AllGather(Q)` + `AllGather(LSE)` + `ReduceScatter(O)` | 完整 Q、softmax 统计量、最终输出 |
| DCP：A2A，Q 未复制 | 通常 2 次 collective | `AllGather(Q)` + packed `AllToAll(O,LSE)` | 完整 Q；局部输出和 LSE |
| DCP：A2A，Q 已复制 | 通常 1 次 collective | packed `AllToAll(O,LSE)` | 局部输出和 LSE |

若模型有 `L` 层、生成 `G` 个 token，DCP 的通信调用量大致为：

```text
AG+RS: 约 3 × L × G
A2A:   约 2 × L × G
Q复制+A2A: 约 1 × L × G
```

这也是 DCP 对互联延迟非常敏感的原因：通信 Tensor 可能不大，但每个生成 token、每个 Attention layer 都要重复 collective。


---


## 6. PCP、DCP 与其他并行策略的关系

| 策略 | 主要切分维度 | 首要问题 | 能否帮助单个超长请求 |
|---|---|---|---|
| DP（数据并行） | 请求/batch | 提高多请求吞吐 | 通常不能拆单请求 |
| TP（张量并行） | 权重、hidden/head | 分摊模型计算与权重 | 能，但通信频繁 |
| PP（流水线并行） | Transformer 层 | 让超大模型跨设备放置 | 能，但有流水线气泡 |
| EP（专家并行） | MoE experts | 分摊专家权重与计算 | 取决于路由和负载 |
| PCP | Prefill token/sequence | 长 Prompt 的 TTFT | 能 |
| DCP | Decode KV sequence | KV 容量、长上下文吞吐 | 能 |

容易混淆的几点：

- **DCP 不是 DP**：DP 把不同请求交给不同副本，DCP 把同一个请求的历史 KV 拆开。
- **PCP 不是 PP**：PCP 按 token/上下文切，PP 按模型层切。
- **Chunked Prefill 不是 PCP**：Chunked Prefill 是把长 Prefill 在时间/调度上切成小块；PCP 是在 Prefill 阶段由多个 ranks 沿 `seq_len` 维并行处理不同 token/Query。两者可以组合。
- **P/D Disaggregation 不是 PCP/DCP 本身**：前者决定 Prefill 和 Decode 放在哪些实例，后两者决定阶段内部如何多卡并行。

---

## 7. 两者如何组合

以 vLLM 当前参数语义为例：

```text
总进程规模（忽略 DP/PP/EP）≈ TP × PCP
DCP 通常复用已有 TP/PCP ranks，不单独扩大 world size
```

一个直观组合是 `TP=2, PCP=2`：4 张 GPU 中，每个 PCP 组包含一套 TP=2 的权重切片，两个 PCP 组分摊 Prefill token。进入 Decode 后，再按框架支持的 DCP 通信组切分 KV。

更干净的生产形态常是 P/D 分离：

```text
请求 ──> Prefill 实例（PCP>1）──传输 KV──> Decode 实例（DCP>1）──> tokens
          优化长 Prompt TTFT                优化 KV 容量与持续生成
```

好处是 Prefill 机器专注计算密集负载，Decode 机器专注容量/带宽密集负载，也避免为 Prefill 增加的 PCP ranks 在 Decode 阶段重复做少量 token 计算。

代价是 KV 传输、KV 布局转换、跨实例调度和容错都会变复杂；若 Prompt 不长或输出很短，额外链路未必划算。

---

## 8. 选型方法

### 8.1 优先考虑 PCP

- Prompt 很长，TTFT 超过目标；
- Prefill kernel 的 GPU 计算占主导；
- 单请求延迟比请求吞吐更重要；
- 多卡互联足以支撑 K/V 通信；
- 性能分析确认不是排队或 KV 传输导致的 TTFT。

### 8.2 优先考虑 DCP

- KV Cache OOM，或单卡只能承载很小的 batch；
- 长上下文 Decode 的显存带宽/容量是瓶颈；
- 模型 KV head 很少，`TP > H` 造成 KV 重复；
- 增加并发后吞吐明显受 KV 容量限制；
- 能接受并优化每层、每步的集合通信。

### 8.3 先不要开 Context Parallel

- 上下文较短，通信开销大于节省的计算；
- 当前瓶颈是模型权重放不下，应先考虑 TP/PP/量化；
- 当前瓶颈是请求数少、GPU 吃不满，可先调 batching/DP；
- 跨卡互联较弱且 Decode 对尾延迟非常敏感；
- 所用模型、Attention 后端或硬件后端没有成熟支持。

---

## 9. 高频误区

1. **“两者都切 sequence，所以只是不同名字。”** 不是。PCP 切 Prefill 的待计算 token，DCP 切 Decode 的历史 KV；目标和通信模式不同。
2. **“DCP=4，Decode 一定快 4 倍。”** 不成立。局部 KV 读取下降了，但多了跨卡归并；最稳定的收益往往是 KV 容量和并发能力。
3. **“PCP 一定把 KV Cache 显存降到 `1/P`。”** 不成立。Partial-Q/Full-KV 路径会复制完整 K/V；vLLM 的 PCP 参数语义也明确区分了 Prefill 计算切分和 Decode KV shard 数。
4. **“局部 Attention 输出直接求平均即可。”** 错。必须携带 max/LSE 等 softmax 统计量做数值稳定的加权合并。
5. **“并行度越大越好。”** 错。短上下文、小 batch 或慢互联下，通信会压过收益。

---

## 10. 面试式自测

**Q1：128K Prompt 导致首 token 等很久，先看什么？**  
A：先确认 Prefill 是否计算主导；若是，优先评估 PCP。

**Q2：首 token 尚可，但长对话时 batch 上不去、KV Cache 快 OOM，先看什么？**  
A：优先评估 DCP，同时检查 KV dtype、分页管理和缓存复用策略。

**Q3：为什么 DCP 对 MLA/MQA/GQA 中 KV head 很少的模型特别有吸引力？**  
A：TP 在 head 维很快切到上限，继续增大 TP 会复制 KV；DCP 能把这些副本改成序列 shard。

**Q4：DCP 合并时为什么要传 LSE？**  
A：每个 shard 只知道自己的 softmax 分母和最大值；LSE 让系统能以数值稳定方式还原全局归一化。

**Q5：PCP 与 DCP 能一起用吗？**  
A：概念上可以，二者是不同阶段/维度；但具体组合、通信组和 Attention backend 支持必须查所用框架版本。P/D 分离时常在 Prefill 实例使用 PCP、Decode 实例使用 DCP。

---

## 11. 参考资料

- [vLLM：Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)
- [vLLM：Serve CLI 中的 PCP/DCP 参数](https://docs.vllm.ai/en/latest/cli/serve/)
- [vLLM Ascend：DCP 的 KV 布局与 Attention 数据流](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/context_parallel.html)
- [论文：Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783)

> 版本提示：PCP/DCP 正在快速演进。部署前应以目标 vLLM/硬件插件版本的模型与 Attention backend 支持矩阵为准，不要仅凭 CLI 中存在参数判断功能已完整可用。
