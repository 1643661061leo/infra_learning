## 1. 一句话结论

**DP Attention 按请求/batch 切分 Attention 和 KV Cache，但 FFN/MoE 仍跨设备使用 TP/EP。**

```text
Attention：不同 GPU 处理不同请求，各自保存这些请求的 KV Cache
FFN/MoE：  汇集或路由各 GPU 的 token，继续共享分布式权重计算
```

它解决的核心问题不是单条序列太长，而是：

- MLA/MQA 等少 KV-head 模型在高 TP 下产生 KV Cache 重复；
- 单卡 KV 容量限制可承载 batch；
- Attention 不是 MoE 模型的主要权重瓶颈，没有必要让所有 TP rank 保存同一批请求的 KV。

最典型收益是：**减少 KV 重复、扩大 batch、提高高并发 Decode 吞吐。**

---

## 2. 为什么普通 TP 在 MLA 上浪费KV Cache

普通 Tensor Parallel（TP）把模型权重、Q heads 等沿张量维切到多张 GPU，但一个请求仍由整个 TP group 协作处理。

对于 GQA，KV heads 可以在一定程度上沿 head 维切分；但 DeepSeek MLA 可以近似理解为只有一个压缩后的 KV/latent head。当 `TP=8` 时，这一个 KV head 无法再切成 8 份，因此相同请求的 KV Cache 往往在 8 个 TP ranks 上重复保存。

```text
普通 TP=8，同一批请求 B：

GPU0: 模型权重 shard 0 + KV(request 0...B-1)
GPU1: 模型权重 shard 1 + KV(request 0...B-1)
...
GPU7: 模型权重 shard 7 + KV(request 0...B-1)

同一批请求的 KV 被保存 8 份
```

若单卡最多放下 128 条请求，那么增加 TP rank 并不会把系统 KV 容量自然扩展到 `8×128`，因为各 rank 保存的是同一批请求。

---

## 3. DP Attention 的核心思路

DP Attention 不再要求所有 GPU 的 Attention 处理同一批请求，而是把请求分给不同 Attention DP ranks：

```text
DP Attention=8：

GPU0 Attention: request 0, 8, 16...  + 只保存这些请求的 KV
GPU1 Attention: request 1, 9, 17...  + 只保存这些请求的 KV
...
GPU7 Attention: request 7, 15, 23... + 只保存这些请求的 KV
```

于是：

- 一个请求的 KV 只属于一个 Attention DP shard；
- 每张 GPU 只保存约 `1/DP` 的请求；
- Attention 可以独立处于 Prefill、Decode 或 idle 模式；
- 进入 FFN/MoE 时，再把不同 DP shards 的 token 汇集或路由给分布式权重。

注意：DPA 没有把**一条请求的 KV 序列**切到多卡。它切的是**请求/batch 维**，所以它不是 DCP。

---

## 4. DP Attention 与普通 DP 的区别

### 4.1 普通整模型 DP

```text
Replica 0: 完整模型 + request batch 0
Replica 1: 完整模型 + request batch 1
...
```

特点：

- 整个模型复制多份；
- 不同副本处理不同请求；
- 单次推理通常不需要跨副本通信；
- 对超大 MoE 模型代价很高，因为几百个 experts 难以在每个副本完整复制。

### 4.2 DP Attention

```text
Attention：DP，按请求拆分
Dense FFN：TP 或局部 DP/TP 混合
Sparse MoE：EP，按 expert 拆分
```

它是**组件级混合并行**：只复制/数据并行相对较轻的 Attention，让巨大的 FFN/MoE 权重继续分布在整个 GPU group 上。

| 对比项 | 普通 DP | DP Attention |
|---|---|---|
| DP 范围 | 整个模型 | 主要是 Attention/KV |
| FFN/MoE 权重 | 每个副本复制 | 可继续 TP/EP 分片 |
| 请求处理 | 每个副本处理不同请求 | 每个 Attention DP shard 处理不同请求 |
| 推理中跨 DP 通信 | 通常没有 | Attention 与 FFN/MoE 边界需要通信 |
| 适合模型 | 可复制的 Dense/中小模型 | MLA + 大型 MoE 模型 |

---

## 5. 并行拓扑：TP、Attention DP 如何组合

记：

```text
T = tp_size
D = dp_size（开启 DP Attention 后表示 Attention DP size）
C = attn_cp_size
A = Attention TP size
```

SGLang 当前逻辑近似为：

```text
A = T / (D × C)
```

暂不使用 Context Parallel，即 `C=1` 时：

```text
Attention TP size = T / D
```

要求通常包括：

```text
D > 1
T % D == 0
```

### 5.1 纯 DP Attention：`TP=8, DP=8`

```text
Attention TP = 8 / 8 = 1
```

8 个 GPU 各自独立完成本地请求的 Attention，不在 Attention 内按 head 做 TP；FFN/MoE 仍可在 8 GPU 上用 TP/EP。

### 5.2 混合 DP Attention：`TP=16, DP=2`

```text
Attention TP = 16 / 2 = 8
```

形成两个 Attention DP groups：

```text
Attention DP group 0：8 个 Attention TP ranks，处理 batch 0
Attention DP group 1：8 个 Attention TP ranks，处理 batch 1
```

它把 KV 重复度从可能的 16 份降到每组内的 8 份，但没有像 `DP=16` 那样完全消除组内重复。混合方式常用于单个 Attention 仍需要一定 TP，或纯 DPA 的权重/计算不合适的场景。

---

## 6. Tensor shape 例子

设：

```text
B  = 总 Decode batch size
S  = 当前 KV cache length
H  = hidden_size
Nq = query heads
Dh = head_dim
Ck = MLA 压缩 KV latent 维度
M  = 本次新 token 数，Decode 通常 M=1
```

### 6.1 普通 `TP=8`

同一个 batch 会出现在所有 TP ranks：

```text
全局输入： [B, M, H]
每 rank Q：约 [B, M, Nq/8, Dh]
每 rank KV Cache（MLA逻辑）：[B, S, Ck]
```

注意每个 rank 的 Q heads 不同，但 `[B,S,Ck]` 是同一批请求的重复缓存。

### 6.2 `TP=8, DP Attention=8`

假设请求负载均衡，每个 Attention DP rank 只处理 `B/8` 条请求：

```text
每 rank 输入：    [B/8, M, H]
每 rank Q：       [B/8, M, Nq, Dh]     # Attention TP=1
每 rank KV Cache：[B/8, S, Ck]
Attention 输出：  [B/8, M, H]
```

相对普通 TP，在固定系统总 batch `B` 下：

```text
单 rank KV Cache：从 [B,S,Ck] 下降为 [B/8,S,Ck]
```

进入 FFN/MoE 前，本地 token activation 可写为：

```text
GPU d 本地：X_d [N_d, H]
全局 token：X   [ΣN_d, H]
```

系统通过 DP gather 或 EP dispatch，让分布式 FFN/MoE 能处理所有 DP ranks 的 token；计算完成后，再把结果送回 token 原来的 Attention DP rank。

### 6.3 `TP=8, DP Attention=2`

```text
Attention DP groups = 2
每组 Attention TP   = 4
每组处理请求数       ≈ B/2
```

每个组内 rank 的典型逻辑 shape：

```text
Q_local： [B/2, M, Nq/4, Dh]
KV Cache：[B/2, S, Ck]          # MLA 下可能在组内4个 ranks重复
```

所以 DP Attention size 从 1 增大到 2，单卡 KV 约降为原来的 `1/2`；继续增大到 8，才达到每 GPU 独立保存不同请求的纯 DPA 形态。

---

## 7. 一层 Transformer 的数据流

以 `TP=8, DP=8, EP=8` 为例：

```text
                 ┌────────────────────────────┐
不同请求 ──────> │ Attention DP                │
                 │ GPU d 只计算本地请求与本地KV │
                 └─────────────┬──────────────┘
                               │ local tokens [N_d,H]
                               ▼
                 DP Gather / EP All-to-All Dispatch
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ MoE Expert Parallel         │
                 │ 各GPU计算自己持有的experts   │
                 └─────────────┬──────────────┘
                               │
                      All-to-All Combine
                    / ReduceScatter(v)
                               │
                               ▼
                 token回到原Attention DP rank
                               │
                               ▼
                       下一层本地Attention
```

### 7.1 Attention 内部

纯 DPA（Attention TP=1）时，不同 Attention DP ranks 之间通常不需要为 Attention Tensor 做 collective：

- 本 rank 有本地请求的完整 Attention 权重；
- 本 rank 有本地请求的完整 KV Cache；
- Q 只读取本地 KV；
- 输出仍属于本 rank 的本地 token。

如果是混合 DPA、`Attention TP>1`，每个 Attention DP group 内仍需要普通 TP 通信。

### 7.2 Attention 与 FFN/MoE 边界

各 Attention DP ranks 的 token 数可能不同，因此实现会根据 backend、padding 模式和并行拓扑选择：

- `AllGather` / `AllGatherv`：汇集不同 DP ranks 的 token；
- 带零填充的 `AllReduce`：构造全局 token buffer；
- `ReduceScatter` / `ReduceScatterv`：把 FFN/MoE 输出归并并送回原 DP shard；
- EP `AllToAll` Dispatch：将 token 发给被 gate 选中的 experts；
- EP `AllToAll` Combine：把 expert 输出送回 token 原位置。

因此 DPA 不是“完全没有通信”，而是把通信从 Attention 内部/重复 KV，转移到 Attention 与分布式 FFN/MoE 的边界，并尽量与 MoE 路由融合。

---

## 8. 为什么它能扩大 batch

设固定总 batch 的一份 MLA KV Cache 大小为 `K(B,S)`。

普通 `TP=T`：

```text
每 GPU KV ≈ K(B,S)
整个 TP group KV 总量 ≈ T × K(B,S)
```

开启 `DP Attention=D`、不考虑 CP：

```text
每 GPU KV ≈ K(B/D,S)
Attention TP size A = T/D
整个 group 的逻辑 KV 重复度从 T 降到 A
```

纯 DPA，即 `D=T, A=1`：

```text
每 GPU KV ≈ K(B/T,S)
整个 GPU group 合起来只保存一份全局请求集合的 KV
```

这释放的显存可以用于：

- 增大 `max_running_requests`；
- 增大 Decode batch；
- 承载更多并发长上下文；
- 提高总输出 tokens/s。

它不能让一条超长请求自动跨多卡放置；单请求 KV 放不下时应考虑 DCP、KV offload、压缩或缩短上下文。

---

## 9. 与 PCP、DCP、TP、EP 的区别

| 策略 | 切分维度 | Attention/KV 所有权 | 主要目标 |
|---|---|---|---|
| TP | head/hidden/权重 | 同一请求由 TP group 协作，KV 可能重复 | 分摊权重和计算 |
| DP Attention | 请求/batch | 每个请求属于一个 Attention DP shard | 去 KV 重复、扩大 batch |
| PCP | 单请求的 Prefill `seq_len` | 多 rank 分摊同一 Prompt token | 降低长 Prompt TTFT |
| DCP | 单请求的 Decode KV `seq_len` | 同一请求 KV 被切到多 rank | 降低单请求 KV 占用 |
| EP | expert | token 按 gate 路由给不同 experts | 分摊 MoE 权重和计算 |

最容易记混的是 DPA 与 DCP：

```text
DPA：request 0 去 GPU0，request 1 去 GPU1
DCP：request 0 的 KV 前/偶数部分在 GPU0，后/奇数部分在 GPU1
```

DPA 优化“很多请求如何铺到多卡”；DCP 优化“一条请求的长 KV 如何跨卡”。

---

## 10. 为什么不同 DP ranks 可以处于不同模式

因为 Attention 阶段的请求彼此独立：

```text
DP rank 0：可能在处理一个长 Prefill
DP rank 1：可能在处理 Decode batch
DP rank 2：可能 idle
```

但它们进入共享 TP/EP 的 FFN/MoE 后需要同步或参与同一轮 token 路由。因此可能出现：

```text
某 rank 的长 Prefill较慢
        ↓
其他 rank 在MoE同步点等待
        ↓
Decode ITL/TPOT长尾增加
```

这也是 DP Attention 常与以下优化组合的原因：

- Prefill/Decode 分离，避免 Prefill rank 和 Decode rank 互相拖累；
- Two-Batch Overlap，让通信与计算重叠；
- DeepEP、EPLB，降低 MoE All-to-All 和 expert 不均衡；
- 合理的 DP 请求路由，平衡各 DP ranks 的 token 和 KV 使用量。

---

## 11. 启动参数怎样读

典型纯 DPA：

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --dp-size 8 \
  --enable-dp-attention
```

含义：

```text
全局 TP/world group = 8
Attention DP size   = 8
Attention TP size   = 1
```

DeepSeek MoE 常见组合：

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --dp-size 8 \
  --ep 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm
```

此时：

```text
Attention：DP8，每GPU处理不同请求
MoE：      EP8，每GPU持有不同experts
```

配置注意：

- 必须显式设置 `--enable-dp-attention`；
- `--dp-size` 应大于 1，否则 DPA 会被关闭；
- `tp_size % dp_size == 0`；
- 不要把这里的 `--dp-size` 简单理解成“复制 8 份完整模型”；开启 DPA 后，它表示 Attention DP 维度；
- 具体模型、量化、Attention/MoE backend、CUDA Graph 和 PD 模式的组合支持需查目标版本。

---

## 12. 什么时候适合开启

### 适合

- DeepSeek、Kimi-K2、MiniMax 等 MLA 模型；
- `TP > KV head数`，KV Cache 在 TP ranks 上大量重复；
- Decode 高并发，显存限制 batch；
- 大型 MoE 权重必须继续 EP/TP 分布；
- 目标是提高总吞吐，而非极致单请求延迟；
- GPU 间有足够快的互联支撑 MoE/边界通信。

### 不一定适合

- 低并发、小 batch、追求最低 TPOT/ITL；
- Attention 权重复制后放不下或本地 Attention kernel 利用率很差；
- 普通 GQA 模型已经能用 TP 有效切分 KV heads；
- DP ranks 负载严重不均，导致 MoE 同步等待；
- 单条请求 KV 太长而 OOM——这更接近 DCP 问题；
- 当前瓶颈是 Prefill TTFT——应先分析 PCP、chunked prefill 或 P/D 分离。

---

## 13. 高频误区

1. **“DP Attention 就是普通 DP。”** 不是。普通 DP 复制整个模型；DPA 主要让 Attention/KV 按请求 DP，FFN/MoE 仍可 TP/EP。
2. **“DPA 会把一条请求的 KV 切成多份。”** 不会，那是 DCP 的语义。
3. **“开启 DPA 后完全没有通信。”** 只有纯 DPA Attention 内部可无跨 DP 通信；进入共享 FFN/MoE 前后仍需 gather、reduce-scatter 或 EP all-to-all。
4. **“DPA 一定降低单请求 TPOT。”** 不一定。它首先优化 KV 容量和高并发吞吐，小 batch 下同步与小矩阵效率可能更差。
5. **“`--dp-size 8` 就是 8 份完整模型。”** 在 `--enable-dp-attention` 语境下不是；它定义 Attention DP size。
6. **“DPA 和 PCP 不能同时存在。”** 两者切分维度不同，概念上可组合；但具体拓扑和框架支持有约束。

---

## 14. 自测题

**Q1：为什么 MLA 特别适合 DPA？**  
A：MLA 的压缩 KV 很难沿 KV-head 维继续 TP 切分，高 TP 会复制 KV；DPA 把不同请求分给不同 ranks，消除或减少这些副本。

**Q2：`TP=8, DP=8` 时 Attention TP 是多少？**  
A：忽略 CP 时为 `8/8=1`，每个 Attention DP rank 独立处理本地请求。

**Q3：为什么进入 MoE 前还要通信？**  
A：Attention 的 token 分散在不同 DP ranks，而 experts 分散在 EP ranks；必须将 token 路由到对应 experts，再把结果送回原 token。

**Q4：DPA 能解决单条 1M context 请求的单卡 KV OOM 吗？**  
A：不能直接解决。该请求仍属于一个 Attention DP shard；需要 KV offload/压缩或减少上下文等方案。

**Q5：为什么 DPA 可能提高吞吐却不改善平均 TTFT？**  
A：它主要释放 KV 容量、扩大并发 batch；Prefill 计算、DP/EP 同步和通信仍存在，短 Prompt 下收益可能不足以覆盖额外开销。

---

## 15. 参考资料

- [SGLang：DP、DPA 与 SMG 指南](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/dp_dpa_smg_guide.mdx)
- [SGLang：DeepSeek-V3 基础使用与 DP Attention](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md)
- [LMSYS：96×H100 上的 DeepSeek PD 分离与大规模 EP](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- [SGLang：DP Attention 实现与拓扑计算](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/dp_attention.py)

> 版本提示：SGLang 的 DP gather/reduce-scatter、LM Head、MoE backend 和混合 CP/DPA 路径仍在快速优化；上线前应以目标版本的配置校验和实际 profiler 为准。
