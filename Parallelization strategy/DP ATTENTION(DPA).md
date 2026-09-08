# DP Attention（DPA）原理与流程

> 口径：本文讨论推理阶段的 DP Attention，并区分 SGLang 与 vLLM 的参数语义。

## 1. 一句话结论

**DPA 按请求/batch 切分 Attention 和 KV Cache；FFN/MoE 仍可跨 GPU 使用 TP/EP。**

```text
GPU0 Attention：请求 A、C…… + 这些请求的 KV
GPU1 Attention：请求 B、D…… + 这些请求的 KV

MoE：所有 GPU 上的 token 再按 gate 路由到对应 expert
```

DPA 主要解决：

- MLA/MQA 等少 KV-head 模型在==高 TP 下的 KV Cache 重复；==
- 多请求并发时 ==KV 容量不足；==
- 大型 MoE ==不能像普通 DP 那样把全部 experts 复制多份。==

DPA **不切一条请求的序列**，所以通常不能解决**单条超长请求的 KV 在单卡放不下**。这属于 DCP、KV offload 或 KV 压缩的范畴。

---

## 2. 为什么 TP 可能浪费 KV Cache

普通 TP 让同一批请求由整个 TP group 协作处理。

以 MLA 为例，可近似理解为只有一个压缩 KV latent。 `TP=8` 时，它无法像 8 个 KV heads 那样自然切到 8 张卡，因此同一请求的 KV 可能在各 TP rank 上重复：

```text
GPU0：权重 shard 0 + KV(A、B……)
GPU1：权重 shard 1 + KV(A、B……)
...
GPU7：权重 shard 7 + KV(A、B……)
```

假设每张卡能保存 40 GB KV：

```text
物理 KV 显存：8 × 40 GB = 320 GB
有效 KV 容量：约 40 GB
```

因为 8 张卡保存的是同一批请求，而不是 8 份不同请求。

GQA 模型不一定完全重复。若 `Nkv=4, TP=8`，每个 KV head 通常会出现约两份副本，重复度取决于 KV-head 数、TP degree 和实现。

---

## 3. DPA 如何扩大有效 KV 容量

开启纯 `DPA=8` 后，不同 Attention rank 处理不同请求：

```text
GPU0：KV(A)
GPU1：KV(B)
...
GPU7：KV(H)
```

此时 8 张卡保存的是不同内容：

```text
物理 KV 显存：8 × 40 GB = 320 GB
有效 KV 容量：约 320 GB
```

所以 DPA 不是让单个请求的 KV 变小，而是：

```text
减少同一请求在多卡上的KV副本
                    ↓
多张卡的KV空间可以保存不同请求
                    ↓
更大并发batch与更高吞吐
```

### 不能解决的情况

如果一条请求需要 60 GB KV，而每张卡只能提供 40 GB：

```text
DPA：整条请求仍属于一个 Attention shard
60 GB > 40 GB
→ 仍然放不下
```

==这时需要 DCP 沿 `seq_len` 切 KV，或者使用 KV offload/压缩。==

---

## 4. Tensor shape 例子

设：

```text
B  = 系统总 batch/request 数
M  = 本轮 query token 数，Decode 通常为 1
S  = 已缓存的序列长度
H  = hidden_size
Nq = query head 数
Dh = head_dim
Ck = MLA KV latent 维度
```

### 4.1 普通 MLA `TP=8`

```text
全局输入：          [B, M, H]
每 rank Q：         [B, M, Nq/8, Dh]
每 rank KV Cache：  [B, S, Ck]
```

Q heads 被切分，但每个 rank 的 `[B,S,Ck]` 可能是同一批请求的重复 KV。

### 4.2 纯 `DPA=8, Attention TP=1`

负载均衡时，每个 rank 处理约 `B/8` 个请求：

```text
每 rank 输入：      [B/8, M, H]
每 rank Q：         [B/8, M, Nq, Dh]
每 rank KV Cache：  [B/8, S, Ck]
Attention 输出：    [B/8, M, H]
```

固定系统总 batch 为 `B` 时，==单 rank KV 从 `[B,S,Ck]` 降到约 `[B/8,S,Ck]`。==

### 4.3 混合 `DPA=2, Attention TP=4`

总共仍是 8 张 GPU：

```text
2个 Attention DP groups
每组4个 Attention TP ranks
每组处理约 B/2 个请求
```

组内典型 shape：

```text
Q_local：  [B/2, M, Nq/4, Dh]
KV Cache： [B/2, S, Ck]   # MLA下可能在组内4个TP ranks重复
```

因此，DPA degree 越大，Attention TP group 越小，KV 重复通常越少。

---

## 5. 一层 Transformer 的数据流

以纯 `DPA=8 + EP=8` 为例：

```text
不同请求
   ↓
Attention DPA
每个rank只计算本地请求并访问本地KV
   ↓
本地token [N_d, H]
   ↓
EP All-to-All Dispatch
把token发给gate选中的experts
   ↓
各GPU计算本地experts
   ↓
EP All-to-All Combine
   ↓
token回到原来的Attention rank
   ↓
下一层Attention
```

==纯 DPA 的 Attention 内部通常不需要跨 DPA ranks 的 collective；主要通信发生在共享 FFN/MoE 的边界。==

混合 DPA 中，==如果 `Attention TP>1`，每个 Attention DP group 内仍然需要 TP 通信。==

MoE 各 DP ranks 的 forward 通常还需要对齐。某些 rank 没有请求时，也可能参与 dummy forward，避免其他 rank 卡在 EP collective。

---

## 6. DPA、普通 DP 与 DCP

| 策略 | 切什么 | 权重布局 | KV 所有权 | 主要作用 |
|---|---|---|---|---|
| 普通 DP | 请求 | 整个模型复制 | 每个副本独立 | 扩展吞吐 |
| DPA | Attention 的请求/batch | Attention DP；FFN/MoE 可 TP/EP | 一条请求属于一个 Attention shard | 减少 KV 重复 |
| DCP | 一条请求的 KV 序列 | 模型通常仍共享并行组 | 同一请求 KV 跨卡切分 | 支撑单请求长上下文 |

最简记忆：

```text
DPA：request A 去 GPU0，request B 去 GPU1
DCP：request A 的偶数/奇数 KV token 分到 GPU0/GPU1
```

---

## 7. SGLang 中怎样启用

SGLang 需要同时设置：

```bash
--dp-size 8 \
--enable-dp-attention
```

若不考虑 Context Parallel：

```text
Attention TP size ≈ TP size / DPA size
```

例如：

```text
TP=8，DP=8  → Attention TP=1，纯DPA
TP=16，DP=8 → Attention TP=2，混合DPA+TP
```

通常要求：

```text
DP > 1
TP % DP == 0
```

`dp-size=1` 时，即使传入 `--enable-dp-attention`，DPA 也会被自动关闭。`dp-size>1` 但没有 DPA 开关，表示普通 DP，而不是 DPA。

典型 MoE 配置：

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp-size 8 \
  --dp-size 8 \
  --ep-size 8 \
  --enable-dp-attention
```

```text
Attention：DPA=8
MoE：      EP=8
```

---

## 8. vLLM 中怎样理解

vLLM 没有 SGLang 式的独立 `--enable-dp-attention` 开关。对于 MoE 部署，`--data-parallel-size>1` 会让不同 DP engines 处理不同请求，并各自拥有独立 KV Cache；专家层则跨 DP/TP ranks 执行。

### 8.1 MoE：DPA + EP

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel
```

vLLM 自动得到：

```text
Attention DPA size = DP
Attention TP size  = TP
Expert EP size     = DP × TP
```

因此：

```text
TP=1, DP=8 → 纯DPA=8 + EP=8
TP=2, DP=4 → DPA=4 + Attention TP=2 + EP=8
```

第二种不是纯 DPA。每个 DP group 内仍有两个 TP ranks，MLA KV 仍可能在组内重复。

如果不传 `--enable-expert-parallel`，MoE experts 默认按 `DP×TP` group 使用 tensor parallel，而不是 EP；Attention 仍按 DP engines 分流请求。

### 8.2 `DP=1`

```text
Attention DPA degree = 1
```

等价于没有请求维并行，因此不产生 DPA 的 KV 去重和容量扩展效果。

### 8.3 Dense 模型

Dense 模型使用：

```bash
--data-parallel-size 4 \
--tensor-parallel-size 2
```

得到的是四个完整 Dense 模型副本，每个副本内部使用 TP=2：

```text
DP replica 0：Attention TP2 + Dense FFN TP2
DP replica 1：Attention TP2 + Dense FFN TP2
DP replica 2：Attention TP2 + Dense FFN TP2
DP replica 3：Attention TP2 + Dense FFN TP2
```

它能通过普通 DP 增加总并发和集群 KV 容量，但整个 Dense 模型都复制了四份。当前标准部署路径不是“只让 Attention 做 DP、Dense FFN 仍跨所有 DP ranks 分片”的专用 DPA，因此没有 MoE DPA+EP 的权重优势。

---

## 9. 对性能和容量的影响

| 指标 | 主要方向 | 原因 |
|---|---:|---|
| 单 rank KV 占用 | **↓** | 每个 rank 只持有部分请求 |
| 集群有效 KV 容量 | **↑** | 减少同一请求的 KV 副本 |
| 最大并发/batch | **↑** | 释放的 KV 空间承载更多请求 |
| 总吞吐 | **↑** | Attention 请求并行、batch 增大 |
| 单请求 TTFT | 可能↑或↓ | 少用 Attention TP，但可能增加边界/EP 通信 |
| TPOT/ITL | 可能↑或↓ | 高并发受益；小 batch 可能受同步和小矩阵影响 |
| 单条请求最大上下文 | 通常不变 | 一条请求仍属于一个 Attention shard |

DPA 的第一目标是 **KV 容量和高并发吞吐**，不是保证单请求延迟下降。

---

## 10. 适用判断

优先考虑 DPA：

- DeepSeek、Kimi、MiniMax 等 MLA/MoE 模型；
- `TP` 大于可切分 KV/latent 数，KV 重复明显；
- Decode 并发高，KV pool 很快满；
- Experts 很大，需要继续使用 EP；
- 目标是提高吞吐和并发容量。

不应首先使用 DPA：

- 单条超长请求 KV OOM：考虑 DCP/offload/压缩；
- 主要问题是长 Prompt TTFT：考虑 PCP、chunked prefill；
- Dense 模型只是需要更多副本：直接使用普通 DP；
- 低并发、追求最低单请求延迟：先实测 TP 与 DPA 的 kernel/通信开销。

---

## 11. 参考资料

- [SGLang：DP、DPA 与 SMG 指南](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/dp_dpa_smg_guide.mdx)
- [SGLang：DP Attention 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/dp_attention.py)
- [vLLM：Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
- [vLLM：Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)

> 参数和支持拓扑变化较快，部署时应以实际使用版本的官方文档与启动日志为准。
