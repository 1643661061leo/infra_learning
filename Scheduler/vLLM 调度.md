# vLLM 调度概览

## 1. 核心目标

vLLM 调度器需要同时解决四个问题：

1. 本轮选择哪些请求执行；
    
2. 每个请求本轮计算多少 Token；
    
3. 如何分配和复用 KV Cache；
    
4. KV Cache 不足时，如何释放资源并保证服务继续运行。
    

可以把整个系统理解为：

```
请求队列
   │
   ▼
Scheduler：分配本轮 Token 预算
   │
   ▼
KVCacheManager：分配物理 KV Block
   │
   ▼
Model Runner：执行一次 Forward
   │
   ▼
采样、停止判断、流式输出
   │
   └────────► 下一轮调度
```

vLLM V1 的关键变化是采用统一调度器：调度器不再严格区分“Prefill Batch”和“Decode Batch”，而是为每个请求分配本轮需要计算的 Token 数量：

```
{
    request_id_1: num_tokens_1,
    request_id_2: num_tokens_2,
}
```

这个统一模型可以同时表达 Prefill、Decode、Chunked Prefill、Prefix Cache 和 Speculative Decode。[vLLM V1 官方说明](https://docs.vllm.ai/en/stable/usage/v1_guide/)

---

## 2. vLLM V1 的统一调度模型

每个请求主要维护两个进度：

```
num_tokens
    当前请求中已经确定的 Token
    包括 Prompt 和已生成的输出 Token

num_computed_tokens
    已经完成模型计算、拥有有效 KV 的 Token
```

调度器关注两者的差值：

```
本轮待计算 Token
=
num_tokens - num_computed_tokens
```

### Prefill 请求

假设：

```
Prompt 长度 = 4,000
num_computed_tokens = 0
```

那么：

```
待计算 Token = 4,000
```

这是一个典型的 Prefill 请求。

### Decode 请求

假设：

```
Prompt + 已生成 Token = 4,101
num_computed_tokens = 4,100
```

那么：

```
待计算 Token = 1
```

这就是一个普通 Decode 请求。

### Prefix Cache 命中

假设一个 4,000 Token 的 Prompt 命中 3,000 Token：

```
num_tokens = 4,000
num_computed_tokens = 3,000
```

本轮只需要计算：

```
4,000 - 3,000 = 1,000 Token
```

因此，Prefill、Decode 和 Prefix Cache 在调度器看来，本质上只是“待计算 Token 数不同”。[V1 Scheduler 接口](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/interface/)

---

## 3. 连续批处理 [[Continuous Batching]]

传统静态批处理要求：

```
一批请求同时开始
等待所有请求结束
再接收下一批
```

问题在于不同请求的输出长度不同：

```
请求 A：生成 20 Token
请求 B：生成 200 Token
请求 C：生成 1,000 Token
```

A 完成后，它在静态 Batch 中的位置可能一直空闲。

vLLM 使用连续批处理：

```
Step 1：A B C
Step 2：A B C
Step 3：  B C D
Step 4：  B C D E
```

每个调度轮次都可以：

- 移除完成请求；
    
- 加入新请求；
    
- 恢复被抢占的请求；
    
- 改变每个请求本轮处理的 Token 数。
    

因此，vLLM 中的 Batch 是**动态变化**的，不是请求进入时就固定到结束。

---

## 4. PagedAttention 与 KV Cache Block

### 5.1 连续 KV Cache 的问题

如果为每个请求预留一段最大长度的连续 KV Cache：

```
请求最大长度 = 32K
当前只用了 2K
```

==剩余空间可能长期闲置。

==不同请求长度不同，也容易产生显存碎片：

```
[请求 A KV][空洞][请求 B KV][小空洞][请求 C KV]
```

### 5.2 分页管理

vLLM 将 KV Cache 划分成固定大小的 Block：

```
KV Cache Pool：

Block 0
Block 1
Block 2
...
Block N
```

每个 Block 保存固定数量 Token 的 KV。

请求通过 Block Table 建立逻辑 Token 到物理 Block 的映射：

```
请求 A 的逻辑 Block：

Logical 0 ──► Physical Block 7
Logical 1 ──► Physical Block 2
Logical 2 ──► Physical Block 18
```

请求的物理 Block 不需要连续。

这与操作系统虚拟内存分页类似：

```
逻辑连续
物理不连续
按需分配
```

PagedAttention 的主要作用是：

- 不需要为最大长度提前分配连续 KV；
    
- 减少外部显存碎片；
    
- 请求增长时按 Block 分配；
    
- 请求结束后按 Block 回收；
    
- Prefix Cache 可以让多个请求共享物理 Block。
    

这是 vLLM 高并发调度的内存基础。[vLLM PagedAttention 介绍](https://vllm.ai/blog/2023-06-20-vllm)

---

## 5. KV Cache 容量

对于普通 Transformer，单 Token KV Cache 大小可以近似理解为：

```
单 Token KV 大小
≈
层数
× 本卡 KV Head 数
× Head Dimension
× 2（Key 和 Value）
× 数据类型字节数
```

因此：

```
KV Token 容量
≈
分配给 KV Cache 的显存
÷ 单 Token KV 大小
```

vLLM 启动时会加载：

- 模型权重；
    
- CUDA Graph 和运行时 Buffer；
    
- 激活所需空间；
    
- KV Cache Pool。
    

`gpu_memory_utilization` 决定当前 vLLM 实例可以使用的 GPU 显存比例。模型权重大小通常是固定的，提高该值主要可能让剩余显存中更多空间用于 KV Cache，而不是让模型权重变小。

如果显式设置 `kv_cache_memory_bytes`，则会直接指定 KV Cache 显存预算，并覆盖基于 `gpu_memory_utilization` 的自动推导。[CacheConfig](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)

---

## 6. `max_num_batched_tokens`

`max_num_batched_tokens` 控制：

> 一次调度迭代最多处理多少输入 Token。

它是计算预算，不是 KV Cache 总容量。

例如：

```
--max-num-batched-tokens 2048
```

本轮可能组成：

```
100 个 Decode 请求：100 Token
1 个 Prefill 请求：1,948 Token
--------------------------------
总计：             2,048 Token
```

也可能是：

```
2 个 Prefill 请求：
请求 A：1,000 Token
请求 B：1,048 Token
```

但系统中所有运行请求已经持有的 KV Token 总量可以远大于 2,048。

### 值较小

效果通常是：

- Prefill Chunk 较小；
    
- 单次 Forward 时间较短；
    
- Decode 被 Prefill 阻塞的时间较短；
    
- TPOT/ITL 通常更好；
    
- 长 Prompt 需要更多轮才能完成；
    
- TTFT 可能增加；
    
- 调度和小 Batch 开销占比增加。
    

### 值较大

效果通常是：

- 单轮处理更多 Prefill Token；
    
- 长 Prompt 更快完成 Prefill；
    
- TTFT 通常更好；
    
- GPU 利用率和吞吐可能提高；
    
- 单轮 Forward 变长；
    
- Decode 请求等待下一轮的时间可能增加；
    
- TPOT/ITL 可能变差。
    

vLLM 官方调优建议也指出：较小值倾向改善 ITL，较大值倾向改善 TTFT，并可能在大 GPU、小模型场景提高吞吐。[Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)

---

## 7. `max_num_scheduled_tokens`

最新 V1 中还存在：

```
max_num_scheduled_tokens
```

它限制调度器一轮实际下发的 Token 数。

通常：

```
max_num_scheduled_tokens
=
max_num_batched_tokens
```

但在推测解码等场景中，模型执行阶段可能额外追加 Draft Token，因此调度器可使用一个==更小的下发预算，给这些额外 Token 留出空间。

普通非推测解码场景，可以先将它理解为与 `max_num_batched_tokens` 相近的本轮 Token 上限。

---

## 8. `max_num_seqs`

`max_num_seqs` 控制：

> 一次迭代最多允许多少个活跃 Sequence 参与执行。

例如：

```
--max-num-seqs 128
```

即使本轮 Token 预算还有剩余，活跃请求数达到 128 后，也不会继续接纳新的请求。

### 增大

可能带来：

- 更高并发；
    
- 更大的 Decode Batch；
    
- 更高吞吐；
    
- 更多 KV Cache 占用；
    
- 单轮 Decode 时间增加；
    
- TPOT 可能变差；
    
- KV 不足和 Preemption 风险增加。
    

### 减小

可能带来：

- 并发和吞吐下降；
    
- 等待队列变长，TTFT 中的排队时间增加；
    
- KV Cache 压力下降；
    
- Preemption 减少；
    
- 单请求时延更加稳定。
    

==因此它是请求数量预算，而 `max_num_batched_tokens` 是本轮 Token 计算预算。==[SchedulerConfig](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)

---

## 9. [[Chunked Prefill]]

假设：

```
Prompt 长度 = 10,000
本轮剩余 Token 预算 = 2,000
```

如果启用 Chunked Prefill，调度器不会要求整个 Prompt 一次完成，而是：

```
第 1 轮：Prefill 2,000
第 2 轮：Prefill 2,000
第 3 轮：Prefill 2,000
第 4 轮：Prefill 2,000
第 5 轮：Prefill 2,000
```

每轮都可以同时执行其他请求的 Decode。

当前 V1 在可用场景中默认支持 Chunked Prefill，并倾向先安排 Decode，再用剩余 Token 预算处理 Prefill；放不下的 Prefill 会自动切块。[官方调优文档](https://docs.vllm.ai/en/latest/configuration/optimization/)

### 为什么有利于 TPOT

Prefill 通常计算量大。若一次执行完整的长 Prompt：

```
Decode ──► 超长 Prefill ─────────────► Decode
```

运行请求在 Prefill 期间长时间拿不到下一个 Token。

切块后：

```
Decode
Prefill Chunk
Decode
Prefill Chunk
Decode
```

Decode 的停顿更短，TPOT/ITL 更稳定。

### 为什么可能增加 TTFT

新请求只有完成自己的全部 Prompt Prefill 后，才能获得第一个输出 Token。

==切成更多 Chunk 意味着它需要经历更多轮调度，因此长请求的 TTFT 可能增加。

---

## 10. Decode 优先与混合批次

Prefill 和 Decode 的计算特征不同：

|阶段|单请求本轮 Token|典型瓶颈|
|---|---|---|
|Prefill|数百到数千|计算密集|
|Decode|通常为 1|显存带宽、KV 读取|

vLLM 可以把二者放在同一次 Forward 中：

```
Batch：
100 个 Decode Token
+
1,948 个 Prefill Token
```

这样既保护 Decode 请求，又用 Prefill 填充剩余计算能力。

但混合也意味着 Prefill Chunk 会拉长这一轮 Forward。因此 `max_num_batched_tokens` 实际是在平衡：

```
更快完成 Prefill、改善 TTFT
            ↕
缩短单步时长、改善 TPOT
```

---

## 11. Automatic Prefix Caching

vLLM 采用基于 Hash 的 [[Prefix Cache]]，而不是 Radix Tree。

每个可缓存前缀 Block 的 Hash 包含：

```
父 Block Hash
+
当前 Block Token
+
额外隔离信息
```

额外信息可能包括：

- LoRA ID；
    
- 多模态输入 Hash；
    
- Cache Salt；
    
- 其他会影响 KV 结果的配置。
    

形成链式 Hash：

```
Block 0 Hash
    │
    ▼
Block 1 Hash = hash(Block 0 Hash + Block 1 Tokens)
    │
    ▼
Block 2 Hash = hash(Block 1 Hash + Block 2 Tokens)
```

新请求到达后，从第一个 Block 开始依次查询：

```
Block 0 命中
Block 1 命中
Block 2 未命中
    │
    ▼
复用 Block 0、1
从 Block 2 开始 Prefill
```

[vLLM Automatic Prefix Caching 设计](https://docs.vllm.ai/en/stable/design/prefix_caching/)

### Prefix Cache 的生命周期

一个 KV Block 可能处于以下状态：

```
正在被请求使用
    ref_cnt > 0
    不允许淘汰

请求已结束但缓存仍保留
    ref_cnt = 0
    可以被新请求命中
    也可以在内存紧张时被淘汰
```

需要新 Block 时，Block Pool 会从空闲队列取 Block：

- 如果是普通空闲 Block，直接分配；
    
- 如果仍保存旧 Prefix Cache，则先删除旧 Hash 映射；
    
- 再把物理 Block 分配给新请求。
    

可淘汰缓存通常按照 LRU 思路管理。

### 对性能的影响

Prefix Cache 主要减少：

```
重复 Prompt 的 Prefill 计算
```

所以主要改善：

- TTFT；
    
- Prefill 吞吐；
    
- 重复系统提示词、长文档、多轮对话场景。
    

它通常不会直接加速新输出 Token 的 Decode，因此对稳定 Decode 阶段的 TPOT 影响较小。

---

## 12. Prefix Cache 命中粒度

经典实现主要以完整 Hash Block 为缓存边界。

假设：

```
Block Size = 16
相同前缀长度 = 30 Token
```

那么常规情况下可以稳定复用：

```
floor(30 / 16) × 16 = 16 Token
```

尾部未填满的 Block 通常无法像完整 Block 一样直接共享。

**最新 V1 还引入了更细的 `prefix_match_unit` 概念，可在特定混合模型（例如K3）中让匹配粒度小于物理 Block；它改变的是匹配边界，不会改变物理 KV Page 大小，这里不再展开。

---

## 13. KV Cache 分配失败与 Preemption

运行请求每生成一个新 Token，都需要继续扩展 KV Cache。

假设：

```
当前运行 100 个 Decode 请求
下一轮需要大约 100 个新 KV 位置
```

如果可用 KV Block 不足，系统首先会复用或淘汰==没有请求引用的 Prefix Cache Block。

如果仍然无法分配，就会触发 Preemption：

```
running
   │
   │ KV 分配失败
   ▼
选择一个请求作为受害者
   │
   ▼
释放该请求持有的 KV Block
   │
   ▼
状态变为 PREEMPTED
   │
   ▼
重新放回 waiting queue
```

被 Preempt 的请求不会直接向客户端返回结束。

它之后会重新进入调度，并重建当前上下文的 KV Cache：

```
Prompt
+
已经生成并保留的输出 Token
    │
    ▼
重新 Prefill / Recompute
    │
    ▼
继续生成
```

==如果此前完整 Block 仍保留在 Prefix Cache 中，恢复时可能重新命中一部分。

vLLM V1 默认主要采用 `RECOMPUTE`，不再依赖旧 V0 的 GPU/CPU KV Swap 来处理普通抢占。[Preemption 官方说明](https://docs.vllm.ai/en/latest/configuration/optimization/)

---

## 14. Preemption 为什么会影响性能

Preemption 会造成：

1. 请求重新进入等待队列；
    
2. 已完成的上下文计算可能需要重算；
    
3. 该请求 TTFT 或后续 Token 延迟增加；
    
4. GPU 执行重复计算；
    
5. 系统有效吞吐下降；
    
6. 尾延迟出现明显波动。
    

但它的作用是保证系统继续前进：

```
没有 Preemption：
可能所有运行请求都无法获得下一块 KV

有 Preemption：
释放部分请求资源
让剩余请求继续执行
```

因此 Preemption 是显存不足时的恢复机制，不是正常的性能优化机制。

---

## 15. 调度策略

vLLM V1 当前主要提供：

### FCFS

```
First Come First Served
```

按到达时间调度请求。

优点：

- 简单；
    
- 可预测；
    
- 不容易产生显式优先级饥饿。
    

### Priority

按请求优先级调度：

```
较小的 priority 数值
    → 更高优先级
```

优先级相同时，再按照到达时间排序。

发生 KV 不足时，Priority 模式也可以优先 Preempt 低优先级的运行请求，让高优先级请求继续执行。

vLLM 的 Prefix Cache 命中与调度排序基本独立：Prefix Cache 会在请求准备进入时进行 Hash 查询，但默认调度策略不会像 SGLang LPM 一样，主动把“前缀命中最长”的请求排到最前面。

---

## 16. 完整调度示例

配置：

```
max_num_batched_tokens = 2048
max_num_seqs = 128
```

当前状态：

```
运行中的 Decode 请求：100 个
新请求 A：
    Prompt = 6,000 Token
    Prefix Cache 命中 = 2,000 Token
```

### 第 1 轮

Decode 使用：

```
100 × 1 = 100 Token
```

剩余预算：

```
2048 - 100 = 1948
```

请求 A 剩余 Prefill：

```
6000 - 2000 = 4000
```

本轮调度：

```
A Prefill 1948 Token
```

### 第 2 轮

假设仍有 100 个 Decode 请求：

```
Decode = 100
Prefill 剩余预算 = 1948
```

请求 A 再 Prefill：

```
1948 Token
```

此时 A 只剩：

```
4000 - 1948 - 1948 = 104 Token
```

### 第 3 轮

```
Decode = 100
A Prefill = 104
```

A 完成 Prompt 计算并生成首个输出，产生 TTFT。

之后 A 也进入正常 Decode：

```
下一轮 Decode 请求数约为 101
```

如果在任意一轮中 KV Block 分配失败：

1. 尝试回收无引用 Prefix Cache；
    
2. 如果仍不足，Preempt 低优先级或队尾运行请求；
    
3. 释放其 KV；
    
4. 将其放回等待队列；
    
5. 继续执行可运行请求。
    

---

## 17. 参数对性能指标的影响

|                             |               |          |              |            |
| --------------------------- | ------------- | -------- | ------------ | ---------- |
| 参数变化                        | TTFT          | TPOT/ITL | 吞吐           | Preemption |
| `max_num_batched_tokens` 增大 | 通常改善          | 可能变差     | 通常提高到饱和      | 可能增加       |
| `max_num_batched_tokens` 减小 | 长 Prompt 可能变差 | 通常改善     | 过小可能下降       | 通常降低       |
| `max_num_seqs` 增大           | 排队可能改善        | 可能变差     | 可能提高         | 增加         |
| `max_num_seqs` 减小           | 排队可能变差        | 更稳定      | 可能下降         | 降低         |
| KV Cache 容量增大               | 改善            | 更稳定      | 提高           | 降低         |
| Prefix Cache 命中提高           | 明显改善          | 基本不直接改变  | Prefill 吞吐提高 | 可能降低       |
| Watermark 增大                | 排队可能增加        | 更稳定      | 并发可能下降       | 降低         |
| Chunk 变小                    | 长请求可能变差       | 改善       | 过小可能下降       | 视负载而定      |
| Preemption 增多               | 变差            | 抖动增大     | 有效吞吐下降       | 已经发生       |

实际效果还取决于：

- Prompt/Output 长度分布；
    
- 到达率；
    
- 模型大小；
    
- GPU 型号；
    
- Prefix Cache 命中率；
    
- 是否使用推测解码；
    
- TP/PP/DP 配置。
    

---

## 18. vLLM 与 SGLang 调度对比

|   |   |   |
|---|---|---|
|维度|vLLM V1|SGLang|
|KV 管理基础|PagedAttention、Block Table|Token/Page Pool|
|Prefix Cache|Hash Block|Radix Tree|
|基本调度策略|FCFS、Priority|FCFS、LPM、DFS Weight、LOF 等|
|缓存感知排序|默认不按命中长度排序|LPM 可优先最长命中|
|Prefill/Decode|统一 Token 调度|Batch/模式更显式|
|Chunked Prefill|根据 Token Budget 自动切分|根据 Chunk 预算切分|
|显存不足|Preemption + Recompute|Retract，重新排队|
|未来输出 KV 控制|主要按需分配；完整 ISL 检查和 Watermark|`new_token_ratio` 和 `schedule_conservativeness`|
|`max_tokens`|不物理预留完整输出 KV|同样不是简单物理预分配，但参与调度估算|

两者核心目标一致：

```
在有限 KV Cache 下
尽可能提高并发和吞吐
同时控制 TTFT、TPOT 和重算风险
```

但侧重点不同：

```
vLLM：
分页内存管理 + 统一 Token Budget

SGLang：
Radix 前缀复用 + 缓存感知排序 + 未来 KV 估算
```

---
