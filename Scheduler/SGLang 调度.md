# SGLang 调度概览

## 1. 一句话理解

SGLang 调度器不断解决两个问题：

1. 等待队列中的请求，**谁先执行？
    
2. 当前 KV Cache 还能安全地**放进多少请求？
    

其中：

- `schedule_policy` 主要回答第一个问题。
    
- `schedule_conservativeness` 主要影响第二个问题。
    

命令行参数使用连字符：

```
--schedule-policy fcfs
--schedule-conservativeness 1.0
```

Python 配置字段则使用下划线：

```
schedule_policy="fcfs"
schedule_conservativeness=1.0
```

---

## 2. 调度器为什么重要

调度器需要在[[推理的两个阶段]]中以下目标之间平衡：

- 提高 GPU 利用率和吞吐量；
    
- 降低 TTFT；
    
- 保证 Decode 输出稳定；
    
- 提高前缀缓存命中率；
    
- 避免 KV Cache 被过度占满；
    
- 避免请求饥饿和频繁回退。
    

---

## 3. 核心运行结构

可以把 SGLang 调度过程抽象为：

```
客户端请求
    ↓
Tokenize
    ↓
waiting_queue
    ↓
按 schedule_policy 排序
    ↓
RadixCache 最长前缀匹配
    ↓
PrefillAdder 检查请求能否进入批次
    ↓
Prefill / Chunked Prefill
    ↓
合并进 running_batch
    ↓
连续 Decode
    ↓
完成 → 释放或缓存 KV
    │
    └── KV 不足 → 淘汰可复用缓存 → Retract 部分请求
                                      ↓
                                 返回 waiting_queue
```

调度器的几个关键对象：

- `waiting_queue`：尚未开始或被回退的请求。
    
- `running_batch`：已经完成 Prefill、正在 Decode 的请求。
    
- `SchedulePolicy`：决定等待请求的顺序。
    
- `PrefillAdder`：根据 Token 和 KV Cache 预算构造 Prefill 批次。
    
- `ScheduleBatch`：一次实际提交给模型执行的批次。
    
- `RadixCache`：保存并复用共享前缀的 KV Cache。
    

[SGLang Scheduler 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)

---

## 4. 请求完整生命周期

### 第一步：请求入队

请求经过 Tokenizer 后转换成 token ID，并建立内部 `Req` 对象，随后放入 `waiting_queue`。

主要状态可以概括为：

```
Waiting → Prefill → Decode → Finished
   ↑                    │
   └──── Retract ───────┘
```

### 第二步：按策略排序

调度器通过 `schedule_policy` 对 `waiting_queue` 排序。

这一步只是在决定候选请求的==优先顺序==，并不代表排在前面的请求一定能够立即运行，**原因见第四步。

### 第三步：匹配前缀缓存

SGLang 使用 [[Radix Tree]] 查找请求已经存在于 KV Cache 中的最长前缀。

例如：

```
缓存中已有：
[system prompt] + [用户问题 A]

新请求：
[system prompt] + [用户问题 B]
```

两个请求共享 `system prompt`，==新请求只需要计算后面没有命中的部分。

大致有：

```
实际 Prefill 长度
= 完整输入长度 - 已匹配前缀长度
```

因此前缀缓存可以同时减少：

- Prefill 计算量；
    
- TTFT；
    
- 新增 KV Cache 数量。
    

[RadixCache 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)

### 第四步：准入检查

`PrefillAdder` 按排序后的顺序尝试加入请求，但必须**同时满足多种资源限制：

- 请求槽位是否充足；
    
- 是否超过 `max_running_requests`；
    
- 本批 Prefill token 是否超过 `max_prefill_tokens`；
    
- 是否超过 `chunked_prefill_size`；
    
- KV Cache 是否足够；
    
- 是否需要为正在 Decode 的请求预留未来 KV；
    
- 当前 LoRA、并行模式等是否允许合批。
    

所以：

> ==`schedule_policy` 决定检查顺序，`PrefillAdder` 决定请求最终能不能进入批次。

[PrefillAdder 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)

### 第五步：Prefill 和 [[Chunked Prefill]]

如果输入很长，SGLang 可以把 Prefill 拆成多个 chunk：

```
长 Prompt
  ├── chunk 1
  ├── chunk 2
  └── chunk 3
```

这样可以避免==一个超长 Prompt 长时间独占 GPU，并给其他请求的 Decode 留出执行机会。

相关参数：

```
--chunked-prefill-size 8192
--max-prefill-tokens 16384
```

启用 mixed chunk 时，还可以把部分 Decode 请求与 Prefill 请求放在同一次 Extend Forward 中执行。

### 第六步：连续批处理

完成 Prefill 的请求进入 `running_batch`。

SGLang 使用 [[Continuous Batching]]，==细粒度:batch -> req：

- 新请求可以动态加入；
    
- 已完成请求可以动态离开；
    
- 不必等待整个批次的所有请求一起结束。
    

例如：

```
时刻 T1：A、B、C 正在 Decode
时刻 T2：B 完成，移出批次
时刻 T3：D 完成 Prefill，加入运行批次
```

这比固定批处理更适合==输出长度不一致的在线请求。

### 第七步：内存不足与 [[Retract]]

每轮 Decode 都需要新增 KV Cache。如果下一轮空间不足，SGLang通常先尝试**淘汰可回收的 RadixCache 节点。

如果仍然不够，就会执行 **Retract：

1. 从 `running_batch` 选择部分请求；
    
2. 释放这些请求占用的 KV Cache；
    
3. 将请求重新放回 `waiting_queue`；
    
4. 后续重新匹配前缀并恢复执行。
    

典型日志：

```
KV cache pool is full. Retract requests.
#retracted_reqs: ...
#new_token_ratio: ... -> ...
```

Retract 可以避免整个服务立即 OOM，但代价包括：

- 请求部分工作可能需要重新计算；
    
- 延迟增加；
    
- 输出可能出现停顿；
    
- 吞吐下降。
    

[Decode 内存检查和回退实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)

---

## 5. `schedule_policy`：谁先执行

默认值：

```
--schedule-policy fcfs
```

### `fcfs`

First Come First Serve，先到先服务。

优点：

- 行为简单；
    
- 公平性较好；
    
- 延迟比较可预测；
    
- 适合一般在线服务。
    

缺点：

- 不主动优化前缀缓存命中率；
    
- 大量共享前缀时吞吐可能不如 `lpm`。
    

### `lpm`

Longest Prefix Match，优先调度缓存前缀匹配最长的请求。

优点：

- 提高 KV Cache 命中率；
    
- 减少重复 Prefill；
    
- 适合共享 system prompt、多轮对话和批量 Agent 请求。
    

缺点：

- 冷请求可能持续排在后面；
    
- 高负载下可能降低公平性；
    
- 前缀匹配和排序本身存在 CPU 开销。
    

当前实现中，当等待队列超过一定规模时，`lpm` 可能临时退化为 `fcfs`，避免调度开销过大。

### `dfs-weight`

根据 Radix Tree 子树权重进行深度优先调度，使共享相同前缀的请求集中执行。

适合：

- 离线吞吐任务；
    
- 共享前缀较多的批量请求；
    
- 更强调缓存局部性而不是严格公平性的场景。
    

### `lof`

Longest Output First，优先处理 `max_new_tokens` 较大的请求。

它可以改变长短输出请求的合批方式，但可能让短请求等待更久。

### `random`

随机打乱等待队列，主要用于实验、基准测试或避免特定固定顺序。

### `routing-key`

优先选择与当前运行批次中常见 `routing_key` 相同的请求，用于提高具有相同路由属性请求的批次局部性。

### 优先级调度说明

请求优先级是独立机制。当前版本通常通过以下参数启用：

```
--enable-priority-scheduling
--schedule-policy fcfs
```

启用后，可以==先按请求优先级排序，再使用 FCFS 或 LOF 处理相同优先级的请求。==不要简单地把优先级调度理解为所有版本都应设置 `--schedule-policy priority`。

[调度策略源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)

---

## 6. `schedule_conservativeness`：敢放多少请求

默认值：

```
--schedule-conservativeness 1.0
```

这个参数不直接决定请求顺序，而是影响调度器对“未来单个请求 Decode 还需要多少 KV Cache”的估算。

简化后的预留逻辑是：

```
未来 KV 预留
≈ min(剩余 max_new_tokens, 估算上限)
  × new_token_ratio
```

而 `schedule_conservativeness` 会影响 `new_token_ratio` 的初始值：

```
初始 new_token_ratio
≈ 默认比例 × schedule_conservativeness
```

因此：

### 值增大

例如：

```
--schedule-conservativeness 1.3
```

效果：

- 为运行请求预留更多未来 KV Cache；
    
- 新请求更难进入批次；
    
- 并发和吞吐可能下降；
    
- Retract 风险降低；
    
- 运行更稳定。
    

### 值减小

例如：

```
--schedule-conservativeness 0.3
```

效果：

- 为未来生成预留更少 KV；
    
- 可以接纳更多并发请求；
    
- 吞吐可能提高；
    
- 更容易发生 KV Cache 不足和 Retract。
    

==`new_token_ratio` 不是完全固定的：运行稳定时会**逐步衰减**，使调度变得更积极；发生 Retract 时则会根据实际请求重新估算。

[NewTokenRatioTracker 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py)

---

## 7. 为什么 `max_new_tokens` 会影响调度

即使请求最后只生成 20 个 token，如果调用方设置：

```
{
  "max_new_tokens": 8192
}
```

调度器在请求开始前并不知道它一定会很早遇到 EOS，因此**必须预估它可能继续生成**。

注意：`max_new_tokens` 较大时，调度器通常会为请求估算**更多未来 KV Cache 需求**，因此**减少新请求准入**，调度更加保守。降低 `schedule_conservativeness` 会缩小这部分账面预留，从而允许**更多请求并发运行**；如果这些请求最终**没有提前结束，而是持续生成较长输出**，实际 KV Cache 增长可能超过先前估算，最终导致**空间不足和 Retract**。

当前源码会对调度阶段的超大输出估算做裁剪，但这只影响内存估算，不改变请求真正的停止条件。

---

## 8. 两个参数的关系

| 参数                          | 决定的问题    | 主要优化目标         | 主要风险                |
| --------------------------- | -------- | -------------- | ------------------- |
| `schedule_policy`           | 谁先进入候选队列 | 公平性、缓存命中率、TTFT | 请求饥饿、排序开销           |
| `schedule_conservativeness` | 同时接纳多少请求 | 并发、吞吐、内存稳定性    | 过低导致 Retract，过高降低吞吐 |

==可以记成：

```
schedule_policy
    ↓
决定尝试顺序
    ↓
schedule_conservativeness + KV 预算
    ↓
决定最终能否进入批次
```

---

## 9. 与调度相关的其他重要参数

### `mem_fraction_static`

==控制模型权重和 KV Cache 静态内存池占 GPU 显存的比例。

它决定 KV Cache 池的基础容量，而 `schedule_conservativeness` 决定调度器如何使用这部分容量。

### `max_running_requests`

==限制同时处于运行状态的最大请求数，是请求数量维度的硬限制。

### `max_total_tokens`

==限制 KV Cache 内存池能够容纳的 token 总数，通常由系统**自动根据显存计算。

### `max_prefill_tokens`

==限制一次 Prefill 批次处理的 token 数量。

值太大：

- Prefill 吞吐可能提高；
    
- 单次执行时间变长；
    
- Decode 可能出现抖动。
    

值太小：

- TTFT 或总体 Prefill 效率可能下降；
    
- 调度更加细粒度。
    

### `chunked_prefill_size`

把超长 Prompt 拆成多大的 chunk。

==它主要控制 Prefill 对 Decode 的干扰程度。

### `page_size`

==KV Cache 分配和匹配粒度。

- 小 page：前缀匹配更细，缓存复用率高；
    
- 大 page：管理及部分 Kernel 效率可能更高，但只能复用完整 page。
    

### `disable_radix_cache`

==关闭 RadixAttention 前缀缓存。关闭后，`lpm` 等缓存感知策略无法发挥作用，并可能退化为 `fcfs`。

[SGLang Server 参数定义](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py)

---

## 10. 常见场景配置

### 普通在线聊天服务

```
--schedule-policy fcfs
--schedule-conservativeness 1.0
```

先使用默认值，重点观察 TTFT、ITL 和 Retract。

### 大量共享 system prompt

```
--schedule-policy lpm
--schedule-conservativeness 1.0
```

重点观察：

- Cache hit rate 是否提高；
    
- 吞吐是否提高；
    
- 冷请求的尾延迟是否恶化。
    

### 请求设置了很大的 `max_new_tokens`，但经常提前结束

可以逐步测试：

```
--schedule-conservativeness 0.7
```

确认稳定后再尝试：

```
--schedule-conservativeness 0.5
```

不建议没有监控就直接降到非常低。

### 频繁出现 Retract

可以尝试：

```
--schedule-conservativeness 1.2
```

或者：

```
--schedule-conservativeness 1.3
```

如果仍然频繁出现，还应检查：

- `mem_fraction_static` 是否过低；
    
- `max_running_requests` 是否过大；
    
- 请求的 `max_new_tokens` 是否不合理；
    
- KV Cache 数据类型和模型上下文长度；
    
- 是否存在超长输入或超长输出请求。
    

==总而言之，提前结束 → 激进调度更划算，持续长输出 → 保守调度更安全

---

## 11. 调优时重点观察的指标

不要只看总吞吐，至少同时观察：

- TTFT：首 Token 延迟；
    
- ITL/TPOT：相邻输出 token 的延迟；
    
- tokens/s：整体吞吐；
    
- waiting queue 长度；
    
- running batch size；
    
- KV Cache token usage；
    
- RadixCache hit rate；
    
- Retract 次数；
    
- P50、P95、P99 延迟。
    

推荐调优顺序：

1. 使用 `fcfs + 1.0` 建立基线；
    
2. 先确认 KV Cache 容量合理；
    
3. 根据共享前缀情况测试 `lpm`；
    
4. 再小幅调整 `schedule_conservativeness`；
    
5. 每次只改变一个参数；
    
6. 使用真实 Prompt 和输出长度分布压测。
    

---

## 12. 常见误区

### 误区一：`lpm` 永远比 `fcfs` 快

只有存在较多可复用前缀时，`lpm` 的收益才明显。随机 Prompt 场景可能只有排序开销，没有缓存收益。

### 误区二：提高 `schedule_conservativeness` 会增加并发

恰好相反。值越大，预留越多，调度器通常越不愿意接纳新请求。

### 误区三：Retract 等同于普通缓存淘汰

不是。

- 缓存淘汰主要回收当前没有运行请求依赖的可复用 KV；
    
- Retract 会把正在 Decode 的请求移出运行批次，影响请求延迟。
    

### 误区四：`schedule_policy` 决定批次大小

它主要决定等待请求的排序。真正的批次大小还由 KV Cache、Prefill token 预算、请求槽位和其他运行约束共同决定。

### 误区五：实际输出很短，所以 `max_new_tokens` 可以随便设大

调度器必须参考 `max_new_tokens` 估算未来 KV 占用。设置过大可能降低可接纳并发。

---
