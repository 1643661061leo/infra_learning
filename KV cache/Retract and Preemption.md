# vLLM 与 SGLang 中的 Retract and Preemption 实现

> 本文中的 vLLM 指 vLLM V1。两个项目迭代较快，内容依据 2026-09-01 的官方文档和主分支代码整理。

## 1. 什么是 Retract

推理引擎采用连续批处理时，会同时运行多个请求。每个请求在 Decode 阶段不断生成新 token，因此占用的 KV Cache 也会持续增长。

```text
开始 Decode：请求 A、B、C 都能放入 GPU
                    ↓
序列不断变长，KV Cache 持续增加
                    ↓
下一轮 Decode 已没有足够的 KV 空间
                    ↓
暂时撤回一个或多个请求，释放其 KV Cache
                    ↓
剩余请求继续运行，被撤回请求稍后重新调度
```

这个“**把运行中的请求退回等待队列**”的动作，在 SGLang 中称为 **Retract**，在 vLLM 中通常称为 **Preemption（抢占）**。

它的目标不是让请求失败，而是==避免 KV Cache 真正 OOM，并保证至少一部分请求能够继续完成。==

## 2. Retract 与其他机制的区别

| 机制 | 处理对象 | 结果 |
| --- | --- | --- |
| 请求排队 | 尚未开始运行的请求 | 暂时不分配 KV Cache |
| Prefix Cache 淘汰 | 没有被运行请求占用的缓存块 | 删除冷前缀，为新数据腾出空间 |
| Retract / Preemption | 正在运行、占用 KV 的请求 | 释放其 KV，并把请求退回等待队列 |
| KV Cache Offload | KV Cache 数据 | 将 KV 搬到 CPU、SSD 或远端存储 |
| Abort | 整个请求 | 请求结束并向调用方返回错误 |

Prefix Cache 淘汰通常优先处理“无人使用”的缓存；如果清理这些缓存后仍然无法满足运行请求，才需要 Retract。

==Retract 也不等于 Offload：==

- **Retract** 是调度动作，决定哪个请求暂时停止运行；
- **Offload** 是数据处理方式，决定被移出的 KV 是否保存到其他介质。

一次 Retract 可以选择直接丢弃 KV，之后重新计算；也可以先把 KV 备份到 CPU，之后直接恢复。

## 3. vLLM 中的实现：Preemption

### 3.1 触发条件

vLLM V1 的 Scheduler 会优先调度正在运行的请求。请求需要继续 Prefill 或 Decode 时，Scheduler 调用 `KVCacheManager.allocate_slots()` 分配新的 KV Block。

如果返回 `None`，说明当前 KV Cache 空间不足，Scheduler 就会抢占一个运行中的请求，然后重新尝试分配。

```text
RUNNING 请求申请新 KV Block
          │
          ├─ 分配成功 → 继续运行
          │
          └─ 分配失败 → 选择 victim
                         ↓
                      Preempt
                         ↓
                释放 victim 的 KV Block
                         ↓
                 再次尝试分配空间
```

对应实现位于 vLLM V1 的 [`Scheduler.schedule()`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) 中。

### 3.2 如何选择被抢占请求

- 使用普通调度策略时，从当前 `running` 列表**尾部选择请求**。
- 使用 Priority 调度时，选择**调度优先级最低**的运行请求；同优先级下结合到达时间判断。

如果被选中的请求正是当前无法获得空间的请求，并且已经没有其他请求可以抢占，本轮就不再调度它。

### 3.3 抢占时做了什么

`_preempt_request()` 主要执行以下操作：

1. 释放该请求占用的 KV Block。
2. 释放相关的 Encoder Cache。
3. 将状态从 `RUNNING` 改为 `PREEMPTED`。
4. 把 `num_computed_tokens` 重置为 `0`。
5. 清理尚未确认的 Speculative Decode token。
6. 增加抢占计数。
7. 把请求放回 waiting queue，等待重新调度。

已经生成并确认的输出 token 不会丢失。被清除的是它们对应的计算状态，也就是 KV Cache。

### 3.4 请求如何恢复

请求重新进入调度后，vLLM 会使用“输入 Prompt + 已生成 token”恢复完整上下文：

```text
请求被抢占
    ↓
num_computed_tokens = 0
    ↓
重新进入 waiting queue
    ↓
查询 Prefix Cache 中仍然存在的 KV Block
    ↓
命中的部分直接复用，缺失的部分重新计算
    ↓
恢复 Decode
```

因此，“RECOMPUTE”并不一定表示所有 token 都必然重算：==如果释放后的完整 KV Block 尚未被其他请求覆盖，Prefix Cache 仍可能命中；但在显存紧张时，这些块很可能被逐步淘汰，最坏情况下需要重新计算整个上下文。==

### 3.5 RECOMPUTE 与旧版 SWAP

vLLM V1 默认使用 **RECOMPUTE**：释放 GPU KV，需要时重新计算。旧版曾支持 **SWAP**，即把被抢占请求的 KV 交换到 CPU；这条 V0 路径在 V1 中已不再使用，`--swap-space` 也已移除。

所以默认情况下，vLLM Preemption 应理解为：

> 撤回请求 → 释放 GPU KV → 重新排队 → 尽量利用 Prefix Cache → 重算缺失部分。

## 4. SGLang 中的实现：Retract Decode

### 4.1 调度阶段先预估内存

SGLang 不只检查当前已占用多少 KV，还会使用 `new_token_ratio` **估算运行请求后续可能生成多少 token**，以控制新请求的准入数量。

- 比例较高：调度更保守，运行并发较低，不容易 Retract。
- 比例较低：调度更激进，并发较高，但 KV 估计不足时更容易 Retract。

`--schedule-conservativeness` 会影响这个比例的初始值。

### 4.2 触发条件

每轮 Decode 前，`check_decode_mem()` 会计算下一轮所需的新 KV Page，并让 KV allocator 判断容量是否足够。普通 Decode 通常只需为到达 Page 边界的请求分配新 Page；==Speculative Decode 还要计算额外的临时空间。==

如果下一轮放不下，就调用 `retract_decode()`：

```text
检查下一轮 Decode 所需 KV
          │
          ├─ 足够 → 整个 Batch 继续 Decode
          │
          └─ 不足 → retract_decode()
                         ↓
                 按策略排列运行请求
                         ↓
                 逐个撤回并释放 KV
                         ↓
                 直到剩余 Batch 能运行
```

具体逻辑位于 SGLang 的 [`ScheduleBatch.retract_decode()`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)。

### 4.3 如何选择被撤回请求

默认的 `length` 策略使用下面的排序键：

```text
(已生成 token 数, -输入 token 数)
```

撤回时从排序结果末尾开始取，因此通常：

1. 优先撤回**已生成 token 较少**的请求，减少已经投入的 Decode 计算损失；
2. 已生成长度相同时，**较长输入的请求更可能先被撤回**，从而一次释放更多 KV。

如果启用 `priority` Retract 策略，则先考虑请求优先级，再使用长度规则。

SGLang 会==不断撤回请求并重复检查容量，直到剩余请求能够完成下一轮 Decode。==代码会尽量保留至少一个请求；==如果最后一个请求单独运行仍然放不下，则只能 Abort 该请求。==

### 4.4 普通模式下如何撤回和恢复

在普通的非 PD 分离模式下，SGLang 会：

1. 释放请求正在占用的 KV Cache。
2. 不把这段 KV 插入 Radix Tree，因为系统需要立刻获得可用空间。
3. 调用 `reset_for_retract()` 清理前缀匹配、KV 索引等运行状态。
4. 保留已经确认的 `output_ids`。
5. 将请求重新加入 waiting queue。
6. 再次调度时重新匹配 Radix Cache，并重新 Prefill 缺失部分。

所以普通模式的恢复逻辑与 vLLM RECOMPUTE 类似：==已有公共前缀可能再次命中，但被撤回请求独占的 Decode KV 通常需要重新计算。==

### 4.5 PD Decode 模式下的 KV 备份

在 Prefill-Decode 分离模式的 Decode 节点上，SGLang 还提供专门的 **retraction backup** 路径：

```text
Decode 请求被 Retract
        ↓
KV 从 GPU 备份到 Host
        ↓
释放 GPU KV，进入 retracted queue
        ↓
GPU 再次有足够空间
        ↓
重新分配 GPU Page
        ↓
从 Host 恢复 KV，继续 Decode
```

当前支持的备份后端包括：

- `cpu_tensor`：每个请求使用独立 CPU Tensor；
- `host_pool`：使用预留的 HiCache Host Pool。

这条路径可以避免 ==Decode 节点重新执行长上下文 Prefill==。它是 Retract 的恢复机制，不应与面向跨请求复用的普通 KV Cache Offload 完全等同。

对应参数是 `--disaggregation-decode-retraction-backup`，实现可参考 [`resume_retracted_reqs()`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/decode.py)。

### 4.6 Retract 后调整调度策略

完成 Retract 后，SGLang 会根据剩余请求已经生成的 token 数和 `max_new_tokens`，重新估计 `new_token_ratio`。新的比例通常会让后续准入更加保守，降低立即再次 Retract 的概率。

没有发生 Retract 时，该比例会逐步衰减，使调度慢慢恢复激进，以提高并发和吞吐。

这形成了一个简单的反馈控制过程：

```text
运行稳定 → 逐渐提高并发
    ↑              ↓
调度变保守 ← 发生 Retract
```

## 5. vLLM 与 SGLang 的对比

| 对比项 | vLLM V1 | SGLang |
| --- | --- | --- |
| 常用名称 | Preemption | Retract |
| 主要触发点 | `allocate_slots()` 分配失败 | `check_decode_mem()` 判断下一轮容量不足 |
| 处理粒度 | 逐个抢占运行请求并重试分配 | 逐个撤回，直到剩余 Batch 可以运行 |
| 默认恢复方式 | Prefix Cache 命中 + 重算缺失 KV | Radix Cache 命中 + 重算缺失 KV |
| 默认 GPU KV 处理 | 释放 | 释放，并且不插入 Radix Tree |
| CPU 保存路径 | V1 默认不使用旧式 SWAP | PD Decode 支持 retraction backup |
| 撤回后调度反馈 | 请求重新进入 waiting queue | 重新估计 `new_token_ratio`，调度变保守 |

两者的核心思想相同：

> 当活跃请求的 KV 无法继续增长时，牺牲部分请求的即时进度，释放空间让其他请求先完成。

主要区别在于，SGLang 把 Retract 与输出长度估计、Batch 内存检查和 PD Decode 的 Host 备份结合得更紧；vLLM V1 的基础路径则更直接地采用 Preemption + RECOMPUTE。

## 6. Retract 的性能影响

Retract 能避免服务直接 OOM，但代价比较明显：

- 被撤回请求需要等待，端到端延迟和相邻 token 延迟上升；
- 重算 Prefill 会消耗额外 GPU 算力；
- 重算任务可能与正常 Decode 混合，影响其他请求的生成速度；
- 频繁 Retract 可能形成“释放—恢复—再次释放”的抖动；
- PD Decode 如果备份到 CPU，则会增加 PCIe 传输和 Host 内存开销。

因此，Retract 应当作为内存压力下的保护机制，而不是正常运行时频繁发生的常规路径。

## 7. 如何减少频繁 Retract

### vLLM

- 适当提高 `gpu_memory_utilization`，扩大 KV Cache 空间；
- 降低 `max_num_seqs` 或 `max_num_batched_tokens`，减少同时运行的请求；
- 在硬件允许时调整 TP 或 PP，为 KV Cache 留出更多显存；
- 通过日志和指标观察 Preemption 次数。

### SGLang

- 如果频繁出现 `KV cache pool is full. Retract requests.`，适当提高 `--schedule-conservativeness`；
- 降低 `--max-running-requests`，限制运行并发；
- 在为激活值和 CUDA Graph 留足空间的前提下，调整 `--mem-fraction-static`，扩大 KV Pool；
- 结合 Retract 次数、KV token usage 和延迟指标判断是否存在调度抖动。

## 8. 一句话总结

```text
vLLM Preemption：分配 KV Block 失败后，撤回请求并默认通过重算恢复。

SGLang Retract：提前检查下一轮 Decode 容量，撤回部分请求，
                 同时调整后续调度；PD Decode 还可以从 Host KV 恢复。
```

## 参考资料

- [vLLM：Optimization and Tuning—Preemption](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM V1 Scheduler 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [vLLM：Prefix Caching 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [SGLang：Hyperparameter Tuning](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/hyperparameter_tuning.md)
- [SGLang ScheduleBatch 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- [SGLang PD Decode 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/decode.py)

