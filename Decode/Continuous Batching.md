# Continuous Batching 在 vLLM 与 SGLang 中的实现

> 口径：按 2026 年 8 月两项目官方文档与 `main` 分支代码整理。重点讨论单个推理实例内的迭代级调度。

## 1. 一句话结论

传统静态批处理在开始后固定成员，必须等整个 batch 结束才能接纳新请求；Continuous Batching 则在**每次模型迭代**后移除已完成请求，并立即用等待请求补位：

```text
静态批处理：    [A B] → [A✓ B] → [空 B] → [空 B✓] → 才运行C
Continuous：    [A B] → [A✓ B] → [C  B] → 动态进出
```

因此它也叫迭代级调度（iteration-level scheduling），主要收益是减少空槽和排队时间，提高 GPU 利用率与吞吐。

---

## 2. 为什么生成任务尤其需要它

不同请求生成长度不同。若一个 batch 中 A 只生成 2 个 token，B 要生成 100 个 token，静态批处理会让 A 的位置长期空闲。

Continuous Batching 要求框架能独立维护每个请求的：

- 当前 Prompt/输出位置和停止状态；
- KV Cache 物理位置；
- Attention 元数据和采样结果；
- waiting、running、finished 等调度状态。

每轮 forward 的 batch 大小和成员都可以变化，但每个输出仍通过 request ID 返回给正确的请求。

---

## 3. 数值例子：A 结束后，C 立即补位

假设最多同时运行 2 个请求：

```text
A：Prompt 4 tokens，生成 2 tokens
B：Prompt 4 tokens，生成 5 tokens
C：Prompt 4 tokens，生成 3 tokens；A结束时到达
```

### 3.1 静态批处理

初始 batch 固定为 `[A, B]`：

| 迭代 | 执行内容 | 结果 |
|---:|---|---|
| 1 | Prefill A、B | 得到 `A1、B1` |
| 2 | Decode A、B | `A2` 完成，得到 `B2` |
| 3 | 只 Decode B | A 的槽位空闲，得到 `B3` |
| 4 | 只 Decode B | 得到 `B4` |
| 5 | 只 Decode B | `B5` 完成 |
| 6 | 才 Prefill C | 得到 `C1` |

C 虽然在迭代 2 后已经到达，却要等待 B 完成。

### 3.2 Continuous Batching

迭代 2 结束后，调度器立即删除 A，并让 C 补位。若本轮 token 预算为 5：

```text
B的一步Decode：1 token
C的Prefill：   4 tokens
合计：         5 tokens
```

一种混批执行流程是：

| 迭代 | batch 中的工作 | 结果 |
|---:|---|---|
| 1 | Prefill A(4)、B(4) | `A1、B1` |
| 2 | Decode A(1)、B(1) | `A2` 完成，得到 `B2`；C 到达 |
| 3 | Decode B(1) + Prefill C(4) | 得到 `B3、C1` |
| 4 | Decode B(1)、C(1) | 得到 `B4、C2` |
| 5 | Decode B(1)、C(1) | `B5、C3` 均完成 |

关键不是 batch 一直很大，而是 **A 离开后，C 不必等待 B**。

---

## 4. vLLM：统一 token-budget 调度

vLLM V1 的 Scheduler 每轮都重新构造调度结果，核心状态是 `waiting`、`running` 以及每个请求已计算的 token 数。

简化流程如下：

```text
1. 建立本轮 token_budget = max_num_batched_tokens
2. 先检查 running 请求还需要计算多少 token
3. 通过 KVCacheManager 为本轮 token 分配 KV blocks/slots
4. 在 max_num_seqs、token_budget 和 KV 容量允许时接纳 waiting 请求
5. 执行统一 forward，并按 request ID 更新采样结果
6. 完成请求立即释放引用；内存不足时可抢占并稍后重算
```

vLLM 的特点是：

- 调度器不必硬分 Prefill 和 Decode 阶段，而是统一计算“本轮还欠多少 token”；
- PagedAttention 让不同长度请求拥有各自的非连续 KV block table，batch 成员变化时不需要搬动整块 KV；
- Chunked Prefill 开启时，长 Prefill 可占用 Decode 后剩余的 token 预算，因此数值例子的 `B Decode + C Prefill` 可以在同一次 forward 中执行；
- Prefix Cache 命中会先增加 `num_computed_tokens`，Continuous Batching 只调度未命中的尾部。

主要调节项是 `max_num_seqs` 和 `max_num_batched_tokens`：前者限制并发请求数，后者限制单轮总计算 token 数。

---

## 5. SGLang：动态切换 Extend/Decode batch

SGLang Scheduler 显式维护：

```text
waiting_queue：尚未接纳的请求
running_batch：正在Decode的请求
last_batch：上一轮已提交、等待处理结果的batch
chunked_req：尚未完成Extend的长Prompt
```

每轮核心过程可简化为：

```text
1. 处理 last_batch 的结果，过滤已完成请求
2. 完成 Prefill/Extend 的请求并入 running_batch
3. PrefillAdder 根据请求数、token预算、KV容量和Prefix Cache命中选取等待请求
4. 有新请求时构造 Extend batch；否则继续构造 Decode batch
5. 从 ReqToTokenPool / TokenToKVPool 分配或回收索引
6. 内存压力过大时可 retract 部分请求，之后重新接纳
```

对于上面的 C：

- 未开启 mixed chunk 时，可以在 A 退出后的某轮单独执行 C 的 Extend；B 仍保留在 `running_batch`，随后与 C 一起 Decode；
- 开启 mixed chunk 时，SGLang 会把 B 的 Decode 合入 C 的 Extend batch，形成与数值例子相同的 `1 + 4` 混批。

因此，**Prefill 与 Decode 是否同一次 forward 并不是 Continuous Batching 的定义**；只要请求能在迭代边界动态加入、退出，就属于连续批处理。

---

## 6. 两者的核心差异

| 对比项 | vLLM V1 | SGLang |
|---|---|---|
| 连续调度单位 | 每次 Scheduler 迭代 | 每次 Scheduler event loop |
| batch 抽象 | 统一按待计算 token 调度 | 显式区分 Extend/Decode batch |
| Prefill + Decode | 统一 token budget 下自然混批 | 通过 mixed chunk 路径合并 |
| KV 管理 | KVCacheManager + PagedAttention block table | Req/Token KV Pool + RadixCache |
| 内存不足 | 抢占请求，释放 KV，之后重算 | retract 请求、驱逐缓存，之后重新接纳 |
| 常用限制 | `max_num_seqs`、`max_num_batched_tokens` | `max_running_requests`、`max_prefill_tokens` 等 |

可以这样记：

```text
vLLM：每轮按统一token预算，给所有“还欠计算”的请求分配工作
SGLang：每轮在Extend与Decode批次间组织请求，并可选择把二者混合
```

---

## 7. 与相近概念的区别

- **Dynamic Batching**：通常先等待几毫秒收集请求，再形成一个固定 batch；Continuous Batching 还会在生成过程中逐轮换成员。
- **[[Chunked Prefill]]**：解决长 Prompt 如何分段进入调度；它增强连续批处理，但不是同一个概念。
- **[[Prefix Cache]]**：复用其他请求已经计算的公共前缀 KV；它改变每个请求需要计算多少，不决定请求何时进出 batch。
- **Overlap/Async Scheduling**：让 CPU 准备下一批时 GPU 执行当前批，减少调度气泡；这是执行流水线优化。

代价也要看到：==每轮调度带来 CPU 开销；新 Prefill 可能干扰 Decode 的 ITL；KV 不足会触发抢占/回退。因此系统通常要在吞吐、TTFT、ITL 和公平性之间调节 token 与并发预算。==

---

## 8. 参考资料

- [vLLM：项目特性与 Continuous Batching](https://github.com/vllm-project/vllm)
- [vLLM：V1 Scheduler 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [vLLM：SchedulerConfig](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/)
- [vLLM：Chunked Prefill 调优](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill)
- [SGLang：Scheduler 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
- [SGLang：PrefillAdder 与调度策略](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)
- [SGLang：Server Arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)
