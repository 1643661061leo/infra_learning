# Chunked Prefill 在 vLLM 与 SGLang 中的实现

> 口径：按 2026 年 8 月两项目官方文档与 `main` 分支代码整理。具体默认值和兼容范围可能随版本变化。

## 1. 一句话结论

Chunked Prefill 把长 Prompt 沿 token 维切成多段执行，每段产生的 KV Cache 留给下一段继续使用：

```text
Prompt: [0 ..................... L)
        ├── chunk 1 ──┤── chunk 2 ──┤── chunk 3 ──┤
        写入KV          读取前缀KV并续写   完成后开始Decode
```

它主要解决两类问题：

- 限制一次 Prefill 的 token 数和激活峰值，降低长 Prompt OOM 风险；
- 缩短一次长 Prefill 占用 GPU 的时间，使调度器能穿插 Decode，改善 ITL 长尾。

==它不会减少最终 KV Cache 容量；切得过小还会增加 kernel launch、重复读取历史 KV 等开销，拉长 TTFT。==

---

## 2. 每个 chunk 到底算什么

假设前缀 `[0, s)` 已完成，本轮处理 `[s, e)`：

```text
Query：仅本轮新 token [s, e)
Key/Value：历史 KV [0, s) + 本轮 [s, e)
输出：只写入新 token 对应的 KV
```

所以它不是把各段独立计算后拼接，而是==利用 KV Cache 做增量 Prefill==。只有最后一个 Prompt chunk 完成后，才采样首个输出 token。

---

## 3. 数值例子：10 个 Prompt token 怎么分段

假设新请求的 Prompt 为：

```text
[A B C D E F G H I J]，共10个token
```

同时已有 3 条请求正在 Decode，每轮各需要计算 1 个 token。为便于说明，假设页大小为 1。

### 3.1 vLLM：`max_num_batched_tokens=6`

每轮总预算是 6。3 条 Decode 先占 3 个 token，留给新 Prompt 的预算只有 3：

| 轮次 | Decode token | 本轮 Prefill | 本轮 Query | Attention 可见的 K/V | 完成后 Prompt KV 长度 |
|---|---:|---|---|---|---:|
| 1 | 3 | `A~C` | `A~C` | `A~C` | 3 |
| 2 | 3 | `D~F` | `D~F` | `A~F` | 6 |
| 3 | 3 | `G~I` | `G~I` | `A~I` | 9 |
| 4 | 3 | `J` | `J` | `A~J` | 10 |

第四轮处理完 `J` 后，模型根据 `J` 的输出 logits 采样首个生成 token `Y1`；下一轮才进入普通 Decode，用 `Y1` 作为 Query 生成 `Y2`。

这个例子中并没有显式配置“chunk=3”。它是动态算出来的：

```text
本轮Prefill大小 = 6个总预算 - 3个Decode token = 3
```

如果某轮只剩 1 条 Decode，请求的 Prefill chunk 就可能扩大到 5。

### 3.2 SGLang：`chunked_prefill_size=6`

若不启用 mixed chunk，Prefill 与 Decode 不进入同一次 forward：

```text
第1个Prefill batch：A~F，KV长度变为6
第2个Prefill batch：G~J，读取A~F的KV，KV长度变为10，并采样Y1
```

3 条运行中的 Decode 请求需要由其他 Decode forward 处理。

若启用 mixed chunk，3 条 Decode 会占用本批 3 个 token，`PrefillAdder` 的剩余 chunk 预算变为：

```text
rem_chunk_tokens = 6 - 3 = 3
```

于是 Prefill 也变成 `A~C → D~F → G~I → J`。每轮结束后，SGLang 把已有 KV 索引保存为 `prefix_indices`，并用 `self.chunked_req` 在下一轮继续该请求。

这个例子体现了本质差异：vLLM 从统一的迭代预算自然得到 chunk；SGLang 先建立明确的 Prefill chunk 预算，再选择是否与 Decode 混批。

---

## 4. vLLM：统一 token-budget 调度

vLLM V1 没有在调度器里硬分“Prefill 阶段”和“Decode 阶段”，而是统一记录：

```text
还需要计算的 token 数
= request.num_tokens_with_spec - request.num_computed_tokens
```

每轮核心过程可简化为：

```text
1. token_budget = max_num_batched_tokens
2. 优先安排正在运行的请求，保护 Decode
3. 再从 waiting queue 接纳新 Prefill
4. num_new_tokens = min(尚未计算的tokens, 剩余budget)
5. 只为本轮 token 分配新的 KV slots
6. Prefill chunk 与 Decode token 可组成同一次 forward
7. 更新 num_computed_tokens；Prompt 未完成则下轮续算
```

关键点：

- `max_num_batched_tokens` 是**一次调度迭代的全局 token 预算**，也是主要调优旋钮；
- 一个 Prefill 放不进剩余预算时，调度器直接把 `num_new_tokens` 截断，因此 chunk 大小可随当轮剩余预算变化；
- `long_prefill_token_threshold` 可进一步限制单个长 Prefill 每轮取得的 token 数；
- Prefix Cache 命中的 token 会先计入 `num_computed_tokens`，只计算未命中的尾部；
- V1 在条件允许时默认开启 Chunked Prefill。

直观上，vLLM 把 Chunked Prefill 实现成了统一连续批处理调度的自然结果：**谁还欠计算 token，就在本轮预算内补多少。**

---

## 5. SGLang：显式维护 Extend chunk

SGLang 把 Prefill 称为 `extend`，由 `Scheduler.get_new_batch_prefill()` 和 `PrefillAdder` 单独构造 Prefill batch。

核心流程如下：

```text
1. RadixCache 匹配可复用前缀，得到 prefix_indices
2. 待计算长度 = 完整输入长度 - prefix_indices长度
3. PrefillAdder 同时检查：
   - max_prefill_tokens：本批 Prefill 总量约束
   - chunked_prefill_size：本批 chunk 剩余预算
   - KV pool、请求数及页对齐等约束
4. 若放不下，截断 req.extend_range，并记录为 new_chunked_req
5. forward 后把未完成请求的 KV 索引暂存到 Cache
6. 下一轮通过 self.chunked_req 优先续算，直到最后一个 chunk 完成
7. 完成后请求才并入 running decode batch
```

实现上的两个特色：

- **与 RadixCache 深度结合**：==每个 chunk 的结束位置成为下轮的前缀锚点==；关闭 RadixCache 时，也会用轻量 `ChunkCache` 保存 `prefix_indices`，避免丢失已写 KV 的位置。
- **Prefill/Decode 默认是两类 batch 路径**：开启 `enable_mixed_chunk` 后，才会把 running Decode 请求合入当前 chunked-prefill batch；否则 Prefill chunk 单独执行。

注意，当前实现中的 `chunked_prefill_size` 实际由 `rem_chunk_tokens` 在整个 Prefill batch 内递减，因此更接近**批级 token 上限**，不应简单理解成“每个请求固定切成同样大小”。Pipeline Parallel 场景还可用 dynamic chunking 根据历史长度调整下一段大小。

---

## 6. 两者的核心差异

| 对比项              | vLLM V1                             | SGLang                                                         |
| ---------------- | ----------------------------------- | -------------------------------------------------------------- |
| 调度抽象             | 统一追踪“未计算 token”                     | 显式区分 Extend/Decode batch                                       |
| chunk 决策         | 主要取决于当轮剩余 `max_num_batched_tokens`  | `PrefillAdder` 用 `chunked_prefill_size` 等多重预算截断 `extend_range` |
| 跨轮状态             | `num_computed_tokens` + KV block 状态 | `self.chunked_req` + `prefix_indices` + Radix/ChunkCache       |
| 与 Decode 混批      | 核心调度策略，Decode 优先                    | 需开启 mixed chunk 路径                                             |
| [[Prefix Cache]] | 先计入缓存命中，再计算尾部                       | RadixCache 直接参与 admission 和 chunk 续接                           |
| 调参重点             | `max_num_batched_tokens`            | `chunked_prefill_size`，兼顾 `max_prefill_tokens`                 |

可以这样记：

```text
vLLM：预算不够就少算一点，状态统一留到下轮
SGLang：把 Extend 明确截成一段，并把“未完成请求”显式缓存、续接
```

---

## 7. 调优与常见误区

- chunk/预算较小：单次阻塞和激活峰值更小，ITL 通常更好，但 TTFT、Prefill 吞吐可能下降；
- chunk/预算较大：Prefill kernel 效率和 TTFT 通常更好，但 Decode 更容易被长 Prefill 干扰；
- Chunked Prefill 只限制一次 forward 的工作量，不能解决最终 KV Cache 放不下的问题；
- 它与 Prefix Cache 不冲突：Prefix Cache 复用请求间的公共前缀，Chunked Prefill 拆分仍需计算的尾部；
- 它也不等于 Context Parallel：前者是时间上的多轮调度，后者是同一 Prefill 在多卡间并行切序列。

---

## 8. 参考资料

- [vLLM：Chunked Prefill 与性能调优](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill)
- [vLLM：V1 Scheduler 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [vLLM：SchedulerConfig 源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)
- [SGLang：Scheduler 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
- [SGLang：PrefillAdder 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)
- [SGLang：ChunkCache 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/chunk_cache.py)
- [SGLang：Pipeline Parallel Dynamic Chunking](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pipeline_parallelism.md)
