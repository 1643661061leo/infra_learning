## 1. 伪代码总结

```
while server_is_running:
    # 1. 新请求进入等待队列
    for request in receive_requests():
        req = Request(request)
        scheduler.add_request(req)  # WAITING

    # 2. 开始一轮调度
    token_budget = max_num_scheduled_tokens
    input_budget = max_num_batched_tokens

    # 3. 先调度 RUNNING 请求
    for req in running:
        num_new_tokens = (
            req.num_tokens_with_spec
            - req.num_computed_tokens
        )

        num_new_tokens = min(
            num_new_tokens,
            token_budget,
            input_budget,
            long_prefill_token_threshold,
        )

        blocks = kv_cache_manager.allocate_slots(
            req,
            num_new_tokens,
        )

        # KV 不足：抢占低优先级请求
        while blocks is None:
            victim = select_preemption_victim(running)
            scheduler._preempt_request(victim)

            # 释放 KV 引用
            # num_computed_tokens = 0
            # status = PREEMPTED
            # 重新放回 waiting

            if victim is req:
                break

            blocks = kv_cache_manager.allocate_slots(
                req,
                num_new_tokens,
            )

        if blocks is not None:
            schedule(req, num_new_tokens)
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens

    # 4. 使用剩余预算接纳 WAITING 请求
    while waiting and token_budget > 0:
        if len(running) >= max_num_seqs:
            break

        req = waiting.peek()  # FCFS 或 Priority

        # Prefix Cache 查询
        cached_blocks, cached_tokens, _ = (
            kv_cache_manager.get_computed_blocks(req)
        )

        num_new_tokens = req.num_tokens - cached_tokens

        # Chunked Prefill
        if enable_chunked_prefill:
            num_new_tokens = min(
                num_new_tokens,
                token_budget,
                input_budget,
            )

        blocks = kv_cache_manager.allocate_slots(
            req,
            num_new_tokens,
            new_computed_blocks=cached_blocks,
            full_sequence_must_fit=(
                scheduler_reserve_full_isl
            ),
        )

        if blocks is None:
            break

        waiting.pop()
        running.append(req)
        req.status = RUNNING
        req.num_computed_tokens = cached_tokens

        schedule(req, num_new_tokens)
        token_budget -= num_new_tokens
        input_budget -= num_new_tokens

    # 5. 生成本轮调度结果
    scheduler_output = SchedulerOutput(
        num_scheduled_tokens={
            req_id: num_tokens,
        },
        block_ids=request_block_tables,
    )

    # 6. 执行模型
    model_output = model_executor.execute_model(
        scheduler_output
    )

    # 7. 处理输出
    scheduler.update_from_output(
        scheduler_output,
        model_output,
    )

    for req in scheduled_requests:
        req.append_output_tokens()

        if check_stop(req):
            req.status = FINISHED
            kv_cache_manager.free(req)
```

整体调用链：

```
EngineCore.step()
    │
    ├─ Scheduler.schedule()
    │     ├─ 调度 RUNNING
    │     ├─ KVCacheManager.allocate_slots()
    │     ├─ 必要时 _preempt_request()
    │     ├─ 调度 WAITING
    │     └─ Prefix Cache 查询
    │
    ├─ ModelExecutor.execute_model()
    │
    └─ Scheduler.update_from_output()
          ├─ 追加输出 Token
          ├─ check_stop()
          └─ 释放完成请求的 KV
```

---

## 2. 阅读源码的推荐顺序

1. `EngineCore.step()`  
    理解“调度 → 模型执行 → 结果处理”的总流程。
    
2. `Request`  
    重点看：
    
    ```
    num_tokens
    num_tokens_with_spec
    num_computed_tokens
    status
    ```
    
3. `Scheduler.add_request()`  
    理解请求如何进入 `waiting`。
    
4. `Scheduler.schedule()`  
    先看 RUNNING 调度，再看 WAITING 准入。
    
5. `KVCacheManager.get_computed_blocks()`  
    理解 Prefix Cache 命中如何推进 `num_computed_tokens`。
    
6. `KVCacheManager.allocate_slots()`  
    理解 KV Block 分配、完整 ISL 检查和 Watermark。
    
7. `Scheduler._preempt_request()`  
    理解 KV 不足时如何释放请求并重新排队。
    
8. `Scheduler.update_from_output()`  
    理解 Token 输出、停止判断和资源释放。
    

主要文件：

```
vllm/v1/engine/core.py
vllm/v1/request.py
vllm/v1/core/sched/scheduler.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/block_pool.py
vllm/v1/core/sched/utils.py
```

---

## 3. 最核心的代码认识

vLLM V1 使用统一的计算进度模型：

```
num_new_tokens = (
    request.num_tokens_with_spec
    - request.num_computed_tokens
)
```

不同阶段只是差值不同：

```
Prefill
    → 差值较大

Chunked Prefill
    → 差值较大，但被本轮预算截断

Prefix Cache
    → 直接增加 num_computed_tokens

Decode
    → 差值通常为 1

Preemption 恢复
    → num_computed_tokens 重置后重新匹配或重算
```

调度器主要管理四种资源：

```
max_num_seqs
    → 活跃请求数量

max_num_batched_tokens
    → 单轮输入 Token 预算

max_num_scheduled_tokens
    → 单轮实际下发 Token 预算

KV Block Pool
    → 当前实际 KV Cache 容量
```

关键参数：

```
enable_chunked_prefill
    → Prefill 放不下时是否切块

scheduling_policy
    → FCFS 或 Priority

enable_prefix_caching
    → 是否复用相同前缀的 KV Block

scheduler_reserve_full_isl
    → 准入时检查完整 Prompt 能否容纳

watermark
    → 接纳新请求时保留 KV 安全余量

max_tokens
    → 最大输出长度，不会提前物理分配完整 KV
```

最终准入条件是：

```
队列顺序
    +
max_num_seqs
    +
Prefix Cache 命中
    +
本轮 Token Budget
    +
Chunked Prefill
    +
完整 ISL 检查
    +
Watermark
    +
可用 KV Block
```

KV 不足时：

```
RUNNING 请求分配失败
    ↓
Preemption
    ↓
释放 KV 引用
    ↓
num_computed_tokens = 0
    ↓
退回 Waiting Queue
    ↓
之后重新命中 Prefix Cache 或 Recompute
```

一句话总结：

> vLLM V1 用 `num_computed_tokens` 统一表示请求进度，用 Token Budget 控制每轮计算量，用 Paged KV Block 管理显存，并通过 Prefix Cache、Chunked Prefill 和 Preemption 完成高并发调度。