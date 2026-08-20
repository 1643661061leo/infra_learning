## 1. 伪代码总结

下面的伪代码可以串联整套实现：

```
while server_is_running:
    # 1. 接收请求
    for request in receive_requests():
        req = build_req(request)
        waiting_queue.append(req)

    # 2. 将上轮完成 Prefill 的请求加入运行批次
    running_batch.merge_batch(last_prefill_batch)
    running_batch.filter_finished_requests()

    # 3. 尝试接收新的 Prefill 请求
    policy.sort(waiting_queue)

    adder = PrefillAdder(
        max_prefill_tokens=max_prefill_tokens,
        chunked_prefill_size=chunked_prefill_size,
        new_token_ratio=ratio_tracker.current,
        kv_available=kv_pool.available,
    )

    for req in waiting_queue:
        req.match_radix_prefix()

        if request_slot_is_full:
            break

        if adder.can_admit(req):
            prefill_batch.add(req)

    # 4. Prefill 优先执行
    if prefill_batch:
        allocate_extend_kv(prefill_batch)
        result = model.forward(prefill_batch)
        process_prefill_result(result)
        continue

    # 5. 没有 Prefill 时执行 Decode
    if running_batch:
        if not enough_kv_for_next_decode:
            evict_unused_radix_cache()

        if still_not_enough:
            retracted = running_batch.retract_decode()
            waiting_queue.extend(retracted)
            ratio_tracker.adjust_after_retract()
        else:
            ratio_tracker.decay_step()

        allocate_one_kv_token_per_request()
        result = model.forward(running_batch)
        process_decode_result(result)

        for req in running_batch:
            req.update_finish_state()
```

---

## 2. 阅读源码的推荐顺序

建议按照下面的顺序阅读：

1. `Scheduler.event_loop_normal()`  
    先理解一轮调度做什么。
    
2. `Scheduler.get_next_batch_to_run()`  
    理解 Prefill 和 Decode 的选择与批次合并。
    
3. `Scheduler.get_new_batch_prefill()`  
    理解新请求如何进入 Prefill。
    
4. `SchedulePolicy.calc_priority()`  
    理解 `schedule_policy` 的排序过程。
    
5. `Req.init_next_round_input()`  
    理解 RadixCache 命中和本轮输入范围。
    
6. `PrefillAdder.add_one_req()`  
    理解各种预算如何决定请求能否进入。
    
7. `ScheduleBatch.prepare_for_extend()`  
    理解 Prefill KV 如何实际分配。
    
8. `ScheduleBatch.prepare_for_decode()`  
    理解 Decode 每轮如何新增一个 KV Token。
    
9. `ScheduleBatch.check_decode_mem()`  
    理解 Decode 前的显存检查。
    
10. `ScheduleBatch.retract_decode()`  
    理解 KV 不足时如何撤回请求。
    
11. `BatchResultProcessor`  
    理解 Token 输出、停止条件和资源释放。
    
12. `NewTokenRatioTracker`  
    最后理解 `schedule_conservativeness` 和动态估算。
    

---

## 3. 最核心的代码认识

SGLang 的调度器本质上在同时管理四种资源：

```
请求槽位
Prefill 计算预算
当前 KV Cache
未来 Decode KV 预算
```

几个容易混淆的参数分别控制不同维度：

```
max_running_requests
    → 最多同时维护多少个活跃请求

max_prefill_tokens
    → 一轮最多计算多少 Prefill Token

chunked_prefill_size
    → 长 Prompt 一轮最多推进多少 Token

schedule_policy
    → 谁先尝试进入

schedule_conservativeness
    → 对运行请求未来 KV 增长估算得多保守

max_new_tokens
    → 请求最多生成多少 Token，同时为调度估算提供上界
```

最终准入逻辑不是单一阈值，而是：

```
排序结果
    +
请求槽位
    +
Prefix Cache 命中
    +
本轮 Prefill 预算
    +
Chunk 预算
    +
当前 KV 需求
    +
未来 Decode KV 估算
```

共同决定一个新请求能否进入运行系统。