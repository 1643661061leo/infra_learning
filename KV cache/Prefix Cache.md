# Prefix Cache 在 vLLM 与 SGLang 中的实现

> 口径：按 2026 年 8 月两项目官方文档与 `main` 分支代码整理。这里讨论单个推理实例内的 GPU KV Prefix Cache，不展开跨节点 KV 传输和多级缓存。

## 1. 一句话结论

Prefix Cache 保存已经计算过的前缀 KV。新请求若拥有完全相同的 token 前缀，就直接引用旧 KV，只 Prefill 不同的尾部：

```text
请求1：[公共System Prompt + 长文档] + 问题A
请求2：[公共System Prompt + 长文档] + 问题B
       └────── 直接复用KV ──────┘   └只计算这里
```

它主要降低重复 Prefill 的 TTFT 和计算量，不会加速后续逐 token Decode。

---

## 2. 缓存的是什么

Prefix Cache 缓存的不是文本、token 或最终 hidden state，而是：

```text
每一层Attention中，某段token对应的K/V张量
+ token前缀到KV物理位置的索引
```

若前缀 `[A B C D]` 已缓存，新请求为 `[A B C D X Y]`：

```text
Query：只需计算 X、Y
历史 Key/Value：直接引用 A~D 的缓存
新写 KV：X、Y
```

命中必须从第一个 token 开始连续匹配；中间相同、前面不同不能复用，因为某个 token 的 KV 已包含此前上下文的信息。

---

## 3. 数值例子：10 个 Prompt token

假设两个请求分别是：

```text
请求1：[S1 S2 S3 S4 D1 D2 D3 D4 Q1 Q2]
请求2：[S1 S2 S3 S4 D1 D2 D3 D4 X1 X2]
        └────── 8个公共前缀token ──────┘
```

请求 1 已经执行完 Prefill，因此公共部分的 KV 仍在缓存中。

### 3.1 没有 Prefix Cache

请求 2 必须重新计算全部 10 个 token：

```text
Prefill计算量 = 10 tokens
新分配KV      = 10 tokens
```

### 3.2 命中 Prefix Cache

调度器找到 8 个缓存 token 后：

```text
prefix_len      = 8
extend_input_len = 10 - 8 = 2

复用KV：[S1~S4, D1~D4]，共8个token
计算KV：[X1, X2]，共2个token
```

因此本次 Prefill 跳过了 80% 的输入 token。注意 `X1、X2` 的 Attention 仍可读取完整 8-token 历史 KV，模型语义没有改变。

若请求 2 与缓存内容完全相同，框架通常仍会重算最后一个 token，才能取得用于采样首 token 的 logits；实际命中长度还会受 block/page 对齐限制。

---

## 4. vLLM：哈希链 + Block Pool

vLLM 不维护 token 前缀树，而是按 KV block 建立全局哈希表。假设 `block_size=4`，请求 1 被分成：

```text
B0 = [S1 S2 S3 S4]
B1 = [D1 D2 D3 D4]
B2 = [Q1 Q2]          # 未填满
```

每个完整 block 的 key 近似为：

```text
H0 = hash(None, B0中的tokens, extra_keys)
H1 = hash(H0,   B1中的tokens, extra_keys)
```

父 block 的 hash 被放进下一个 block 的 hash，因此 `B1` 不仅表示 `D1~D4`，还隐含其前缀必须是 `S1~S4`。

请求 2 到来后的流程：

```text
1. 根据请求2的tokens计算H0、H1
2. get_computed_blocks()在哈希表中顺序查找
3. H0命中物理块B0，H1命中物理块B1
4. 增加两个块的ref_cnt，防止运行期间被驱逐
5. num_computed_tokens = 8，只为X1、X2分配新KV slots
6. 执行尾部Prefill；新完整块继续加入哈希表
```

请求结束时，可复用 block 的 `ref_cnt` 变为 0，但物理 KV 不会立即清空，而是进入 free queue 作为可驱逐缓存。内存紧张时，从队首取块并删除对应 hash，随后把物理块重新分配给新请求。

关键特点：

- 查找结构是 `block_hash → physical KV block`，不是 Trie；
- 标准路径主要在完整 block 边界命中，粒度受 `block_size/prefix_match_unit` 影响；
- `extra_keys` 会纳入 LoRA、Multimodal 输入、`cache_salt` 等信息，避免错误共享；
- 当前 V1 在模型支持时默认开启，也可显式配置 `enable_prefix_caching`。

---

## 5. SGLang：Radix Tree + KV 索引

SGLang 使用压缩 Radix Tree。树的 key 是 token 序列，value 是这些 token 在 KV pool 中的物理索引，而不是复制一份 KV：

```text
root
 └─ [S1 S2 S3 S4 D1 D2 D3 D4]
       ├─ [Q1 Q2 ...请求1后续tokens]
       └─ [X1 X2 ...请求2后续tokens]
```

请求 2 到来后的流程：

```text
1. match_prefix()沿Radix Tree查找最长公共前缀
2. 在第8个token后发生分叉，必要时把原树节点split
3. 返回8个KV位置到req.prefix_indices
4. PrefillAdder计算 extend_input_len = 10 - 8 = 2
5. 锁住命中的树路径，避免请求运行时被驱逐
6. 只对X1、X2执行Extend并写入新KV
7. 请求结束或暂存时，把新token路径和KV索引insert回Radix Tree
```

当请求结束后，未被运行请求引用的树节点变为 evictable，但不会立即释放；KV 不足时，从可驱逐叶节点开始回收。共享程度高的上层公共前缀通常会因仍有子树或引用而保留得更久。

关键特点：

- Radix Tree 直接表达公共前缀和分叉，命中位置可细到 `page_size` 对齐边界；
- `prefix_indices` 直接交给 Extend Attention 作为历史 KV；
- `lock_ref` 区分正在使用的 protected 节点与可回收节点；
- `extra_key` 与 `cache_salt` 为不同模型状态或租户隔离缓存命名空间；
- 可用 `lpm`、`dfs-weight` 等 cache-aware 策略优先调度高命中请求；
- Radix Cache 默认开启，`--disable-radix-cache` 可关闭。

---

## 6. 两者的核心差异

| 对比项 | vLLM V1 | SGLang |
|---|---|---|
| 索引结构 | 链式 block hash table | 压缩 Radix Tree |
| 基本复用单位 | KV block/hash unit | token path，按 page 对齐 |
| 跨轮状态 | block table、hash、`ref_cnt` | `prefix_indices`、tree node、`lock_ref` |
| 分叉表达 | 父 hash 隐式编码完整前缀 | 树节点显式产生多个分支 |
| 调度结合 | Scheduler 查询最长连续 block hit | Radix match 可直接参与 LPM/DFS 调度 |
| 回收方式 | ref=0 的缓存块留在 free queue，按分配压力驱逐 | 未锁定叶节点进入 evictable 集合，按策略逐叶驱逐 |

可以这样记：

```text
vLLM：把每段前缀压成hash，用hash找到共享KV block
SGLang：把token序列组织成树，沿树找到最长共享路径
```

---

## 7. 适用场景与误区

适合：

- 多个问题查询同一份长文档；
- 多轮聊天反复携带相同历史；
- 大量请求共享固定 System Prompt、Few-shot 示例或模板。

需要注意：

- 必须是相同 token ID 前缀；文本看起来相同但 chat template 或 tokenizer 不同，也可能无法命中；
- Prefix Cache 省 Prefill，不直接缩短长答案的 Decode 时间；
- 缓存命中省计算，也共享物理 KV，因此并非给每个请求复制一份缓存；
- 缓存会占用 KV pool，但闲置条目是可驱逐的；“请求结束后显存没立即下降”不等于泄漏；
- 单次、无公共前缀的流量收益很小，还会承担哈希或树维护开销；
- Prefix Cache 与 Chunked Prefill 可以同时工作：先命中公共前缀，再把未命中的长尾分 chunk 计算。

---

## 8. 参考资料

- [vLLM：Automatic Prefix Caching 使用说明](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [vLLM：Prefix Caching 设计](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
- [vLLM：KVCacheManager 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)
- [vLLM：BlockPool 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)
- [SGLang：RadixCache 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- [SGLang：Prefix Match 与 Cache-aware Scheduling](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)
- [SGLang：Server Arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)
