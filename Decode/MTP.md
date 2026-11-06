# MTP 推理解码学习笔记

> 只讨论 一般的 MTP 在推理阶段如何作为 drafter，不展开训练目标和 loss。

## 1. 一句话理解

MTP 使用“父位置的 Target hidden + 当前实际 token 的 embedding”，通过一个额外 Transformer Block 预测下一个候选 token；连续 rollout 得到草稿后，再由 Target 一次验证。

```text
Target hidden + 当前 token
             ↓
        MTP drafter
             ↓
      候选 token block
             ↓
       Target verification
             ↓
接受前缀 + correction/bonus
```

`MTP-1` 表示一个额外 MTP 模块，不表示候选可以跳过验证，也不表示一次直接提交两个最终 token。

---

## 2. 常见的MTP（详见4）
### 并行 MTP

多个 heads 从同一个 hidden 并行预测：

```python
hidden = Backbone(prefix)[:, -1, :]  # [B,H]
logits = MTP_Heads(hidden)            # [B,N,V]
```

第 `i` 个 head 通常只依赖原始前缀：
  
$$
q_i(x_{t+i}\mid x_{\le t})  
$$

所以后面的 head 不一定看过前面预测出的 token。

### 顺序 MTP

后一个预测依赖前一个 Draft token 或 hidden：

```text
prefix → draft₁ → draft₂ → ... → draftₙ
```

对应：

$$
q_i(x_{t+i}\mid x_{\le t},d_{<i})  
$$

这种结构候选更连贯，但 Draft 阶段存在串行计算。DeepSeek‑V3 MTP 更接近这一类。

---
## 3. 一轮开始时的状态

MTP 需要一对严格对齐的输入：

```python
parent_hidden                              # [B,H]
current_token_ids                          # [B]

# parent_hidden[b,:] =
# 第 b 条请求中，产生 current_token_ids[b] 时使用的父位置 hidden
#
# current_token_ids[b] =
# 第 b 条请求当前最新的真实 token ID
```

因果关系是：

```text
父位置 hidden h_i 产生当前 token x_{i+1}
MTP 使用 h_i + Emb(x_{i+1}) 预测 x_{i+2}
```

==第一轮状态来自 Target prefill/首次解码；后续来自上一轮 verification。==Target 与 MTP 还分别维护 KV Cache。这里约定 Target Cache 截止到 `current_token` 的前一个位置，不包含 `current_token`，因此后面的 verification 输入需要从 `current_token` 开始。

---

## 4. MTP 生成未来位置的预测

不同 MTP 结构采用不同方式生成未来 `K` 个位置。

### 并行 MTP

多个 heads 从同一个上下文 hidden 同时预测：

```
context_hidden = Backbone(prefix_ids)[:, -1, :]   # [B,H]

draft_logits = MTP_Heads(context_hidden)    # [B,K,V]
draft_probs = softmax(draft_logits, dim=-1) # [B,K,V]

draft_tokens = select(draft_probs)          # [B,K]
```

其中：

```
draft_probs[b,k,:]
= 第 b 条请求对第 k 个未来位置的预测分布
```

第 `k` 个 head 通常只依赖原始前缀：

$$
q_k(\cdot\mid x_{\le t})  
$$

因此所有位置可以并行预测，但后面的 head 没有看到前面实际选出的 Draft token。

### 顺序 MTP

顺序 MTP 逐步生成候选：

```
state = initial_draft_state
current = last_prefix_token

draft_tokens = []
draft_prob_list = []

for k in range(K):
    state, logits_k = MTP_Step(state, current)  # logits_k: [B,V]

    probs_k = softmax(logits_k, dim=-1)      # [B,V]
    token_k = select(probs_k)                # [B]

    draft_tokens.append(token_k[:, None])
    draft_prob_list.append(probs_k[:, None, :])

    current = token_k
```

拼接后：

```
draft_tokens = concat(draft_tokens, dim=1)  # [B,K]

draft_probs = concat(draft_prob_list, dim=1)  # [B,K,V]
```

此时第 `k` 步依赖前面的 Draft token：

$$
q_k(\cdot\mid x_{\le t},d_{<k})  
$$

DeepSeek‑V3 MTP 属于这种顺序生成的具体实现。

---

## 5. Draft block 的统一表示

无论内部采用并行还是顺序结构，单路径候选都可以表示为：

```
draft_tokens                            # [B,K]
draft_probs                             # [B,K,V]
```

其中：

```
draft_tokens[b,k]
= 第 b 条请求在第 k 个未来位置选择的 token

draft_probs[b,k,:]
= 产生该候选时使用的 V 维概率分布
```

但两类结构中 `draft_probs` 的条件不同：

```
并行 MTP：q_k(· | prefix)
顺序 MTP：q_k(· | prefix, draft_<k)
```

这个区别会影响随机采样时可以使用的接受算法。

---

## 6. Target 验证时的位置对齐

假设 Target KV Cache 已经包含完整前缀：

```
[x₁, ..., xₜ]
```

并且保存了前缀最后位置产生的：

```
prefix_next_logits                     # [B,V]
```

它用于预测第一个 Draft token。

将整个 Draft block 一次送入 Target（==这里是把draft token并行计算，从而只需要读取一次权重计算，而不需要读取K次权重计算，这也是投机解码快的关键==）：

```
target_output = TargetModel(
    input_ids=draft_tokens,             # [B,K]
    past_key_values=target_kv_cache,
)

draft_position_logits = target_output.logits  # [B,K,V]
```

位置关系是：

```
prefix_next_logits              → 验证 draft₁
draft_position_logits[:,0,:]    → 验证 draft₂
draft_position_logits[:,1,:]    → 验证 draft₃
...
draft_position_logits[:,K-2,:]  → 验证 draft_K
draft_position_logits[:,K-1,:]  → 生成 bonus token
```

如果系统没有保存 `prefix_next_logits`，也可以重新处理前缀最后一个 token。两种方式只是 Cache 约定不同。

---

## 7. 得到验证分布与 bonus 分布

将位置重新对齐：

```
target_verify_logits = concat(
    [
        prefix_next_logits[:, None, :],      # [B,1,V]
        draft_position_logits[:, :-1, :],    # [B,K-1,V]
    ],
    dim=1,
)                                           # [B,K,V]

target_verify_probs = softmax(
    target_verify_logits,
    dim=-1,
)                                           # [B,K,V]
```

其中：

```
target_verify_probs[b,k,:]
= Target 在 prefix + draft_<k 条件下
  对第 k 个 Draft token 的预测分布
```

最后一个位置用于产生 bonus：

```
bonus_probs = softmax(
    draft_position_logits[:, -1, :],
    dim=-1,
)                                           # [B,V]
```

Target 虽然一次处理整个 block，但因果 attention 会保证每个位置只能看到它前面的 token。

---

## 8. Greedy 解码的接受规则

Greedy 解码下，先取得 Target 在各位置选择的 token：

```
target_tokens = argmax(
    target_verify_probs,
    dim=-1,
)                                           # [B,K]

match_mask = draft_tokens == target_tokens  # [B,K]
```

第一次不匹配后，后续 Draft token 全部失效：

```
accept_prefix = cumprod(
    match_mask.to(int),
    dim=1,
)                                           # [B,K]

accepted_count = accept_prefix.sum(dim=1)   # [B]
```

对第 `b` 条请求：

```
a_b = accepted_count[b]
```

如果中途出现不匹配：

```
if a_b < K:
    correction = target_tokens[b, a_b]

    committed_b = concat(
        [
            draft_tokens[b, :a_b],
            correction[None],
        ],
        dim=0,
    )
```

如果全部匹配：

```
else:
    bonus = argmax(bonus_probs[b])

    committed_b = concat(
        [
            draft_tokens[b, :],
            bonus[None],
        ],
        dim=0,
    )
```

总结：

```
中途不匹配：接受前缀 + Target correction
全部匹配：全部 Draft + Target bonus
```

该规则适用于并行和顺序 MTP，并保持与普通 Target greedy decoding 相同的输出。

---

## 9. 随机采样的接受规则

标准 rejection sampling 要求 Drafter 提供实际生成候选时使用的条件分布：

$$
q_k(\cdot\mid x_{\le t},d_{<k})  
$$

因此它通常可以直接用于顺序 Drafter。

对于候选 `draft_tokens[b,k]`：

```
target_prob = target_verify_probs[b, k, draft_token]
draft_prob = draft_probs[b, k, draft_token]

accept_prob = min(
    1,
    target_prob / draft_prob,
)
```

如果接受，则继续验证下一个位置。

如果拒绝，则从残差分布采样 correction：

```
target_dist = target_verify_probs[b, k, :]  # [V]
draft_dist = draft_probs[b, k, :]           # [V]

residual = clamp(
    target_dist - draft_dist,
    min=0,
)

residual = residual / residual.sum()
correction = sample(residual)
```

如果全部 Draft token 都接受，则从 `bonus_probs` 采样 bonus。

需要注意：

```
顺序 MTP：
q_k 条件于前面的 Draft，可使用匹配的逐 token
rejection sampling。

并行 MTP：
q_k 通常只条件于原始前缀，不能直接套用上述公式。
应使用 greedy 验证、专门的联合验证算法，或明确接受近似采样。
```

因此，rejection sampling 不是所有 MTP 的固定组成部分。

---

## 10. 提交结果并更新 Cache

首先将本轮确认的 token 写入输出：

```
output_ids[b] = concat(
    [
        output_ids[b],
        committed_b,
    ],
    dim=0,
)
```

Target 验证时临时处理了全部 `K` 个 Draft token，但只能保留已经接受的连续前缀：

```
保留：原始前缀 + accepted Draft tokens
丢弃：第一个拒绝位置及其后缀
```

本轮产生的 correction 或 bonus 只是从 Target logits 中选出，通常还没有作为输入写入 KV Cache。因此可以选择：

```
方案一：立即对 correction/bonus 补一次 Target forward；

方案二：下一轮验证时，把它作为尚未进入 Cache 的
第一个输入一起处理。
```

MTP 状态的处理取决于具体结构：

```
并行 MTP：
通常没有独立的 Draft KV Cache，只需从新的前缀 hidden
重新产生未来预测。

顺序 MTP：
需要裁剪或重建 Draft 状态，不能保留被拒绝后缀对应的
临时 hidden 或 KV Cache。
```

通用闭环为：

```
生成 Draft block
      ↓
Target 批量验证
      ↓
接受连续前缀
      ↓
提交 correction 或 bonus
      ↓
将 Target 与 Draft 状态对齐到新前缀
      ↓
开始下一轮
```
## 11. SGLang 的 EAGLE/树形候选

上面是最容易理解的单路径。SGLang 将原生 MTP 按 EAGLE/NEXTN speculative decoding 使用，还可以：

- 连续执行多次 MTP forward；
- 每步保留 top-k 个候选；
- 形成候选树；
- Target 用 tree attention 批量验证节点；
- 最终只提交一条通过验证的连续路径。

若候选树有 `N` 个节点：

```text
node_logits： [N,V]  每个节点有 V 个候选分数
node_tokens： [N]    每个节点实际选择的 token ID
parent_ids：  [N]    每个节点的父节点编号
```

---

## 12. 如何启用

SGLang 当前示例：

```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --speculative-algorithm EAGLE \
  --trust-remote-code \
  --tp 8
```

当前文档默认参数：

```text
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

```text
speculative-num-steps：       每轮 MTP rollout 的 forward 步数
speculative-eagle-topk：      每步保留的候选分支数
speculative-num-draft-tokens：送给 Target 验证的节点预算
```

参数不是越大越好，应根据 batch size、接受率和硬件 benchmark。

---

## 13. 优缺点

总结：
- MTP 降低的是生成候选时的计算量和访存量；
- MTP 减少的是完整 Target 的串行调用次数；
- MTP 不会消除 Target 权重和 Target KV Cache；
- 相比独立的小 Draft 模型，MTP 因为共享 Embedding、LM Head 且只有少量层，额外显存可能更低。

### 优点

- 复用 Target hidden，并共享 embedding/LM Head，候选通常比无关小模型更贴近 Target。
- 一个 MTP Block 远轻于重新执行完整 Target，==K+1次串行forward>>k次便宜的forward+1次并行forward。==
- 不需要完全独立的 Draft 模型或 tokenizer。
- 可结合 EAGLE 候选树、批量验证与 overlap scheduler。
- 正确 rejection sampling 保持 Target 输出分布不变。

### 缺点

- 多步 rollout 是自回归的，后一步依赖前一步候选。
- 长候选链会累积误差，后缀接受率通常下降。
- 一个完整额外 Transformer 层的==权重、KV Cache 和通信开销==不可忽略。
- 加速依赖 batch size、接受率、树大小与硬件利用率，配置不当可能负优化。
- 双 Cache、候选树、回滚和 CUDA Graph shape 增加部署复杂度。

---

## 14. 与 DSpark 的推理区别

| 对比项       | MTP / EAGLE                        | DSpark                                     |
| --------- | ---------------------------------- | ------------------------------------------ |
| Draft 主体  | 一个额外 Transformer Block 多步 rollout  | 并行 Backbone 一次产生整个 block                   |
| token 依赖  | 每一步依赖前一步候选                         | Backbone 并行，轻量 Markov/RNN Head 串行          |
| 候选结构      | 常配合 top-k 候选树                      | 主要是单条并行 block                              |
| Target 特征 | Target hidden + 当前 token embedding | Target 多层 hidden 作为每层 K/V context          |
| 验证长度      | EAGLE/服务参数控制                       | Confidence Head + hardware-aware scheduler |

```text
MTP：用轻量 Transformer Block 自回归地 rollout 候选。
DSpark：并行算整个 block，再用更轻的 Head 补 token 依赖。
```

---

