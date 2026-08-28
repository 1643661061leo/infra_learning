# MTP 推理解码学习笔记

> 只讨论 DeepSeek-V3 MTP 在推理阶段如何作为 drafter，不展开训练目标和 loss。

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

## 2. 结构与符号

一个 DeepSeek-V3 MTP 模块包含：

```text
共享 Token Embedding：[V,H]
hnorm、enorm：         RMSNorm
eh_proj：              [2H,H]
一个额外 Transformer Block
共享 LM Head：         [H,V]
```

| 符号 | 含义 |
|---|---|
| `B` | 同时解码的请求数 |
| `S` | `current_token` 之前、已写入 Target KV Cache 的上下文长度 |
| `H` | 每个 token 的 hidden 特征数 |
| `V` | 词表中的候选 token 数 |
| `K` | 一轮 rollout 的 Draft token 数 |
| `a_b` | 第 `b` 条请求接受的 Draft token 数 |

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

第一轮状态来自 Target prefill/首次解码；后续来自上一轮 verification。Target 与 MTP 还分别维护 KV Cache。这里约定 Target Cache 截止到 `current_token` 的前一个位置，不包含 `current_token`，因此后面的 verification 输入需要从 `current_token` 开始。

---

## 4. MTP 单步预测一个候选 token

```python
# -------------------------------------------------
# ① 当前每条请求各有一个实际 token ID
# -------------------------------------------------

current_token_ids                          # [B]

# [B] = B 条请求，每条请求一个 token ID


# -------------------------------------------------
# ② 查共享 Embedding
# -------------------------------------------------

current_embedding = Emb(current_token_ids)  # [B,H]

# current_embedding[b,:] =
# 第 b 条请求当前 token 的 H 个 embedding 特征


# -------------------------------------------------
# ③ 归一化父 hidden 和当前 token embedding
# -------------------------------------------------

hidden_normed = hnorm(parent_hidden)         # [B,H]
embedding_normed = enorm(current_embedding)  # [B,H]


# -------------------------------------------------
# ④ 沿 hidden 维拼接两路信息
# -------------------------------------------------

fused = concat(
    [hidden_normed, embedding_normed],
    dim=-1,
)                                           # [B,2H]

# fused[b,:] 包含：
# 1. 历史上下文的 H 维语义表示
# 2. 当前实际 token 的 H 维表示


# -------------------------------------------------
# ⑤ 将 2H 维融合特征压缩回 H 维
# -------------------------------------------------

mtp_input = fused @ eh_proj                  # [B,H]

# eh_proj：[2H,H]
# mtp_input[b,:] 是第 b 条请求的 H 维 MTP 输入


# -------------------------------------------------
# ⑥ 运行一个额外 Transformer Block
# -------------------------------------------------

next_hidden = MTP_Block(
    mtp_input.unsqueeze(1),                 # [B,1,H]
    past_key_values=mtp_kv_cache,
).squeeze(1)                                # [B,H]

# next_hidden[b,:] =
# 第 b 条请求当前 MTP 步骤产生的 H 维表示


# -------------------------------------------------
# ⑦ 使用共享 LM Head 得到 V 个候选分数
# -------------------------------------------------

draft_logits = LM_Head(next_hidden)          # [B,V]

# [B,V] =
# B 条请求
# × 每条请求 V 个候选 token 分数


# -------------------------------------------------
# ⑧ 对每条请求的 V 个候选做 softmax
# -------------------------------------------------

draft_probs = softmax(draft_logits, dim=-1) # [B,V]

# draft_probs[b,:] =
# 第 b 条请求对下一个 token 的 V 个候选概率
# 这一行 V 个概率之和为 1


# -------------------------------------------------
# ⑨ 每条请求采样一个候选 token
# -------------------------------------------------

draft_token = sample(draft_probs)            # [B]

# draft_token[b] =
# 第 b 条请求从自己的 V 个候选中采出的一个 token ID
```

单步关系：

```text
parent hidden [B,H] + current token [B]
→ next hidden [B,H] + Draft distribution [B,V]
→ Draft token [B]
```

---

## 5. 连续 rollout K 个 Draft token

```python
parent = parent_hidden                     # [B,H]
current = current_token_ids                 # [B]

draft_tokens = []
draft_prob_list = []

for k in range(K):
    # -------------------------------------------------
    # ① 当前实际/候选 token 转成 embedding
    # -------------------------------------------------

    token_embedding = Emb(current)          # [B,H]

# token_embedding[b,:] =
# 第 b 条请求当前 rollout token 的 H 维 embedding


    # -------------------------------------------------
    # ② 与产生当前 token 的父 hidden 融合
    # -------------------------------------------------

    fused = concat(
        [hnorm(parent), enorm(token_embedding)],
        dim=-1,
    )                                       # [B,2H]

    mtp_input = fused @ eh_proj             # [B,H]


    # -------------------------------------------------
    # ③ MTP Block 产生下一个位置的 hidden
    # -------------------------------------------------

    next_hidden = MTP_Block(
        mtp_input.unsqueeze(1),             # [B,1,H]
        past_key_values=mtp_kv_cache,
    ).squeeze(1)                            # [B,H]


    # -------------------------------------------------
    # ④ 每条请求得到 V 个候选概率
    # -------------------------------------------------

    logits_k = LM_Head(next_hidden)          # [B,V]
    probs_k = softmax(logits_k, dim=-1)      # [B,V]

# probs_k[b,:] =
# 第 b 条请求在第 k 个 Draft 位置的 V 个候选概率


    # -------------------------------------------------
    # ⑤ 每条请求采样一个 Draft token
    # -------------------------------------------------

    token_k = sample(probs_k)               # [B]

# token_k[b] =
# 第 b 条请求在第 k 个 Draft 位置采出的 token ID


    # -------------------------------------------------
    # ⑥ 保存 token 和完整 Draft 分布
    # -------------------------------------------------

    draft_tokens.append(token_k[:, None])   # 每项 [B,1]
    draft_prob_list.append(probs_k[:, None])# 每项 [B,1,V]


    # -------------------------------------------------
    # ⑦ 当前输出成为下一步输入
    # -------------------------------------------------

    parent = next_hidden                    # [B,H]
    current = token_k                       # [B]
```

拼接 `K` 步：

```python
draft_tokens = concat(draft_tokens, dim=1)  # [B,K]

# draft_tokens[b,k] =
# 第 b 条请求、第 k 个位置的候选 token ID

draft_probs = concat(draft_prob_list, dim=1)# [B,K,V]

# draft_probs[b,k,:] =
# 第 b 条请求、第 k 个位置的 V 个候选概率
```

这部分是自回归的：后一个候选依赖前一个采样结果。因此 rollout 越长，Draft 延迟和误差累积通常越明显。

---

## 6. 构造 Target 验证输入

先看最简单的单路径、统一长度 `K` 情况。

```python
# -------------------------------------------------
# ① 将当前真实 token 放在 Draft block 前面
# -------------------------------------------------

verify_input_ids = concat(
    [
        current_token_ids[:, None],         # [B,1]
        draft_tokens,                       # [B,K]
    ],
    dim=1,
)                                           # [B,K+1]

# 第 b 条请求的输入：
# [current_token, draft_1, ..., draft_K]
#
# [B,K+1] =
# B 条请求
# × 每条请求 1 个当前 token + K 个 Draft token
```

位置对齐：

```text
current_token 位置的 Target logits → 验证 draft_1
draft_1 位置的 Target logits       → 验证 draft_2
...
draft_K 位置的 Target logits       → 生成 bonus token
```

---

## 7. Target 一次验证整个 block

```python
# -------------------------------------------------
# ① Target 使用已有 KV Cache，一次处理 K+1 个位置
# -------------------------------------------------

target_output = TargetModel(
    input_ids=verify_input_ids,             # [B,K+1]
    past_key_values=target_kv_cache,
    output_hidden_states=True,
)

target_logits = target_output.logits         # [B,K+1,V]

# [B,K+1,V] =
# B 条请求
# × 每条请求 K+1 个输入位置
# × 每个位置 V 个候选 token 分数


# -------------------------------------------------
# ② 每个位置的 V 个分数转换成概率
# -------------------------------------------------

target_probs = softmax(target_logits, dim=-1)
                                                # [B,K+1,V]

# target_probs[b,j,:] =
# 第 b 条请求在第 j 个输入位置对下一个 token 的 V 个概率


# -------------------------------------------------
# ③ 前 K 行与 MTP 的 K 个 Draft 分布对齐
# -------------------------------------------------

target_verify_probs = target_probs[:, :K, :]# [B,K,V]

# target_verify_probs[b,k,:]：Target 的 V 个候选概率
# draft_probs[b,k,:]：       MTP 的 V 个候选概率
# draft_tokens[b,k]：        MTP 实际选出的 token ID
```

最后一行：

```python
bonus_probs = target_probs[:, K, :]          # [B,V]

# 只有全部 K 个 Draft token 都接受时才使用
```

---

## 8. Rejection Sampling

```python
# -------------------------------------------------
# ① 取出 Target 给每个已提议 token 的概率
# -------------------------------------------------

selected_target_prob = gather(
    target_verify_probs,                    # [B,K,V]
    draft_tokens,                           # [B,K]
)                                           # [B,K]

# selected_target_prob[b,k] =
# target_verify_probs[b,k,draft_tokens[b,k]]


# -------------------------------------------------
# ② 取出 MTP 自己给同一 token 的概率
# -------------------------------------------------

selected_draft_prob = gather(
    draft_probs,                            # [B,K,V]
    draft_tokens,                           # [B,K]
)                                           # [B,K]


# -------------------------------------------------
# ③ 计算每个候选 token 的接受概率
# -------------------------------------------------

accept_prob = minimum(
    1,
    selected_target_prob / selected_draft_prob,
)                                           # [B,K]

# accept_prob[b,k] =
# 第 b 条请求、第 k 个 Draft token 的接受概率


# -------------------------------------------------
# ④ 对每个位置进行随机接受测试
# -------------------------------------------------

accept_mask = random(shape=(B, K)) < accept_prob
                                                # [B,K]


# -------------------------------------------------
# ⑤ 第一次拒绝后丢弃同一请求的所有后缀
# -------------------------------------------------

accept_prefix = cumprod(accept_mask, dim=1) # [B,K]

# 例如：
# accept_mask   = [1,1,0,1]
# accept_prefix = [1,1,0,0]


# -------------------------------------------------
# ⑥ 得到每条请求接受的连续 Draft token 数
# -------------------------------------------------

accepted_count = accept_prefix.sum(dim=1)   # [B]

# accepted_count[b] = a_b
```

---

## 9. correction 与 bonus

对第 `b` 条请求：

```python
a_b = accepted_count[b]                    # 标量
```

中途拒绝时：

```python
if a_b < K:
    # -------------------------------------------------
    # ① 取第一个拒绝位置的完整 Target/Draft 分布
    # -------------------------------------------------

    target_dist = target_verify_probs[b, a_b, :]
                                                # [V]
    draft_dist = draft_probs[b, a_b, :]      # [V]

# [V] = 被拒绝位置对整个词表 V 个 token 的概率


    # -------------------------------------------------
    # ② 从归一化残差分布采样 correction
    # -------------------------------------------------

    residual = clamp(target_dist - draft_dist, min=0)
                                                # [V]
    residual = residual / residual.sum()     # [V]
    next_token_b = sample(residual)          # 标量


    # -------------------------------------------------
    # ③ 提交接受前缀和 correction
    # -------------------------------------------------

    committed_b = concat(
        [draft_tokens[b, :a_b], next_token_b[None]],
        dim=0,
    )                                       # [a_b+1]
```

全部接受时：

```python
else:
    # -------------------------------------------------
    # ① Target 最后一行预测 Draft block 后的 token
    # -------------------------------------------------

    next_token_b = sample(bonus_probs[b, :])# 标量


    # -------------------------------------------------
    # ② 提交全部 Draft token 和 bonus
    # -------------------------------------------------

    committed_b = concat(
        [draft_tokens[b, :], next_token_b[None]],
        dim=0,
    )                                       # [K+1]
```

总结：

```text
中途拒绝：接受前缀 + correction
全部接受：全部 Draft + bonus
```

---

## 10. 更新缓存并进入下一轮

```python
# -------------------------------------------------
# ① 只提交真实接受的 token
# -------------------------------------------------

output_ids[b] = concat(
    [output_ids[b], committed_b],
    dim=0,
)


# -------------------------------------------------
# ② 丢弃拒绝位置后的临时 KV Cache
# -------------------------------------------------

target_kv_cache.crop_to_committed_prefix()
mtp_kv_cache.crop_to_committed_prefix()

# rejected suffix 没进入最终序列，不能留在 Cache


# -------------------------------------------------
# ③ 为下一轮准备对齐的 hidden-token 对
# -------------------------------------------------

parent_hidden_next = hidden_that_predicted(next_token_b)
                                                # [H]
current_token_next = next_token_b           # 标量

# 下一轮继续：
# parent_hidden_next + Emb(current_token_next)
# → MTP 预测再下一个候选
```

闭环：

```text
本轮 correction/bonus → 下一轮 current token
产生它的 Target hidden → 下一轮 parent hidden
```

---

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

### 优点

- 复用 Target hidden，并共享 embedding/LM Head，候选通常比无关小模型更贴近 Target。
- 一个 MTP Block 远轻于重新执行完整 Target。
- 不需要完全独立的 Draft 模型或 tokenizer。
- 可结合 EAGLE 候选树、批量验证与 overlap scheduler。
- 正确 rejection sampling 保持 Target 输出分布不变。

### 缺点

- 多步 rollout 是自回归的，后一步依赖前一步候选。
- 长候选链会累积误差，后缀接受率通常下降。
- 一个完整额外 Transformer 层的权重、KV Cache 和通信开销不可忽略。
- 加速依赖 batch size、接受率、树大小与硬件利用率，配置不当可能负优化。
- 双 Cache、候选树、回滚和 CUDA Graph shape 增加部署复杂度。

---

## 14. 与 DSpark 的推理区别

| 对比项 | MTP / EAGLE | DSpark |
|---|---|---|
| Draft 主体 | 一个额外 Transformer Block 多步 rollout | 并行 Backbone 一次产生整个 block |
| token 依赖 | 每一步依赖前一步候选 | Backbone 并行，轻量 Markov/RNN Head 串行 |
| 候选结构 | 常配合 top-k 候选树 | 主要是单条并行 block |
| Target 特征 | Target hidden + 当前 token embedding | Target 多层 hidden 作为每层 K/V context |
| 验证长度 | EAGLE/服务参数控制 | Confidence Head + hardware-aware scheduler |

```text
MTP：用轻量 Transformer Block 自回归地 rollout 候选。
DSpark：并行算整个 block，再用更轻的 Head 补 token 依赖。
```

---

## 15. 完整 Tensor 主线

```text
parent hidden： [B,H]
current token： [B]
        ↓ Embedding
current embed： [B,H]
        ↓ concat
fused：         [B,2H]
        ↓ eh_proj
MTP input：     [B,H]
        ↓ MTP Block
next hidden：   [B,H]
        ↓ shared LM Head
Draft probs：   [B,V]
        ↓ sample、重复 K 步

Draft tokens： [B,K]
Draft probs：  [B,K,V]
        ↓ 与 current token 拼接

Verify input： [B,K+1]
        ↓ Target 一次 forward
Target probs： [B,K+1,V]
        ↓ 前 K 行与 Draft probs 对齐

Accept prob：  [B,K]
        ↓ 最长连续接受前缀
Accepted count：[B]
        ↓
接受前缀 + correction/bonus
        ↓
更新 Target/MTP Cache 与下一轮 hidden-token 对
```

---

## 参考资料

- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)
- [DeepSeek-V3 官方 MTP 权重结构](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md)
- [SGLang DeepSeek-V3 MTP / EAGLE 文档](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx)
- [SGLang speculative decoding 文档](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/speculative_decoding.mdx)
