```
"""FlashAttention v1/v2 的精简 PyTorch 教学实现。

只演示 forward、分块 Online Softmax 和 LSE。
输入形状：[batch, head, sequence, head_dim]。
`block_q`、`block_kv` 分别对应论文中的 B_r、B_c。
"""

import math
import torch

def _online_softmax_step(scores, old_max):
    """更新最大值，并把新块转换到同一指数基准。"""
    new_max = torch.maximum(old_max, scores.amax(dim=-1, keepdim=True))
    alpha = torch.exp(old_max - new_max)       # 旧统计量换到新最大值基准
    prob = torch.exp(scores - new_max)         # 当前块未归一化权重
    return new_max, alpha, prob


def flash_attention_v1(q, k, v, block_q=64, block_kv=64):
    """FA1：外层遍历 K/V，内层遍历 Q；每轮都归一化 O。"""
    assert q.shape == k.shape == v.shape  # 论文基本算法：Q、K、V ∈ R^(N×d)
    output_dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    batch, heads, q_length, _ = q.shape
    kv_length, dim_k = k.shape[-2:]
    scale = 1.0 / math.sqrt(dim_k)
    shape = (batch, heads, q_length, 1)
    row_max = torch.full(shape, -torch.inf, device=q.device)
    row_sum = torch.zeros(shape, device=q.device)
    out = torch.zeros_like(q)

    # FA1：先固定一块 K/V，再依次更新所有 Q 块。
    for kv_start in range(0, kv_length, block_kv):
        k_block = k[..., kv_start : kv_start + block_kv, :]
        v_block = v[..., kv_start : kv_start + block_kv, :]

        for q_start in range(0, q_length, block_q):
            q_end = q_start + block_q
            q_block = q[..., q_start:q_end, :]
            old_max = row_max[..., q_start:q_end, :]
            old_sum = row_sum[..., q_start:q_end, :]
            old_out = out[..., q_start:q_end, :]
            scores = q_block @ k_block.transpose(-2, -1) * scale
            new_max, alpha, prob = _online_softmax_step(scores, old_max)
            new_sum = alpha * old_sum + prob.sum(dim=-1, keepdim=True)

            out[..., q_start:q_end, :] = (
                alpha * old_sum * old_out + prob @ v_block
            ) / new_sum
            row_max[..., q_start:q_end, :] = new_max
            row_sum[..., q_start:q_end, :] = new_sum

    # 原始 FA1 保存 m、ell；这里额外返回等价的 LSE，方便与 FA2 对照。
    lse = (row_max + torch.log(row_sum)).squeeze(-1)
    return out.to(output_dtype), lse


def flash_attention_v2(q, k, v, block_q=64, block_kv=64):
    """FA2：外层遍历 Q；循环中保存分子 Z，最后只归一化一次。"""
    assert q.shape == k.shape == v.shape  # 论文基本算法：Q、K、V ∈ R^(N×d)
    output_dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    batch, heads, q_length, _ = q.shape
    kv_length, dim_k = k.shape[-2:]
    scale = 1.0 / math.sqrt(dim_k)
    outputs, lses = [], []

    # FA2：先固定 Q 块；不同 Q 块可由不同 thread block 并行计算。
    for q_start in range(0, q_length, block_q):
        q_block = q[..., q_start : q_start + block_q, :]
        rows = q_block.shape[-2]
        shape = (batch, heads, rows, 1)
        row_max = torch.full(shape, -torch.inf, device=q.device)
        row_sum = torch.zeros(shape, device=q.device)
        numerator = torch.zeros_like(q_block)

        for kv_start in range(0, kv_length, block_kv):
            k_block = k[..., kv_start : kv_start + block_kv, :]
            v_block = v[..., kv_start : kv_start + block_kv, :]
            scores = q_block @ k_block.transpose(-2, -1) * scale
            new_max, alpha, prob = _online_softmax_step(scores, row_max)

            row_sum = alpha * row_sum + prob.sum(dim=-1, keepdim=True)
            numerator = alpha * numerator + prob @ v_block
            row_max = new_max

        # FA2 的关键：遍历完所有 K/V 块后才除一次。
        outputs.append(numerator / row_sum)
        lses.append((row_max + torch.log(row_sum)).squeeze(-1))

    return torch.cat(outputs, dim=-2).to(output_dtype), torch.cat(lses, dim=-1)


# 用法：out1, lse1 = flash_attention_v1(q, k, v)
#      out2, lse2 = flash_attention_v2(q, k, v)
# 两者结果相同；差别主要是循环中维护“已归一化 O”还是“未归一化分子 Z”。

```