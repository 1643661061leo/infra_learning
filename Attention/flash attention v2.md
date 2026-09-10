# FlashAttention v2（FA2）

> 口径：基于 2023 年 FA2 论文、作者说明和官方仓库整理，面向没有 GPU 系统基础的读者。

## 1. 一句话结论

**FA2 保留 FA1“少访问 HBM”的核心算法，进一步减少慢速的非矩阵运算，并重新切分 thread block 与 warp 的工作，让更多 GPU 计算单元同时忙起来。**

三个关键词：

```text
更少非矩阵运算：少做重复缩放和归一化
序列维度并行：长序列切成更多独立任务
Split-Q：每个warp负责不同输出，减少通信和同步
```

FA2 仍是完整的密集 Attention。它主要改变计算顺序和工作分配，不把计算复杂度从 `O(N²)` 变为线性。

---

## 2. FA1 还有什么问题

FA1 已经大幅减少 HBM 读写，但在 A100 上通常只达到理论最大计算吞吐的约 `25%–40%`。主要原因是：

1. Softmax 的 `exp`、除法、缩放等普通运算，相对 Tensor Core 矩阵乘更慢；
2. batch 或 head 数较少时，FA1 创建的任务不够多，一部分 SM 没活可做；
3. 一个 thread block 内的 warp 分工不理想，需要频繁通过 Shared Memory 交换并合并部分结果。

FA2 没有推翻 FA1，而是在同一算法骨架上优化“怎么算”和“工作怎样分”。

---

## 3. 名词解释

继续把 GPU 想成一座有许多车间的工厂：

| 名词 | 通俗理解 |
|---|---|
| **SM** | GPU 中的一个“小车间”，能运行若干队线程 |
| **thread** | 最小的执行工人 |
| **warp** | 32 个一起执行相同指令的 thread，可理解为一个固定小组 |
| **thread block / CTA** | 多个 warp 组成的协作团队，通常被安排到同一个 SM |
| **Tensor Core** | 专门高速执行矩阵乘法的机器 |
| **non-matmul FLOPs** | `exp`、比较、除法、缩放等非矩阵乘运算 |
| **occupancy / 利用率** | GPU 同时有多少资源正在做有效工作；任务太少时会有 SM 空闲 |
| **parallelism / 并行度** | 能拆出多少个可以同时执行的任务 |
| **Shared Memory / SMEM** | 同一 thread block 内 warp 共享的片上工作台 |
| **synchronization / 同步** | 多个小组互相等待，确认大家都完成某一步再继续 |
| **reduction / 归并** | 把多个小组算出的部分结果求和或合并成最终结果 |
| **sequence tile** | 沿序列方向切出的一段 Q 或 K/V |

---

## 4. 三项核心优化

### 4.1 减少 non-matmul FLOPs

在 A100 上，FP16/BF16 Tensor Core 的理论矩阵乘吞吐约为 `312 TFLOPs/s`，而论文引用的 FP32 非矩阵运算吞吐约为 `19.5 TFLOPs/s`。也就是说，同样数量的运算，`exp`、除法等更容易拖慢整体速度。

FA1 在处理每个 K/V 小块时，会反复缩放并归一化当前输出。FA2 改写 online softmax 的更新公式：循环中保留尚未最终归一化的输出，处理完全部小块后只统一除一次。

**图例 1：把重复归一化移到最后**

```text
FA1：
K/V块0 → 更新统计量 → 缩放、归一化O
K/V块1 → 更新统计量 → 再缩放、再归一化O
K/V块2 → 更新统计量 → 再缩放、再归一化O

FA2：
K/V块0 → 更新未归一化结果 ─┐
K/V块1 → 更新未归一化结果 ─┼→ 最后统一归一化一次 → O
K/V块2 → 更新未归一化结果 ─┘
```

每一块仍会做维持数值稳定所必需的校正，但总的缩放、除法等非矩阵运算更少。

**好处**：把更多时间留给 Tensor Core 擅长的矩阵乘。

### 4.2 沿序列维度增加并行任务

FA1 主要按 `batch × attention head` 创建 thread block。若 batch 和 head 数量小，即使序列很长，也可能没有足够任务填满 GPU。

FA2 进一步把 Q 序列切成多个行块。forward 中，每个 thread block 独立负责一块 Q 和对应的输出行，再依次扫描全部 K/V 块。

**图例 2：把一项长工作拆给更多车间**

```text
假设GPU有8个SM，但batch×head只有2组任务：

FA1：
SM0 [整段Q，任务A]    SM1 [整段Q，任务B]
SM2 [空闲]  SM3 [空闲]  SM4 [空闲] ... SM7 [空闲]

FA2把Q序列继续切块：
SM0 [Q0]  SM1 [Q1]  SM2 [Q2]  SM3 [Q3]
SM4 [Q4]  SM5 [Q5]  SM6 [Q6]  SM7 [Q7]
             ↑ 更多SM可以同时工作
```

在 backward 中，FA2 采用适合梯度计算的另一方向，主要按 K/V 的列块并行，以减少写入冲突。

**好处**：尤其在“batch/head 少但序列长”的情况下，提高 GPU 利用率。

### 4.3 从 Split-K 改为 Split-Q

一个 thread block 通常含有多个 warp。关键问题是：这些 warp 各自负责矩阵的哪一部分？

#### FA1：Split-K

FA1 让不同 warp 负责不同 K/V 列块，而 Q 由大家共享。每个 warp 都只能得到同一输出块的一部分，最后必须写进 Shared Memory，再同步和求和。

```text
Warp 0：Q × K0/V0 → 部分输出A ─┐
Warp 1：Q × K1/V1 → 部分输出B ─┼→ 写入SMEM → 等待 → 求和 → O
Warp 2：Q × K2/V2 → 部分输出C ─┤
Warp 3：Q × K3/V3 → 部分输出D ─┘
```

#### FA2：Split-Q

FA2 让不同 warp 负责不同 Q 行，K/V 由大家共享。每个 warp 从头到尾拥有不同的输出行，因此不用与其他 warp 合并输出。

**图例 3：每个小组直接完成自己的结果**

```text
共同读取K/V：          [K/V块]
                        ↓  ↓  ↓  ↓
Warp 0：[Q行0] ─────→ [最终O行0]
Warp 1：[Q行1] ─────→ [最终O行1]
Warp 2：[Q行2] ─────→ [最终O行2]
Warp 3：[Q行3] ─────→ [最终O行3]

各自负责不同输出，不需要跨warp求和。
```

**好处**：减少 Shared Memory 读写、同步和 reduction 开销。

这里的 Split-Q 只是一个 thread block 内部的 warp 分工，不要与多 GPU 推理框架里的张量并行混为一谈。

---

## 5. FA1 与 FA2 的关系

| 对比项               | FA1              | FA2              |
| ----------------- | ---------------- | ---------------- |
| 首要目标              | 减少 HBM IO        | 提高并行度和硬件利用率      |
| 分块、Online Softmax | 有                | 保留并改写更新公式        |
| 完整 `S`、`P` 写回 HBM | 不需要              | 不需要              |
| forward 的任务切分     | 主要按 batch、head   | 再沿 Q 序列切分        |
| block 内 warp 分工   | Split-K，需要合并部分输出 | Split-Q，各自拥有不同输出 |
| 非矩阵运算             | 已显著优化            | 进一步减少            |
| 主要计算复杂度           | `O(N²d)`         | 仍为 `O(N²d)`      |
| Attention 额外中间存储  | `O(N)`           | 仍为 `O(N)`        |

最简记忆：

```text
FA1：少搬数据
FA2：少做慢运算，并把工作分得更细、更合理
```

---

## 6. 论文性能结论

论文在 A100 上报告：

- 相比 FA1，Attention kernel 约快 `1.7–3.0×`；
- 最高达到约 `230 TFLOPs/s`，约为 A100 理论峰值的 `73%`；
- 相比普通 Attention 实现，约快 `3–10×`；
- GPT 风格模型的端到端训练，相比 FA1 最高约 `1.3×`，相比未使用 FlashAttention 最高约 `2.8×`。

kernel 加速不会等比例变成整个模型加速，因为模型还要执行线性层、MLP、通信等工作。实际结果也会随 GPU、序列长度、head dimension、mask、数据类型和框架版本变化。

---

## 7. 使用和理解边界

- FA2 仍计算完整的密集 Attention，不是稀疏或线性 Attention；
- 它继续使用 backward 重算，因此训练时仍是“多算一点、少存很多”；
- tile 越大不一定越好：它能减少管理开销，却会占用更多 Shared Memory 和 register；
- 论文性能主要来自 A100 测试，当前官方实现已扩展到更多 GPU，具体支持范围应以仓库说明为准；
- 长序列 Prefill/训练通常更容易发挥矩阵乘并行度；单 token Decode 还会受到 KV Cache 读取和调度限制。

---

## 8. 面试式总结

> FlashAttention v2 沿用 FA1 的 IO-aware 分块算法，重点提高 GPU 利用率。它改写 online softmax，减少较慢的缩放、除法等非矩阵运算；沿 Q 序列增加 thread block 数量，使小 batch、少 head、长序列时也有足够并行任务；并把 warp 分工从 Split-K 改为 Split-Q，让每个 warp 独立完成不同输出行，减少 Shared Memory 通信、同步和归并。因此 FA2 不改变 Attention 的 `O(N²)` 主要计算量和 `O(N)` 额外存储量，却比 FA1 更接近 GPU 的理论吞吐。

最简记忆：

```text
FA2 = FA1的少IO + 更少慢运算 + 更多并行任务 + 更少warp通信
```

---

## 参考资料

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [作者博客：FlashAttention-2](https://princeton-nlp.github.io/flash-atttention-2/)
- [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)

