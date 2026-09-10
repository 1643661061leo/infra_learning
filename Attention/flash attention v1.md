# FlashAttention v1（FA1）

> 口径：基于 NeurIPS 2022 论文整理，面向没有 GPU 系统基础的读者。

## 1. 一句话结论

**FA1 不减少 Attention 必须完成的主要计算，而是避免把巨大的中间矩阵反复写入、读出 GPU 主存，因此既更快，也更省显存。**

三个关键词：

```text
Tiling + Online Softmax：分块计算，不保存完整注意力矩阵
Kernel Fusion：多个步骤在同一个GPU任务中连续完成
Recomputation：反向传播时重新计算小块，少保存中间结果
```

---

## 2. 普通 Attention 为什么慢

标准 Attention 可以简写为：

```text
S = QKᵀ / √d
P = softmax(S)
O = PV
```

- **Q（Query）**：当前 token 想找什么；
- **K（Key）**：每个 token 用来参与匹配的特征；
- **V（Value）**：匹配后真正取回的内容；
- `QKᵀ`：计算每对 token 的相关程度；
- `softmax`：把相关程度变成总和为 1 的权重。

若序列长度为 `N`，`S` 和 `P` 都有 `N×N` 个元素。`N` 翻倍，它们就变成原来的 **4 倍**。

**普通做法：中间矩阵多次往返大仓库**

```text
Q、K ─→ [计算QKᵀ] ─→ S矩阵 ─→ 写入HBM
                                  │
                                  ▼ 再读回来
                [计算Softmax] ←───┘
                       │
                       └─→ P矩阵 ─→ 写入HBM
                                          │
                                          ▼ 再读回来
                              V ─────→ [计算PV] ─→ O
```

GPU 的矩阵乘很快，等待这些数据搬运反而可能占用更多时间。FA1 因此被称为 **IO-aware（关注数据搬运的算法）**。这里的 IO 指 GPU 内存之间的数据移动，不是磁盘或网络。

---

## 3. 名词解释

把 GPU 想成一座工厂：

| 名词 | 通俗理解 |
|---|---|
| **GPU kernel** | 一次交给 GPU 执行的任务；启动和任务间交接都有开销 |
| **HBM** | GPU 的大容量主存，像远处的大仓库；容量大，但来回搬运较慢 |
| **SRAM / SMEM** | GPU 计算单元旁的小型高速存储，像车间工作台；很快但放不下大矩阵 |
| **register** | 单个线程自己的极小高速存储，像工人的口袋 |
| **tile / block** | 从大矩阵中切出、一次放到工作台处理的一小块 |
| **forward** | 根据输入计算 Attention 输出，即前向传播 |
| **backward** | 训练时根据误差计算梯度，即反向传播 |
| **FLOPs** | 完成了多少次数字运算 |
| **memory complexity** | 随输入变大，需要多少额外存储空间 |
| **exact attention** | 没有通过稀疏或抽样省略 token 对；数学目标仍是完整 Attention |

论文把片上高速存储统称为 SRAM；CUDA 代码中常见的是 Shared Memory（SMEM）和 register。理解时都可以先看作“小而快的车间空间”。

---

## 4. 三项核心优化

### 4.1 分块 + Online Softmax：小工作台也能处理大矩阵

FA1 不一次生成完整的 `N×N` 矩阵，而是：

1. 把 Q、K、V 切成小块；
2. 每次把一组小块搬到片上高速存储；
3. 计算当前注意力小块；
4. 只保存每行的运行中最大值、权重和与输出；
5. 继续处理下一块，并更新这些统计量。

**图例 1：完整大矩阵变成逐块处理**

```text
HBM大仓库                                  片上工作台

Q：[Q0][Q1][Q2] ───搬一块───────────────→ [Q0]
K：[K0][K1][K2] ───依次搬运─────────────→ [K0] → [K1] → [K2]
V：[V0][V1][V2] ───依次搬运─────────────→ [V0] → [V1] → [V2]
                                                │
                                                ▼
                                  计算分数、Softmax和局部输出
                                                │
                                                ▼
                                  更新：最大值、权重和、输出O0

完整的S、P矩阵始终没有写入HBM。
```

#### Online Softmax 为什么可行

普通 Softmax 好像必须先看到一整行，才能知道最大值和总和。Online Softmax 则像分批统计全班成绩：

```text
第1页：最高分 8，先记录这一页的加权结果
第2页：发现最高分 10
       ↓
把第1页的旧结果乘以 e^(8-10)，统一到新基准
       ↓
合并两页的权重和与加权结果，再继续读第3页
```

因此只要保留少量运行中统计量，就能得到与完整 Softmax 相同的数学结果，不必保留所有分数。

**好处**：额外 Attention 中间存储从随 `N²` 增长，降为随 `N` 增长。

### 4.2 Kernel Fusion：让数据留在工作台上

普通实现常把矩阵乘、Softmax、mask 等拆成多个 kernel。一个任务完成后，中间结果写回 HBM，下一个任务再读回来。

FA1 把这些步骤融合在一个 kernel 中，在片上直接衔接：

**图例 2：减少仓库往返**

```text
普通实现：
[QKᵀ] → HBM → [mask] → HBM → [Softmax] → HBM → [PV]
            多次写回、读出、启动新任务

FA1：
HBM → [ QKᵀ → mask → Online Softmax → PV ] → HBM
                中间数据留在片上
```

**好处**：减少 HBM 访问和 kernel 之间的交接开销。

### 4.3 Recomputation：用便宜的重算换昂贵的存储

训练的 backward 需要用到 forward 的部分中间结果。普通做法会保存庞大的 `S` 或 `P`；FA1 只保存输出 `O` 和每行的 Softmax 统计量。

backward 时，再从 Q、K、V 重新算出当前需要的 `S`、`P` 小块：

**图例 3：保存大矩阵 vs 按需重算**

```text
普通做法：
Forward  → 保存完整S/P ─────────────→ Backward读取
           ↑ 占用大量HBM

FA1：
Forward  → 只保存O和每行统计量 ─────→ Backward
                                         │
                              读取Q/K/V并重算当前小块
```

这会增加一部分 FLOPs，但矩阵乘在 GPU 上很快，而 HBM 搬运代价较高，所以“多算一点、少搬很多”通常更快。

---

## 5. FA1 到底改变了什么

| 项目 | 普通 Attention | FA1 |
|---|---|---|
| 数学结果 | 完整密集 Attention | 仍是完整密集 Attention |
| 主要计算量 | `O(N²d)` | 仍为 `O(N²d)` |
| Attention 额外中间存储 | `O(N²)` | `O(N)` |
| 完整 `S`、`P` 写入 HBM | 通常需要 | 不需要 |
| backward | 保存大中间矩阵 | 保存少量统计量，按块重算 |

因此，FA1 不是把 `O(N²)` Attention 变成线性计算，也不是默认使用稀疏 Attention。论文另有可选的 block-sparse 扩展，但它与密集版 FA1 是两件事。

“精确”也不表示每个 bit 都一定与另一种实现相同：浮点运算顺序改变时，末尾舍入可能略有差异；它表示没有用近似算法省略注意力项。

---

## 6. 论文性能结论

论文在 A100 等当时硬件和指定工作负载上报告：

- Attention kernel 在常见序列长度下，相对优化基线最高约 `3×`；
- 一个 GPT-2 Attention 示例中，Attention 计算最高约 `7.6×`；
- GPT-2 训练最高约 `3×`，长序列任务也获得明显加速；
- 显存占用随序列长度近似线性增长，使更长序列成为可能。

这些是 2022 年论文中的特定测试结果，不能直接当作当前模型的固定加速比例。端到端收益还取决于 Attention 占整个模型的比例、序列长度、数据类型、GPU、框架和其他算子。

---

## 7. 推理场景如何理解

```text
Prefill：一次处理较长输入，Q也很长
         → 形成较大的Attention矩阵，FA1思路通常很重要

Decode：每次通常只生成一个token，Q长度接近1
        → 更常受KV Cache读取、batch和调度影响
```

所以，FA1 论文中的长序列训练结果不能直接推导出逐 token Decode 的加速比例。

---

## 8. 面试式总结

> FlashAttention v1 是一个 IO-aware 的精确 Attention 算法。它用 tiling 和 online softmax 在片上逐块计算，不把完整的 `N×N` 注意力矩阵写入 HBM；通过 kernel fusion 减少中间结果往返；训练反向传播时再按块重算，而不是保存大矩阵。它没有降低 `O(N²)` 的主要计算量，但把 Attention 的额外中间存储降到 `O(N)`，并因显著减少 HBM 访问而加速。

最简记忆：

```text
FA1 = 切小块 + 留在片上算 + 只记统计量 + backward按需重算
```

---

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（NeurIPS 2022）](https://papers.nips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)
- [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)

