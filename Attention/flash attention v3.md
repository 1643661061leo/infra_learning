# FlashAttention v3（FA3）

> 口径：基于 NeurIPS 2024 最终论文与官方仓库整理。论文主要讨论 Hopper（H100/H800）上的全量 Attention kernel。

## 1. 一句话结论

**FA1/FA2 已减少 HBM 读写并改进任务切分，FA3 进一步利用 Hopper 的异步执行和 FP8 Tensor Core，让搬运、矩阵乘与 Softmax 尽量重叠。**

FA3 的三个关键词：

```text
Warp Specialization：搬运与计算分工
GEMM–Softmax Pipeline：矩阵乘与Softmax重叠
FP8：块量化 + 离群值打散，兼顾吞吐与精度
```

FP16/BF16 路径仍是**精确 Attention 算法**：不使用稀疏近似，也不改变标准 Attention 的语义；FP8 路径会引入常规量化误差。==FA3 主要优化访存和硬件利用率，计算复杂度仍通常为 `O(N²)`。==

---

## 2. FA3 在解决什么问题

标准 Attention：

```text
S = QKᵀ / √d
P = softmax(S)
O = PV
```

直接实现会把大小为 `[N, N]` 的 `S`、`P` 写入 HBM。FlashAttention 通过==分块和 online softmax==，只在片上存储当前 tile，避免物化完整的 `S`、`P`：

```text
Q tile 固定在片上
    ↓
依次读取 K/V tile
    ↓
增量更新 row max、row sum 和输出 O
```

FA2 已解决大量 IO 问题，但在 H100 上只达到约 35% 的理论峰值。原因是它仍偏同步：

- 数据搬运、GEMM 和 Softmax 之间存在等待；
- Hopper 的 TMA、异步 WGMMA、动态寄存器分配没有被充分利用；
- FP8 GEMM 更快后，吞吐较低的 `exp` 等 Softmax 操作更容易暴露为瓶颈。

---

## 3. 三项核心优化

### 3.0 硬件名词

可以先把一张 GPU 想象成一座有很多小车间的工厂：

| 名词                            | 通俗理解                                                                |
| ----------------------------- | ------------------------------------------------------------------- |
| **Hopper / H100**             | Hopper 是 NVIDIA 的一代 GPU 架构，H100 是采用它的一款显卡                           |
| **SM**                        | GPU 中的一个“小车间”，里面有计算单元、共享内存等                                         |
| **CTA**                       | 被分配到同一个 SM 中合作的一队线程，也叫 thread block；它不是硬件                           |
| **thread / warp / warpgroup** | thread 是最小工人；32 个 thread 是一个 warp；Hopper 中 4 个 warp 可组成一个 warpgroup |
| **HBM**                       | GPU 的大容量主存，像远处的仓库；容量大，但取数据有等待                                       |
| **SMEM**                      | 一个 SM 内的 Shared Memory，像车间工作台；小但快，可被同一 CTA 共享                       |
| **register**                  | 每个 thread 私有的极小高速存储，像工人的口袋；最快，但数量有限                                 |
| **tile / block**              | 把大矩阵切成、一次处理的一小块数据                                                   |
| **GEMM**                      | 矩阵乘法；Attention 中主要是 `QKᵀ` 和 `PV`                                    |
| **Tensor Core**               | 专门做矩阵乘法的计算单元，像高速专用机器                                                |
| **CUDA Core**                 | 做普通加、乘等运算的通用计算单元。`exp` 还会使用特殊函数单元，因此下图统称“常规计算单元”                    |
| **异步**                        | 发出任务后不原地等它完成，可以先安排其他工作                                              |
| **pipeline**                  | 把不同步骤重叠起来，像流水线一样连续处理多块数据                                            |

### 3.1 Warp Specialization：搬运与计算分工

一个 CTA 内的 warpgroup 被分成不同角色：

```text
Producer warpgroup
  用 TMA：HBM → 环形 Shared Memory Buffer
                ↓
Consumer warpgroup
  用 WGMMA：QKᵀ、PV
```

- **TMA**：Hopper 的专用数据搬运器，像叉车，把数据从 HBM 仓库搬到 SMEM 工作台；
- **WGMMA**：让整个 warpgroup 调用 Tensor Core 做矩阵乘法的异步指令；
- **环形 SMEM buffer**：工作台上准备多个可循环使用的格子；计算 A 时，搬运组把 B 放进下一个格子；
- **`setmaxnreg`**：调整寄存器配额。搬运组少留一些“口袋”，把更多“口袋”让给需要保存中间结果的计算组。

**图例 1：搬运和计算重叠**

```text
没有分工： [搬A] → [算A] → [搬B] → [算B] → [搬C] → [算C]
                       大量时间在互相等待

FA3：
Producer warpgroup：   [搬A] → [搬B] → [搬C] ─────→
Consumer warpgroup：           [算A] → [算B] → [算C]
                                 ↑ 搬B时同时算A
```


结果：隐藏 HBM 搬运和指令发射延迟，提高 Tensor Core 利用率。

### 3.2 GEMM–Softmax Pipeline：隐藏非矩阵计算

问题是两类硬件各有所长：

- Tensor Core 很擅长 `QKᵀ`、`PV` 这类矩阵乘；
- Softmax 包含求最大值、加法、除法和 `exp`，主要走常规计算路径。

单个 tile 内仍有依赖：

```text
QKᵀ → Softmax → PV
```

**图例 2：矩阵乘和 Softmax 重叠**

```text
时间阶段：          1          2             3          4

Tensor Core：    [QKᵀ(A)]   [QKᵀ(B)]       [PV(A)]    [PV(B)]
CUDA Core：                [Softmax(A)]   [Softmax(B)]
                              ↑ 同时工作      ↑ 同时工作

A块依赖：QKᵀ(A) ─→ Softmax(A) ─→ PV(A)
B块依赖：          QKᵀ(B) ─────→ Softmax(B) ─→ PV(B)
```

这里没有改变 A、B 各自的计算顺序，只是把不同块交错执行。两个 consumer warpgroups 还可以轮流承担矩阵乘和 Softmax，这就是 **ping-pong scheduling（乒乓调度）**。

两种互补方式：

- **Ping-pong scheduling**：两个 consumer warpgroups 交替做 GEMM 和 Softmax；
- **两级流水**：利用 WGMMA 的异步执行，让一块的 `PV` 与下一块的 Softmax 重叠。

**好处**：Tensor Core 做矩阵乘时，其他计算单元也在做 Softmax，减少硬件空转。

代价是==需要保存更多中间结果，寄存器压力上升。==论文中的三级流水反而因寄存器压力和编译器调度效果不佳，弱于两级流水。

### 3.3 FP8：更高吞吐，同时控制量化误差

FP8 Tensor Core 的 GEMM 吞吐更高，但 E4M3 尾数只有 3 bit，且 LLM 激活中的离群值会放大量化误差。FA3 使用：

1. **Block Quantization**：每个 Q/K/V tile 单独计算 scale，而不是整个 tensor 共用一个 scale；
2. **Incoherent Processing**：量化前用随机正交变换打散离群值。

**图例 3：先打散，再按块选择刻度**

```text
原始数据：     [ 100, 1, -1, 0.5 ]
                   ↑ 一个数远大于其余数
                       │ 正交变换
                       ▼
打散后（示意）：[ 50, 50, 49, 51 ]
                       │ 每个小块单独选择scale
                       ▼
FP8量化：      小块内数值接近，较少信息被粗刻度吞掉
```

对 Q、K 同乘正交矩阵 `M`：

```text
(QM)(KM)ᵀ = QMMᵀKᵀ = QKᵀ
```

因为 `MMᵀ = I`，变换前后的 Q、K 点积不变，也就是 Attention score 不变。实际使用的是一种快速 Hadamard 变换。

**好处**：使用更快的 FP8 Tensor Core，同时比简单的“整个 tensor 共用一个 scale”保留更多数值精度。

---

## 4. FA1、FA2、FA3 的关系

| 版本 | 核心目标 | 关键方法 |
|---|---|---|
| FA1 | 减少 HBM IO | Tiling、kernel fusion、online softmax |
| FA2 | 提高并行度与负载均衡 | 更好的序列切分、warp 分工，减少非 GEMM FLOPs |
| FA3 | 用满 Hopper 新硬件 | TMA/WGMMA 异步流水、warp specialization、FP8 |

最简记忆：

```text
FA1：少搬数据
FA2：更好地分工作
FA3：让搬运、GEMM、Softmax并行推进，并引入更稳健的FP8路径
```

---

## 5. 论文性能结论

测试环境为 **H100 80GB SXM5**，结果是 Attention kernel 的 microbenchmark：

- BF16 forward 相对 FA2：`1.5–2.0×`；
- BF16 backward 相对 FA2：`1.5–1.75×`；
- BF16 最高约 `840 TFLOPs/s`，约为理论峰值的 `85%`；
- FP8 forward 最高约 `1.3 PFLOPs/s`；
- FP8 的 RMSE 相比 per-tensor 量化基线降低 `2.6×`。

这些数字不能直接等价为模型端到端加速：实际收益还取决于 Attention 占比、序列长度、mask、head dimension、通信及其他算子。

---

## 6. 推理场景如何理解

```text
Prefill：Q长度较大，矩阵计算多 → 更容易受益于FA3的流水与Tensor Core
Decode：Q长度通常为1，主要读取大量KV → 更偏访存/调度瓶颈
```

因此论文中的长序列全量 Attention 数据，不能直接推导出 vLLM/SGLang Decode 的加速比例。在线推理还要综合看 paged KV cache、continuous batching、split-KV kernel 和调度开销。

---

## 7. 使用边界

- 论文和官方 CUDA 实现主要针对 **Hopper H100/H800**；
- 官方仓库当前仍将 FA3 标为 beta，要求 `CUDA >= 12.3`，推荐 CUDA 12.8；
- 官方已提供 FP16/BF16 forward、backward，以及 FP8 forward；
- FP8 并不是“免费加速”：需要量化策略、数据布局处理，并验证模型级精度；
- FA3 不会消除 Attention 的 `O(N²)` 计算量，也不能单独解决超长上下文的 KV Cache 容量问题。

---

## 8. 面试式总结

> FlashAttention-3 延续了 FlashAttention 的分块与 online softmax，但针对 Hopper 重新设计流水线：用 TMA producer 和 WGMMA consumer 做 warp specialization，以环形 Shared Memory buffer 重叠访存与计算，再跨 tile 重叠 GEMM 和 Softmax。FP8 路径通过块量化与正交变换打散离群值，在提高 Tensor Core 吞吐的同时降低量化误差。它提升的是 Attention kernel 的 IO 和硬件利用率；FP16/BF16 路径不改变标准 Attention 的语义，计算复杂度仍为二次。

## 参考资料

- [FlashAttention-3：NeurIPS 2024 论文](https://papers.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf)
- [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)
- [作者博客](https://tridao.me/blog/2024/flash3/)
