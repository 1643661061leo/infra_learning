# FlashAttention v4（FA4）

> 口径：基于 2026 年 3 月 FA4 论文与官方仓库整理。论文重点是 Blackwell B200/GB200 上的 BF16 Attention；官方 CuTe-DSL 实现也面向 Hopper。

## 1. 一句话结论

**FA4 解决的是“新 GPU 各部分升级速度不一致”的问题：Blackwell 的矩阵乘法快了约一倍，但 Softmax 的指数计算和 Shared Memory 没有同样变快，因此 FA4 开始重点优化矩阵乘之外的瓶颈。**

```text
FA3：重点让搬运、矩阵乘、Softmax重叠

FA4：在此基础上继续解决
     ① Tensor Core太快，其他步骤跟不上
     ② Softmax的exp成为瓶颈
     ③ 反向传播反复读取Shared Memory
```

FA4 仍计算完整的密集 Attention，不是稀疏 Attention。它使用多项式近似计算一部分 `exp`，因此结果不保证逐 bit 相同，但论文显示其误差在 BF16 精度下基本被 BF16 自身的舍入误差覆盖。

---

## 2. 为什么 FA3 到 Blackwell 上不够用

H100 到 B200 的升级并不均衡：

- BF16 Tensor Core 理论吞吐：约 `1 → 2.25 PFLOPS`；
- 指数计算单元：仍约 `16 次/时钟/SM`；
- Shared Memory 读取带宽：仍约 `128 Byte/时钟/SM`。

可以把它理解成：工厂的加工机器翻倍了，但传送带和包装工没有翻倍。

**图例 1：瓶颈发生转移**

```text
相对硬件吞吐能力（条越长代表越快）

H100：
传送数据  ███████
矩阵乘法  ██████████
Softmax   ████████

B200：
传送数据  ███████          ← 速度提升较少
矩阵乘法  ████████████████████  ← 约快一倍
Softmax   ████████          ← exp速度基本未变

结果：更快的Tensor Core经常在等数据或等Softmax。
```

论文估算，在典型 Blackwell Attention 中，Shared Memory 和指数计算所需时间可能比矩阵乘多 `25%–60%`。

---

## 3. 名词解释

继续使用“GPU 是一座工厂”的类比：

| 名词                   | 通俗理解                                                                       |
| -------------------- | -------------------------------------------------------------------------- |
| **Blackwell / B200** | Blackwell 是 NVIDIA 的一代 GPU 架构，B200 是采用它的数据中心 GPU                           |
| **MMA**              | Matrix Multiply-Accumulate，矩阵乘并累加，是 Tensor Core 的主要工作                      |
| **accumulator**      | 矩阵乘的中间结果，后续还要继续累加                                                          |
| **TMEM**             | Blackwell 每个 SM 新增的 256 KB Tensor Memory，像紧挨 Tensor Core 的专用中转区，专门放矩阵乘中间结果 |
| **MUFU**             | Multi-Function Unit，负责 `exp` 等特殊函数，Softmax 会大量使用它                          |
| **FMA**              | 一次完成 `a × b + c` 的普通计算指令，适合高效计算多项式                                         |
| **CTA**              | 在同一个 SM 中合作的一队线程，也叫 thread block                                           |
| **2-CTA MMA**        | 两队线程合作完成一次更大的矩阵乘，每队只准备部分数据                                                 |
| **DSMEM**            | 同一个 cluster 中，不同 CTA 可以互相访问的 Distributed Shared Memory                     |
| **atomic add**       | 多个 CTA 更新同一结果时使用的安全累加操作；能避免覆盖，但通常较慢                                        |
| **rescale**          | online softmax 发现更大值时，对先前结果重新缩放，避免数值溢出                                     |
| **LPT 调度**           | Longest Processing Time First，优先安排耗时最长的任务，减少最后只剩一个长任务的拖尾                   |
| **CuTe-DSL**         | 写在 Python 中的 GPU kernel 描述语言，最终仍会编译成 GPU 机器指令执行                            |

TMEM 与前面提到的存储位置关系：

```text
HBM：    大仓库，容量最大，但距离远
SMEM：   车间公共工作台，小而快
TMEM：   Tensor Core旁的专用中转区，只服务矩阵计算
Register：每个线程自己的口袋，最快但很有限
```

---

## 4. 三项核心硬件优化

### 4.1 更大的全异步流水线

Hopper 的矩阵乘结果主要写入 register；中间结果太多时，线程的“口袋”容易装满，限制 tile 大小和流水深度。

Blackwell 的 MMA 可以把结果**异步直接写入 TMEM**：

- Tensor Core 不必等待结果写回 register；
- TMEM 可以暂存更多中间结果；
- 单个 MMA tile 从 Hopper 常见的 `64×128` 增大到 Blackwell 的 `128×128`；
- Softmax、矩阵乘、输出修正可以由不同 warpgroup 交错执行。

**图例 2：TMEM 让流水线容纳更多在途任务**

```text
FA3：
Tensor Core → [Register装中间结果] → Softmax/缩放 → 下一块
                    ↑ 口袋有限，容易产生等待

FA4：
Tensor Core： [QKᵀ(A)] [QKᵀ(B)] [PV(A)] [PV(B)] ──→
                         │         │
TMEM：                  [A块结果] [B块结果]
常规计算：                 [Softmax A] [Softmax B]
修正组：                              [修正旧输出]

不同硬件同时处理不同数据块，Tensor Core更少空转。
```

**好处**：更大的 tile 减少管理开销，TMEM 缓解寄存器压力，更深的流水线隐藏 Softmax 和访存时间。

### 4.2 给 Softmax 的 `exp` 分流，并减少 rescale

#### 用多项式近似一部分 `exp`

正常情况下，`exp` 主要由 MUFU 计算，但 MUFU 的吞吐没有随 Tensor Core 同步增长。FA4 将一部分 `2ˣ` 拆成：

```text
2ˣ = 2的整数次幂 × 2的小数次幂
```

- 整数部分可通过浮点数的 bit 操作快速构造；
- 小数部分用多项式和 FMA 近似；
- 只分流约 `10%–25%` 的元素，其余仍交给 MUFU。

**图例 3：两条通道并行计算 `exp`**

```text
原来：100个exp ─────────────────→ [MUFU单通道] → 排队

FA4： 75～90个exp ──────────────→ [MUFU]
      10～25个exp → [多项式近似] → [FMA单元]
                                      ↑ 两路同时工作
```

论文中三次多项式在 `99%` 的测试输入上与硬件结果相差不超过 `1 BF16 ULP`。这里的 ULP 可以理解为“BF16 相邻两个可表示数之间的一格”。

#### 不必每次都重新缩放

online softmax 会维护“目前见过的最大值”。传统做法只要最大值变化，就重新缩放之前的输出。FA4 允许先保留少量余量，只有最大值上升超过阈值时才 rescale，最后再统一归一化。

```text
新最大值只增加一点：继续累计，暂不rescale
新最大值增加很多：  执行rescale，避免溢出
最后：               用最终统计量统一归一化
```

论文常用阈值为 `log₂(256)=8`。最终归一化仍会校正结果。

**好处**：让原本闲置的 FMA 单元分担 `exp`，同时跳过大量不必要的缩放计算。

## 5. 调度优化：先做长任务

Causal Attention、变长请求和混合 Prefill/Decode 中，不同 tile 的工作量不一样。若短任务都先完成，最后可能只剩一个 SM 处理长任务，其他 SM 空闲。

**图例 5：LPT 减少拖尾**

```text
随意顺序：
SM0：[短][短][────────长────────]
SM1：[短][短][短]                 ← 提前空闲

最长优先：
SM0：[────────长────────][短]
SM1：[短][短][短][短][短]
                              ↑ 更接近同时结束
```

FA4 会结合 causal mask、head、batch 和 L2 Cache 局部性安排 tile；变长 batch 还可先按预计耗时排序。论文在 H200 上测得，LPT 对 MHA 提升约 `4%–8%`，对 MQA-8 提升约 `7%–14%`。

---

## 6. 为什么改用 CuTe-DSL

FA2/FA3 大量使用 CUDA C++ 模板，生成不同 kernel 时编译较慢。FA4 改用嵌入 Python 的 CuTe-DSL：

```text
Python中的CuTe-DSL描述
        ↓ 编译
PTX中间代码
        ↓ ptxas
SASS GPU机器指令
        ↓
在GPU上运行
```

因此“用 Python 写”不代表 kernel 在 Python 解释器里慢慢执行。论文中的单 kernel 编译时间：

| | FA3 | FA4 | 加速 |
|---|---:|---:|---:|
| Forward | 55 s | 2.5 s | 22× |
| Backward | 45 s | 1.4 s | 32× |

这是**编译速度**，不是 Attention 运行速度；主要价值是更容易开发、调试和组合新的 Attention 变体。

---

## 7. FA3 与 FA4 对比

| | FA3 | FA4 |
|---|---|---|
| 论文主要硬件 | Hopper H100 | Blackwell B200/GB200 |
| 主要矛盾 | 搬运、GEMM、Softmax 没充分重叠 | Tensor Core 太快，exp 与 SMEM 成为瓶颈 |
| 矩阵中间结果 | 主要放 register | MMA 直接写 TMEM |
| 流水线 | TMA/WGMMA + ping-pong | 更大 tile、全异步 MMA、TMEM 流水 |
| Softmax | 与 GEMM 重叠 | 继续重叠，并分流 `exp`、跳过部分 rescale |
| Backward | 单 CTA 为主 | 2-CTA MMA，减少 SMEM 和 atomic add |
| 实现语言 | CUDA C++/CUTLASS | Python 内嵌 CuTe-DSL，编译为 GPU 指令 |

最简记忆：

```text
FA3：把等待藏到矩阵乘后面
FA4：矩阵乘已经太快，开始给exp和Shared Memory“减负”
```

---

## 8. 论文性能结论

测试环境为 **B200、BF16**，序列长度 `1K–32K`，总 token 数固定为 32K：

- 相对 cuDNN 9.13：约 `1.1–1.3×`；
- 相对 B200 Triton 实现：约 `2.1–2.7×`；
- 最高约 `1613 TFLOPs/s`，即 B200 BF16 理论峰值的 `71%`；
- 中长序列（`≥4K`）收益更稳定；
- 较新的 cuDNN 已吸收多项类似技术，性能可能接近 FA4。

这些都是 Attention kernel microbenchmark，不能直接等价为模型端到端加速。

---

## 9. 推理场景与使用边界

- 论文主要测试完整的 Prefill/训练 Attention；Decode 通常更受 KV Cache 读取和调度影响；
- LPT 调度也考虑了变长 Decode 和 Prefill/Decode 混合 batch，但实际收益要结合推理框架测试；
- FA4 论文性能口径是 BF16，不要因为 Blackwell 支持 FP4，就把 FA4 默认理解为 FP4 Attention；
- 软件 `exp` 近似可能改变最低有效位，不适合要求逐 bit 一致的比较；
- backward 提供 deterministic 模式，但串行化原子归约会牺牲部分速度；
- 官方仓库中的 FA4 实现同时面向 Hopper 和 Blackwell，可通过 `flash-attn-4` 包安装；论文中的主要新算法和性能数据仍以 B200 为准。

---

## 10. 面试式总结

> FlashAttention-4 针对 Blackwell 的非对称硬件升级重新设计 Attention：Tensor Core 吞吐翻倍后，瓶颈转移到 Softmax 的指数计算和 Shared Memory。FA4 利用全异步 MMA、较大的 tile 和 TMEM 构建更深的流水线；让 FMA 单元通过多项式近似分担部分 `exp`，并跳过不必要的 online softmax rescale；backward 使用 2-CTA MMA 减少 Shared Memory 读取和 dQ atomic add。此外，它用 LPT 改善不均衡任务的拖尾，并用 CuTe-DSL 把单 kernel 编译时间缩短约 20–30 倍。

## 参考资料

- [FlashAttention-4 论文（arXiv 2603.05451）](https://arxiv.org/abs/2603.05451)
- [FlashAttention-4 HTML 全文](https://arxiv.org/html/2603.05451)
- [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)
- [官方 CuTe-DSL 实现目录](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute)
