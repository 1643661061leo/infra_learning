# FlashAttention-3 / FlashAttention-4 核心改进

## 一、整体演进逻辑

FlashAttention 各版本优化的对象不同：

```
FA1：减少 HBM 读写
 ↓
FA2：提高并行度，改善工作划分
 ↓
FA3：充分利用 Hopper 的异步硬件
 ↓
FA4：解决 Blackwell 上新的非矩阵乘瓶颈
```

最值得记住的是：

> FA3 的重点是==让数据搬运、矩阵乘和 Softmax 尽可能重叠==；FA4 的重点是当 Tensor Core 变得更快后，==让 Softmax 和其他非 MMA 操作能够跟上==

---

# 二、FlashAttention-3：Hopper 上的三点改进

FA3 面向 ==NVIDIA Hopper GPU。==

Hopper 提供了三类重要能力：

```
TMA
→ 负责异步搬运数据

WGMMA / Tensor Core
→ 负责矩阵乘加

CUDA Core / 特殊函数单元
→ 负责 Softmax、exp、归约等操作
```

FA3 的核心问题是：

> 这些工作由不同硬件执行，怎样让它们同时工作，而不是依次等待？

FA3 论文将改进总结为三项：warp specialization 下的计算—搬运重叠、GEMM—Softmax 重叠，以及 FP8 的 block quantization 与 incoherent processing。[FlashAttention-3 论文](https://arxiv.org/abs/2407.08608)

---

## 1. Warp Specialization + TMA

### 原有问题

FA2 虽然已经使用分块计算，但整体执行过程仍然比较同步：

```
加载 K/V
   ↓
计算当前 tile
   ↓
再加载下一块 K/V
```

Tensor Core 可能因为等待下一块数据而空闲。

### FA3 的做法

FA3 将不同 warp 分配为不同角色：

```
Producer warp
    │
    └── 使用 TMA 加载下一块 K/V
                    ↓
              Shared Memory

Consumer warpgroups
    │
    └── 使用 WGMMA 执行 QKᵀ 和 PV
                    ↓
               Tensor Core
```

于是可以形成流水线：

```
时间 ─────────────────────────→

TMA：
Load K0/V0   Load K1/V1   Load K2/V2
████████     ████████     ████████

Tensor Core：
             GEMM 0       GEMM 1
             ████████     ████████
```

即：

```
计算当前 tile
       ||
加载下一个 tile
```

### 为什么有效

TMA 是专门的数据搬运单元，Tensor Core 是矩阵计算单元，两者能够并行工作。

因此，下一块数据的加载延迟可以被当前矩阵乘的执行时间隐藏。

### 一句话总结

> Warp specialization 负责分工，TMA 负责异步预取，从而让数据搬运和矩阵计算重叠。

---

## 2. GEMM 与 Softmax 重叠

### 原有问题

Attention 的基本依赖是：

$$
[  
S_j=QK_j^T  
]
$$
$$
[  
P_j=\operatorname{softmax}(S_j)  
]
$$
$$
[  
O_j=P_jV_j  
]
$$

对于同一个 tile，必须满足：

```
QK_jᵀ
  ↓
Softmax_j
  ↓
P_jV_j
```

这个依赖不能打破。

如果完全串行执行：

```
GEMM → Softmax → GEMM → Softmax
```

Tensor Core 在执行 Softmax 时就可能空闲。

### FA3 的做法

虽然同一个 tile 内部不能并行，但不同 tile 之间可以交错。

例如：

```
Tensor Core：
计算 QK_{j+1}ᵀ

        ||

CUDA Core / 特殊函数单元：
计算 Softmax_j
```

完整的流水线可以近似理解为：

```
时间 ─────────────────────────────────→

Tensor Core：
QK0       QK1       PV0       QK2
██████    ██████    ██████    ██████

Softmax：
           SM0                  SM1
           ████                 ████

TMA：
      Load K1/V1    Load K2/V2
```

同一时刻可能同时发生：

```
TMA             → 搬运下一块 K/V
Tensor Core     → 执行另一个 tile 的 GEMM
CUDA/SFU        → 执行上一个 tile 的 Softmax
```

FA3 还使用 ping-pong scheduling，让两个 consumer warpgroups 交替工作：

```
Warpgroup A 做 Softmax
       ||
Warpgroup B 使用 Tensor Core

随后交换角色
```

### 为什么有效

矩阵乘主要使用 Tensor Core，而 Softmax 主要使用普通浮点、归约和特殊函数单元。

只要跨 tile 调度，就能让两类硬件同时忙起来，从而把一部分 Softmax 时间隐藏在 GEMM 后面。

### 一句话总结

> FA3 没有消除 `QKᵀ → Softmax → PV` 的依赖，而是通过跨 tile 流水线，将一个 tile 的 Softmax 与另一个 tile 的 GEMM 重叠。

---

## 3. FP8 Attention

### 原有问题

Hopper Tensor Core 对 FP8 提供很高的计算吞吐，但 FP8 数值范围和精度有限。

如果整个张量只使用一个量化 scale：

```
整个 Q → 一个 scale
整个 K → 一个 scale
整个 V → 一个 scale
```

少量 outlier 会把 scale 拉大，导致其他普通数值只能使用 FP8 动态范围中的一小部分，量化误差随之增加。

Attention 又包含 Softmax，因此对 QK 分数误差比较敏感。

### FA3 的做法一：Block Quantization

FA3 使用更细粒度的分块量化：

```
Q block 0 → scale 0
Q block 1 → scale 1

K block 0 → scale 0
K block 1 → scale 1
```

不同 block 根据自己的数值范围选择 scale。

这样可以：

- 更充分地使用 FP8 动态范围；
    
- 减少局部 outlier 对整个张量的影响；
    
- 同时利用 FP8 Tensor Core 的高吞吐。
    

### FA3 的做法二：Incoherent Processing

FA3 还在量化前对数据进行保持内积不变的变换，将集中在少数位置的 outlier 能量分散到更多维度。

概念上可以理解为：

```
变换前：
[普通值, 普通值, 超大 outlier, 普通值]

变换后：
[较均匀, 较均匀, 较均匀, 较均匀]
```

数值更加均匀后，更适合使用 FP8 表示。

### 为什么有效

```
Block quantization
→ 局部选择量化范围

Incoherent processing
→ 分散 outlier

FP8 Tensor Core
→ 提高矩阵乘吞吐
```

论文报告 FA3 的 FP8 forward 接近 1.2 PFLOPS，并且相比基础 FP8 Attention，数值误差降低约 2.6 倍。[FlashAttention-3 论文](https://arxiv.org/abs/2407.08608)

### 一句话总结

> FA3 不是简单地把 FP16 换成 FP8，而是利用 block quantization 和 incoherent processing控制误差，再获得 FP8 Tensor Core 的吞吐优势。

---

## FA3 最终记忆版

```
1. Warp Specialization + TMA
   → 搬下一块数据与计算当前块重叠

2. GEMM + Softmax overlap
   → Tensor Core 与 Softmax 执行单元重叠

3. FP8 Attention
   → block quantization + incoherent processing
   → 提高吞吐，同时控制量化误差
```

FA3 的核心画面：

```
TMA             → 搬运下一块数据
       ||
Tensor Core     → 执行 GEMM
       ||
CUDA Core/SFU   → 执行 Softmax
```

> FA3 的核心是让 Hopper SM 中的不同硬件单元尽可能同时工作。

---

# 三、FlashAttention-4：Blackwell 上的两点重点改进

FA4 面向 Blackwell GPU。

Blackwell 出现了明显的非对称硬件增长：

```
Tensor Core 吞吐
       ↑↑↑

Shared Memory 带宽
       ↑

exp / 普通 ALU 吞吐
       ↑ 或基本不变
```

以 B200 为例，BF16 Tensor Core 每个 SM 的吞吐相较 H100 翻倍，但指数运算单元仍为每时钟每 SM 16 次操作。

因此，瓶颈发生了迁移：

```
以前：
Tensor Core 计算较慢
其他工作可以被 GEMM 隐藏

现在：
Tensor Core 很快
Softmax、exp 和 SMEM 反而跟不上
```

FA4 的分析表明，典型 Blackwell Attention 中，Shared Memory 流量和指数运算的预计耗时可能比 MMA 高出约 25%～60%。[FlashAttention-4 论文](https://arxiv.org/abs/2603.05451)

---

## ① TMEM + Fully Async MMA：让流水线排得更自由

### 1. Hopper 上的限制

在 Hopper 上，WGMMA 的 accumulator 与线程寄存器联系紧密。

Attention 又需要同时保存：

```
QKᵀ 的中间结果
PV 的累加结果
Softmax 的统计量
其他流水线状态
```

这些数据会造成较大的 register pressure。

寄存器使用过多会限制：

- 同时驻留的 warp 数量；
    
- tile 大小；
    
- 流水线深度；
    
- 可选择的调度顺序。
    

尤其在 backward 中，需要同时协调多个矩阵乘，中间 accumulator 放在寄存器里会产生很强的执行顺序约束。

---

### 2. Blackwell 的 TMEM

Blackwell 引入了：

```
TMEM
Tensor Memory
```

它是专门用于保存 Tensor Core 中间结果的片上存储。

Blackwell 的 MMA 可以：

```
发起 MMA
   ↓
Tensor Core 异步计算
   ↓
结果直接写入 TMEM
```

而不必让大量 accumulator 始终占据线程寄存器。

可以把两代硬件粗略理解为：

```
Hopper：

Tensor Core
    ↓
Register accumulator
    ↓
线程必须“拿着”计算结果
```

```
Blackwell：

Tensor Core
    ↓
TMEM
    ↓
结果先放在专用工作区
```

Blackwell 每个 SM 提供 256 KB TMEM，并支持面积更大的 `128×128` MMA tile；Hopper 的典型 WGMMA tile 为 `64×128`。[FlashAttention-4 论文](https://arxiv.org/abs/2603.05451)

---

### 3. Fully Async MMA

Fully async MMA 表示线程发出 MMA 后，不需要同步等待计算立即完成：

```
控制线程：
发出 MMA
   ↓
继续安排其他工作

Tensor Core：
后台执行 MMA
   ↓
把结果写入 TMEM
```

与此同时，其他 warpgroup 可以继续：

```
准备下一次 MMA
执行 Softmax
搬运下一块 K/V
处理输出修正
```

于是流水线可以更加灵活：

```
时间 ───────────────────────────────→

Tensor Core：
QK0 ───── QK1 ───── PV0 ───── QK2

Softmax：
          SM0                  SM1

TMA：
     Load K1/V1   Load K2/V2

TMEM：
保存并传递 MMA 中间结果
```

### 4. TMEM 与 Fully Async MMA 的分工

两者不是同一个概念：

```
TMEM
→ 解决中间结果放在哪里

Fully Async MMA
→ 解决 MMA 发出后，线程是否必须等待
```

组合起来后：

```
MMA 与线程执行流程解耦
+
accumulator 与寄存器解耦
```

FA4 因而能在 MMA、Softmax 和数据搬运之间安排更深、更自由的软件流水线。

### 最终效果

```
Register pressure ↓
中间结果调度自由度 ↑
可使用更大的 MMA tile
流水线深度 ↑
MMA 与 non-MMA overlap ↑
```

### 一句话总结

> TMEM 给 Tensor Core 中间结果提供专用工作区，fully async MMA 让线程发出计算后可以继续安排其他任务；二者共同减少流水线中的资源和顺序约束。

---

## ② Software Exp + Conditional Rescale：解决 Softmax 新瓶颈

Softmax 需要计算：

[  
p_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}  
]

其中包含大量：

```
max reduction
exp
sum reduction
output rescale
```

当 Tensor Core 速度翻倍而指数运算单元没有同步变快时，Softmax 就可能成为新的关键路径。

FA4 从两个方向优化：

```
Software Exp
→ 提高 exp 的总吞吐

Conditional Rescale
→ 减少需要执行的 rescale
```

---

### 2.1 Software Exp：把一部分 exp 分流给 FMA 单元

#### 原有问题

传统实现主要使用硬件特殊函数单元计算 `exp2`：

```
所有 exp
   ↓
MUFU.EX2
   ↓
指数单元排队
```

B200 的 Tensor Core 变快了，但 MUFU 的指数吞吐没有同步增长，因此可能出现：

```
Tensor Core 已完成 GEMM
        ↓
等待 Softmax 的 exp
```

#### FA4 的做法

FA4 使用普通 FMA 指令，通过范围缩减和多项式近似计算一部分 `exp2`。

概念上：

$$
[  
2^x=2^n2^r,\qquad x=n+r  
]
$$

对于范围有限的 (r)，可以使用多项式近似：

$$
[  
2^r\approx a_0+a_1r+a_2r^2+a_3r^3+\cdots  
]
$$

多项式可以使用 Horner 形式计算：

$$
[  
a_0+r(a_1+r(a_2+r a_3))  
]
$$

这非常适合 FMA：

```
FMA
 ↓
FMA
 ↓
FMA
```

于是：

```
硬件 MUFU：
计算一部分 exp

        ||

普通 FMA 单元：
用多项式计算另一部分 exp
```

#### 为什么有效

重点并不是：

> 软件近似一定比硬件 exp 更快。

真正的原因是：

> 硬件 exp 单元已经拥堵，而 FMA 单元仍有可利用的执行能力，因此可以把 exp 工作分流到两类硬件并行完成。

最终的有效指数吞吐来自：

```
MUFU.EX2
    +
FMA software exp
```

#### 一句话总结

> Software Exp 不是单纯替换硬件 exp，而是利用 FMA 多项式近似与硬件 exp 并行执行，缓解单一指数流水线的吞吐瓶颈。

---

### 2.2 Conditional Rescale：减少 Online Softmax 重缩放

#### 标准 Online Softmax

FlashAttention 分块扫描 K/V，并维护：

```
m → 当前 running max
l → 当前指数和
O → 当前输出累加器
```

处理一个新 tile 后：

$$
\max(m_{\text{old}},m_{\text{tile}})  

$$

旧的输出累加器需要转换到新的数值尺度：

$$
  
O_{\text{old}}  
\leftarrow  
O_{\text{old}}  
\exp(m_{\text{old}}-m_{\text{new}})  

$$

然后再加入当前 tile：

$$
O_{\text{old}}  
\exp(m_{\text{old}}-m_{\text{new}})  
+  
P_{\text{tile}}V_{\text{tile}}  

$$

每次 running max 变化，都可能需要：

```
计算 rescale factor
        +
缩放整行输出 accumulator
```

输出 accumulator 的维度通常较大，因此这种逐元素 rescale 并不便宜。

---

#### FA4 的做法

FA4 不再要求每次 running max 发生较小变化时，都立刻缩放整个旧输出。

概念上：

```
max 小幅变化
→ 暂时保留当前输出尺度

max 再次小幅变化
→ 继续保留

尺度差距达到条件
→ 执行一次统一修正
```

需要注意：

> “跳过 rescale”不是直接忽略尺度变化，而是保留相应的尺度信息，把输出修正延后到确实有必要时再执行。

因此：

```
传统方式：

tile0   tile1   tile2   tile3
          ↓       ↓       ↓
       rescale rescale rescale
```

```
Conditional Rescale：

tile0   tile1   tile2   tile3
                          ↓
                    必要时统一修正
```

FA4 还能借助 TMEM 将输出修正交给独立的 correction warpgroup，从关键路径中移走一部分 rescale 工作。

#### 为什么有效

Conditional Rescale 减少的是：

```
整行 accumulator 的逐元素乘法
+
相关的指数和尺度计算
```

所以它的核心并不是让单次 rescale 更快，而是：

> 减少 rescale 真正发生的次数，并把必要的修正安排到更合适的位置。

#### 一句话总结

> Conditional Rescale 延迟小幅尺度变化引起的输出修正，只有满足条件时才统一处理，从而减少 Online Softmax 中昂贵的逐元素缩放。

---

## Software Exp 与 Conditional Rescale 的关系

二者都针对 Softmax，但解决的是不同问题：

```
Softmax 新瓶颈
       │
       ├── exp 吞吐不足
       │       ↓
       │   Software Exp
       │   将部分工作分流给 FMA
       │
       └── 输出 rescale 太频繁
               ↓
          Conditional Rescale
          减少重缩放次数
```

可以这样记：

```
Software Exp：
必须做的 exp，换更多硬件一起做

Conditional Rescale：
不必立即做的 rescale，尽量少做或延后做
```

---

# 四、FA3 与 FA4 对比

|版本|目标硬件|主要矛盾|核心思路|
|---|---|---|---|
|FA3|Hopper / H100|异步硬件没有被充分重叠利用|让 TMA、Tensor Core 和 Softmax 单元同时工作|
|FA4|Blackwell / B200|Tensor Core 增长太快，Softmax、SMEM 等 non-MMA 工作跟不上|更自由的异步流水线，并专门降低 exp、rescale 等开销|

---
