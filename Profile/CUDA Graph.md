# [[CUDA Graph]] ：从 Kernel Launch 开销到 LLM 推理

> 口径：按 2026 年 8 月 NVIDIA CUDA、PyTorch、vLLM、SGLang 与 TensorRT-LLM 官方资料整理。框架配置可能随版本变化，原理部分相对稳定。

## 1. 一句话结论

CUDA Graph 把一段会重复执行的 GPU 工作及其依赖关系预先记录成 DAG，完成一次实例化后，CPU 只需一次 `cudaGraphLaunch()` 就能提交整段工作：

```text
普通 Stream：CPU ─launch A─launch B─launch C─launch D─→
                  ↓        ↓        ↓        ↓
GPU：            kernel A  kernel B  kernel C  kernel D

CUDA Graph：  CPU ─────────graph launch───────────────→
                              ↓
GPU：            kernel A → kernel B → kernel C → kernel D
```

它的主要价值是减少 Python、框架、CUDA Runtime/Driver 的逐 kernel 提交开销，以及 GPU kernel 之间的空隙。它**不会自动减少算子本身的 FLOPs，也不等于 kernel fusion**。

最适合的场景是：**相同执行结构被反复运行，并且其中包含许多较短的 kernel**。LLM 的逐 token Decode 正是典型场景。

---

## 2. 它解决的到底是什么问题

一次普通 CUDA kernel launch 大致经历：

```text
Python/业务代码
  → 框架调度与参数准备
  → CUDA Runtime/Driver 准备 launch descriptor
  → GPU 接收任务
  → kernel 执行
```

这些步骤大多是异步的，但每个 kernel 仍要单独提交。当 kernel 本身只执行几微秒时，提交和调度开销就可能与计算时间处于同一量级，使 GPU 时间线上出现空隙：

```text
GPU eager： [K1]··[K2]··[K3]··[K4]··[K5]
                    ↑ “··”是等待下一次工作提交的气泡

GPU graph： [K1][K2][K3][K4][K5]
```

CUDA Graph 在实例化阶段提前完成大量验证、依赖分析和工作描述符准备；稳态运行时只提交整个图，因此==把重复支付的逐节点开销变成一次性的准备成本。==

### 2.1 一个示意计算

假设一轮有 30 个短 kernel，每个计算 3 μs，普通方式平均每个 kernel 额外产生 4 μs 的提交或调度空隙。仅用于建立直觉：

```text
普通方式：30 × (3 + 4) = 210 μs
Graph：   30 × 3 + 一次Graph提交 ≈ 90 μs + 少量固定开销
```

实际收益取决于 GPU、驱动、kernel 长度、依赖结构和 CPU 是否跟得上，不能直接套用这个数字。若单个 kernel 已运行数毫秒，省下几微秒通常不重要了。

---

## 3. CUDA Graph 的结构与生命周期

Graph 是一个有向无环图（DAG）：

```text
H2D Copy
   │
   ▼
Kernel A ─────┐
   │          │
   ▼          ▼
Kernel B   Kernel C
   └────┬─────┘
        ▼
     D2H Copy
```

- **节点（Node）**：kernel、memcpy、memset、host function、event、child graph 等操作；新版本 CUDA 还支持 memory node、conditional node 等高级节点。
- **边（Edge）**：节点之间的依赖；一个节点的前驱全部满足后，CUDA 才能调度它。
- **`cudaGraph_t`**：可编辑的图定义或模板。
- **`cudaGraphExec_t`**：图经过验证和实例化后的可执行快照，真正用于 replay。

完整生命周期是：

```text
1. Warmup
   初始化库、完成JIT/算法选择、建立内存池
        ↓
2. Definition / Capture
   得到 cudaGraph_t
        ↓
3. Instantiate
   cudaGraph_t → cudaGraphExec_t
        ↓
4. Replay × N
   cudaGraphLaunch(graphExec, stream)
        ↓
5. Destroy
   释放 graphExec、graph、stream 等资源
```

需要区分三种成本：

- Warmup 和 Capture：为第一次录制准备真实执行路径；
- Instantiate：可能较贵，但通常只做一次；
- Replay：追求的稳态低开销路径。

因此性能测试不能把首次捕获、实例化时间混入稳态 replay 延迟。

---

## 4. 最关键的语义：固定的是地址和执行结构，不是数据值

把 CUDA Graph 简化理解成“录屏”容易产生误解。更准确地说，它记录的是：

```text
执行哪些操作
+ 操作的依赖关系
+ kernel参数（包括指针值）
+ launch配置
```

以 PyTorch 为例，capture 时 kernel 读写了地址 `0x1000` 和 `0x2000`，以后 replay 仍会访问这两个地址。可以改地址里的数据，但不能直接换成另一个 tensor 地址：

```text
固定：static_input.data_ptr() == 0x1000
可变：0x1000 里本轮存的是 batch A，下轮可以覆盖成 batch B
```

标准做法因此是：

```text
动态输入 batch
      │ copy_
      ▼
固定地址的 static_input
      │
      ▼
graph replay
      │
      ▼
固定地址的 static_output
```

同理，capture 时按值传入 kernel 的普通标量不会因为之后修改同名 CPU 变量而自动变化。需要动态标量时，可把它放进固定地址的 device buffer，或在底层 CUDA API 中更新对应 graph node 参数。

### 4.1 “Graph 必须完全静态”也不绝对

要分 API 层次理解：

- 对一次不做 update 的 `cudaGraphExec_t` replay，拓扑、节点及其参数按实例化结果执行；
- CUDA 原生 API 可以更新单个 executable node 参数，也可以用拓扑兼容的新图调用 `cudaGraphExecUpdate()`；
- 若节点类型或整体拓扑发生重大变化，通常需要重新实例化；
- PyTorch `CUDAGraph.replay()` 的常规用法仍应按固定形状、固定地址、固定控制流来设计。

工程上更常见的做法不是每轮 update 整张图，而是准备**有限个静态 graph variant，再按运行时形状选择。**

---

## 5. 为什么 LLM Decode 特别适合 CUDA Graph

### 5.1 Prefill 与 Decode 的计算形态不同

```text
Prefill：一次处理很多Prompt token
         GEMM通常较大，GPU计算更饱和，单kernel提交开销占比较小

Decode： 每个请求每轮通常只新增1个token
         反复执行大量小kernel，更容易受CPU提交和GPU launch气泡影响
```

Transformer 一层包含 RMSNorm、QKV 投影、RoPE、Attention、输出投影、MLP、残差等工作；多层模型每生成一个 token 会启动大量 kernel。若 Decode batch 较小，kernel 也较短，CPU 逐个提交很可能成为明显开销。

因此典型推理路径是：

```text
调度器准备本轮输入和KV元数据
             ↓
把内容写入预分配的固定地址buffer
             ↓
选择匹配本轮batch形态的CUDA Graph
             ↓
一次replay完成模型forward
             ↓
采样并进入下一轮Continuous Batching
```

CUDA Graph 优化的是**单轮 forward 的提交过程**；Continuous Batching 决定每轮有哪些请求。两者互补，但一个偏静态、一个偏动态，因此框架需要做额外适配。

### 5.2 动态 batch 怎么套进静态 Graph

假设预先捕获的 Decode batch size 是：

```text
[1, 2, 4, 8, 16, 32]
```

本轮真实 batch size 为 13，可以选择：

```text
13个真实请求 → pad到16 → replay graph_bs_16
                           └─3个空槽，占该bucket的18.75%
```

这就是 **shape/batch bucketing**：

- bucket 稀疏：graph 少，启动快、占用显存少，但 ==padding 浪费更多；==
- bucket 密集：更贴合真实 batch，计算浪费少，但 ==capture 时间、graph buffer 和显存占用增加；==
- 超出已捕获范围或形态不兼容：回退 eager，或临时捕获新 graph。

TensorRT-LLM、SGLang 等框架都提供 graph batch size 或最大 capture batch 的相关配置；它们的调优本质就是**在 padding 计算、显存、启动时间和覆盖率之间取舍。**

### 5.3 KV Cache 每轮都变，为什么仍能 replay

Paged KV Cache 的物理 block 分配会变化，但框架可以保持以下对象的存储地址稳定：

- token input buffer；
- position、sequence length 等元数据 buffer；
- slot mapping / block table buffer；
- KV Cache 大内存池的基址；
- logits 或采样输出 buffer。

每轮改变的是这些固定 buffer 中的**内容**。Attention kernel 通过更新后的 block table 间接访问本轮请求的 KV block，因此不必把每种请求组合都做成新图。

### 5.4 Full Graph、Piecewise Graph 与 Decode-only

CUDA Graph replay的是预先录制好的执行模板：

- **Shape 固定**：Shape 会影响 kernel 选择、grid/block 配置、临时内存大小和算子数量。Shape 变化后，原图可能不再适用。
- **内存地址固定**：Graph 记录的是指针地址。Replay 时仍访问相同地址，但地址中的数据可以更新。
- **动态选择 Kernel**：若每轮根据数据或 Shape 选择不同 kernel，执行拓扑会变化，无法复用同一张完整 Graph。
- **通信后端不兼容**：某些通信操作包含 CPU 同步、动态内存注册或不支持 Capture 的 stream/event 操作，不能放进 Graph。

所以使用 **Piecewise CUDA Graph**：

```
稳定计算段 → 动态/通信算子 → 稳定计算段
  Graph          Eager          Graph
```

它只捕获 Shape、地址和执行路径稳定的部分，把 Attention、MoE 路由或不兼容通信留在 Eager 模式。**这样牺牲少量 Graph 边界开销，换取更好的动态性和兼容性。**

| 模式                         | 做法                            | 优点          | 代价                             |
| -------------------------- | ----------------------------- | ----------- | ------------------------------ |
| Full Graph                 | 整个 forward 一张图                | launch 开销最低 | 对 Attention backend、形状和控制流要求最高 |
| Piecewise Graph（vllm/v1引入） | 捕获 graph-safe 片段，动态算子保持 eager | 兼容性、灵活性更好   | 图之间仍有 Python/Runtime 提交开销      |
| Decode-only                | 只捕获规则、重复度高的 Decode            | 收益/复杂度比通常较好 | Prefill 或 mixed batch 不受益      |
| Eager fallback             | 不兼容的 batch 正常逐算子执行            | 保证功能覆盖      | 该轮没有 graph 加速                  |

以当前 vLLM V1 为例，运行时会结合 batch 的 token 数、请求数、是否 uniform 以及后端能力，在 Full、Piecewise 和 eager 路径之间分派。这个设计说明生产系统中的 CUDA Graph 通常不是简单的“总开关”，而是针对不同 batch 形态的图缓存和 fallback 机制。

---

## 6. CUDA Graph、`torch.compile` 与 Kernel Fusion 的区别

| 技术                | 主要优化对象                   | 是否减少 kernel 数 |          是否减少提交开销 |
| ----------------- | ------------------------ | ------------: | ----------------: |
| CUDA Graph        | 一串 GPU 工作的提交与依赖调度        |         通常不减少 |                 是 |
| Kernel Fusion     | 多个算子的访存和中间结果             |             是 |              间接减少 |
| `torch.compile`   | Python/算子图、代码生成、融合和调度    |          可能减少 | 可能；也可结合 CUDAGraph |
| Persistent Kernel | 把调度循环长期留在一个 GPU kernel 内 |          大幅改变 |         是，但编程模型不同 |

它们可以叠加：先由编译器融合和生成更好的 kernel，再把剩余 kernel 序列捕获成 CUDA Graph。是否更快仍要实测，因为编译、图缓存和静态内存都会增加复杂度。

---
