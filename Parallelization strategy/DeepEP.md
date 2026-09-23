# DeepEP：MoE 专家并行通信学习笔记

> 更新口径：2026-09-23。本文聚焦 DeepEP 的 EP 通信原理、数据流与推理应用。接口和框架参数变化较快，部署时应以实际版本的官方文档、`--help` 和启动日志为准。

## 1. 核心结论

**DeepEP 是 MoE 专家并行通信库**，核心操作是 Dispatch 和 Combine。

Shape 层面的特点：
Non-expand 去重 + 可选 Expand

元数据层面的特点：
EPHandle + Cached Dispatch

通信实现层面的特点：
GPU 发起通信 + 对称内存 + NVLink/RDMA +异步重叠 + 低 SM 占用 + FP8

---

## 2. DeepEP 在 MoE 系统中的位置

一个 MoE 层可以拆成五部分：

```
输入 hidden states
    ↓
Router / Top-K
    ↓
Dispatch 通信
    ↓
本地 Expert MLP
    ↓
Combine 通信与 Top-K 聚合
    ↓
MoE 输出
```

DeepEP 主要负责：

```
Router 输出之后
    ↓
Dispatch
    ↓
为本地 expert 组织输入布局
    ↓
Combine
```

它通常不负责：

- 计算 Router logits；
    
- 决定 logical Top-K experts；
    
- 执行完整的 expert MLP 数学计算；
    
- 切分或传输 Attention KV Cache；
    
- 自动解决所有 expert 负载不均问题；
    
- 决定请求调度和 batch 组成。
    

这些模块之间的关系是：

```
Router
  产生 topk_ids / topk_weights
          ↓
Expert placement
  把 logical expert 映射到 physical rank
          ↓
DeepEP
  执行 assignments 的 Dispatch / Combine
          ↓
本地 expert runner
  计算当前 rank 上的 experts
```

---

## 3. DeepEP 的 Shape 变化

前置输入沿用普通 EP 定义：

```
x            [T, H]
topk_idx     [T, K]
topk_weights [T, K]
```

DeepEP V2 的关键区别是 Dispatch 支持两种接收布局。

### 2.1 Non-expand 模式（默认）

```
do_expand = False
```

DeepEP 不会立即把每个 token 展开成 `K` 条 assignment，而是按“源 token—目标 rank”去重。

如果一个 token 选中的多个专家位于同一个目标 rank，该 token 只发送一次：

```
token a → expert 2、expert 3
expert 2、3 都位于 rank 1

普通 assignment 视角：
[a→e2, a→e3]       # 两条

DeepEP non-expand：
rank 1 只接收一次 a
```

目标 rank 上的布局为：

```
recv_x            [U_r, H]
recv_topk_idx     [U_r, K]
recv_topk_weights [U_r, K]
```

其中：

```
U_r = 发往 rank r 的不重复 token 数
```

`recv_topk_idx` 中不属于当前 rank 的专家位置会被标记为无效，例如：

```
原始 topk_idx： [e2, e7]
e2 在当前 rank，e7 在其他 rank

接收后：        [e2, -1]
```

专家计算在逻辑上可以理解为：

```
[U_r, H]
   ↓ 根据 recv_topk_idx 调用本地专家
[U_r, K, H]
   ↓ 本地聚合
[U_r, H]
```

随后 Combine：

```
combine([U_r, H], handle)
    → combined_x [T, H]
```

==`handle` 保存反向路由关系，因此 Combine 能直接复用，把结果送回原 token 所在 rank，并完成跨 rank 聚合。==

### 2.2 Expand 模式

```
do_expand = True
```

此模式才接近传统的“一条 assignment 对应一行”：

```
token a → expert 2、expert 3

展开后：
[a_for_e2, a_for_e3]
```

接收数据按照本地专家连续排列：

```
[expert 0 的 token]
[对齐 padding]
[expert 1 的 token]
[对齐 padding]
...
```

设本地专家 `e` 实际接收 `n_e` 个 token，对齐粒度为 `a`：

```
M_r = Σ align(n_e, a)
```

Dispatch 输出：

```
recv_x            [M_r, H]
recv_topk_idx     None
recv_topk_weights [M_r]
```

这里不再需要二维 `recv_topk_idx`，因为每一行已经属于唯一专家，专家归属由分段位置和 handle 中的元数据确定。

专家计算和 Combine：

```
recv_x       [M_r, H]
    ↓ 按专家分段执行
expert_out   [M_r, H]
    ↓ Combine：去除 padding、反向发送、加权聚合
combined_x   [T, H]
```

### 2.3 两种模式的核心区别

|模式|Dispatch 后每行代表什么|第一维|
|---|---|---|
|Non-expand|一个发往当前 rank 的不重复 token|`U_r`|
|Expand|一个具体的 token-expert assignment，包括 padding|`M_r`|

最值得记住的是：

```
Non-expand：
一个 token 在同一个目标 rank 上只出现一次，
多个本地专家通过 [U_r, K] 的路由表表示。

Expand：
一个 token-expert assignment 占一行，
数据已经按专家连续排布。
```

FP8 不改变上述 token 维度，只是 `recv_x` 可能变成：

```
(fp8_data, scales)
```

Cached Dispatch 也不改变 shape，只是复用上一次生成的路由元数据
## 4. DeepEP 为什么难做

DeepEP 主要面对四个问题：

- **通信量动态**：Router 每轮结果不同，接收数量和 buffer shape 无法完全固定。
- **负载不均**：热门专家会使部分 rank 同时成为通信和计算瓶颈。
- **链路不同**：节点内使用 NVLink/P2P，节点间使用 RDMA，需要分别优化。
- **资源竞争**：通信内核也占用 GPU SM，可能挤压专家计算。

因此，优化目标不只是提高通信带宽，还要：

```
降低启动与同步开销
+ 减少通信占用的 SM
+ 尽量重叠通信和专家计算
```

---

## 5. DeepEP 的关键机制

|机制|作用|
|---|---|
|GPU 发起通信|V2 基于 NCCL GIN，减少 CPU 同步和通信启动开销|
|对称内存与分层传输|节点内走 NVLink/P2P，节点间走 RDMA|
|`EPHandle`|保存 Dispatch 的布局和反向映射，供 Combine 直接复用|
|异步 Event|管理 stream 依赖，支持通信与计算重叠|
|FP8 Dispatch|压缩激活通信量；Combine 通常更注重输出精度|
|JIT 与资源配置|针对 shape、EP size 和硬件配置通信内核、SM、QP 与 buffer|

由于篇幅原因，不再展开，可以自行结合仓库源码学习，我们把 DeepEP 的设计概括为：

> 让 GPU 直接、异步地完成 MoE token 通信，并用尽可能少的计算资源获得较高带宽和较低延迟。
## 6. 常见误区

### 误区一：DeepEP 传输的是整个请求

不是。它传输的是当前 MoE 层需要送往 experts 的 activation assignments 及相关元数据。

### 误区二：DeepEP 会切分 KV Cache

不会。KV Cache 属于 Attention 路径；EP 通信不等于 KV Cache 并行。

### 误区三：Dispatch 后的 `[R_r,H]` 是最终输出

不是。它只是当前 expert rank 收到的 assignment 输入。

### 误区四：本地 expert 输出就是最终 token 输出

不是。它仍是每个 assignment 的结果，还要 Combine、恢复 `[T,K,H]` 并沿 `K` 加权求和。

### 误区五：DeepEP 自动解决 expert 热点

DeepEP 优化通信执行；logical 路由、expert placement、冗余副本和 EPLB 仍由模型或上层框架负责。

### 误区六：EP size 越大一定越快

EP 增大后：

- 每卡 expert 权重减少；
    
- 每卡本地 expert 数减少；
    
- 单 expert GEMM 可能变得更小；
    
- A2A 通信范围扩大；
    
- 跨节点比例上升；
    
- 最慢 rank 长尾更明显。
    

因此 EP size 是显存、计算粒度和通信成本之间的权衡。

### 误区七：异步接口就等于通信被隐藏

只有时间线上存在独立计算，并且 stream/event 依赖正确时，通信才会真正被隐藏。

### 误区八：把 V1 模式名称直接解释成 V2 内部实现

框架可能保留 `normal`、`low_latency` 等名称，但 DeepEP V2 已统一到 `ElasticBuffer` 和 NCCL Gin，应区分 workload 标签、框架包装层与底层实现版本。

---
