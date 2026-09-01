# 多级 KV Cache Offload与管理 学习笔记

## 1. 为什么需要 KV Cache Offload

大模型推理分为两个阶段：

- **Prefill**：一次性处理输入 Prompt，计算量较大，同时为每个 token 生成 KV Cache。
- **Decode**：逐个生成新 token，并不断读取已有 KV Cache，通常更受显存容量和带宽限制。

单个 token 的 KV Cache 大小约为：

$$
2 \times \text{层数} \times \text{KV Head 数} \times \text{Head 维度} \times \text{数据类型字节数}
$$

其中，`2` 代表 Key 和 Value。

上下文越长、并发请求越多，KV Cache 占用的 GPU 显存就越大。显存不足时，如果直接删除 KV，后续再次使用相同内容就必须重新执行 Prefill。

**KV Cache Offload** 的思路是：把暂时不用的 KV 从 GPU 转移到容量更大、速度较慢的 CPU 内存或外部存储，需要时再加载回来，从而用较低的加载成本换取显存空间和重复计算的减少。

判断 Offload 是否值得的核心条件是：

$$
\text{加载 KV 的成本} < \text{重新执行 Prefill 的成本}
$$

## 2. 多级 KV Cache 的基本结构

```text
L1：GPU HBM / VRAM
    活跃请求、热点前缀
          ↓
L2：CPU DRAM
    近期使用、等待复用的 KV
          ↓
L3：NVMe / 远端内存 / 分布式文件系统
    大容量、可跨节点共享的 KV
          ↓
L4：S3 等对象存储
    冷数据或需要持久保存的数据
```

| 层级  | 常见介质             | 优点           | 缺点              |
| --- | ---------------- | ------------ | --------------- |
| L1  | GPU HBM          | 延迟最低，可直接参与计算 | 容量小、成本高         |
| L2  | CPU DRAM         | 容量较大，访问速度较快  | 需要经过 PCIe 等链路传输 |
| L3  | NVMe、远端 DRAM、3FS | 容量大，可跨节点共享   | I/O 或网络延迟较高     |
| L4  | S3 等对象存储         | 容量大、持久性强     | 延迟最高，不适合频繁访问    |

多级缓存的目标不是**把所有 KV 都长期保存**，而是根据访问热度在==容量、速度和成本==之间取得平衡。

## 3. 两种不同的使用场景

### 3.1 跨请求 KV 复用

多个请求可能包含相同的长前缀：

```text
请求 1：公共 System Prompt + 问题 A
请求 2：公共 System Prompt + 问题 B
请求 3：公共 System Prompt + 问题 C
```

第一次请求计算并缓存公共前缀的 KV，后续请求直接复用，只计算各自不同的部分。

常见场景包括：

- 公共 System Prompt；
- 多轮对话；
- Agent 工作流；
- RAG 中反复使用的文档；
- 同一模型上的重复长前缀。

LMCache 和 HiCache 主要解决这类==缓存查找、复用和淘汰==问题。

### 3.2 Prefill-Decode 分离

```text
Prefill Worker
    │ 计算 Prompt，生成 KV
    ↓
高速传输 KV
    ↓
Decode Worker
    └─ 逐 token 生成
```

这种架构把计算密集的 Prefill 和访存密集的 Decode 分配给不同节点，重点是快速、稳定地传输 KV。Mooncake Transfer Engine 和 NIXL 都可以提供底层传输能力。

两种场景的区别是：

- **PD 分离**关注单次请求在不同 Worker 之间的交接；
- **跨请求复用**关注 KV 的查找、保存、复用和淘汰等长期管理。

## 4. 一次缓存请求的完整流程

一次请求通常按以下步骤执行：

1. 请求到达推理引擎。
2. 系统对 Prompt 进行分块或前缀匹配。
3. 按 ==GPU、CPU、外部存储==的顺序查找已有 KV。
4. 将命中的 KV 加载或预取到 GPU。
5. 跳过命中部分的 Prefill，只计算未命中的 token。
6. 使用完整 KV 继续 Decode。
7. 将新生成的 KV **异步**写入 CPU 或外部存储。
8. 空间不足时，把**低热度** KV 淘汰或降级到下一层。

```text
外部存储 ──预取──> CPU ──加载──> GPU
                                      │
Prompt ───────────────────────────────>│ 推理
                                      │
外部存储 <──异步写入── CPU <──Offload─┘
```

为了避免数据传输阻塞计算，实际系统通常会使用==异步写入、提前预取和批量传输。==

## 5. LMCache

LMCache 是一个相对独立的 **KV Cache 管理层**，可通过 Connector 接入 vLLM、SGLang 等推理引擎。

```text
vLLM / SGLang
       ↓
    LMCache
       ├─ CPU DRAM
       ├─ 本地 NVMe
       ├─ Mooncake
       ├─ NIXL
       ├─ Redis
       └─ S3 等后端
```

它主要负责三件事：

1. **Offload**：把暂时不用的 KV 异步转移到 CPU 或外部存储。
2. **复用**：根据 token 分块和哈希查找已有 KV，跳过重复 Prefill。
3. **解耦**：向上连接推理引擎，向下连接不同的存储和传输后端。

LMCache 适合==以 vLLM 为主、同时存在多种推理引擎，或需要构建跨节点共享 KV Cache 服务的环境。==

除普通前缀复用外，LMCache 的 CacheBlend 还能复用 Prompt 中非前缀位置的文本，但需要选择性重算，以减小上下文和位置变化带来的误差。

## 6. HiCache

HiCache 是 SGLang 原生的多级 KV Cache 系统，可以看作 RadixAttention 从 GPU 向 CPU 和外部存储的扩展。

```text
SGLang RadixAttention
         ↓
L1：GPU KV Pool
         ↓
L2：Host Memory Pool
         ↓
L3：Mooncake / NIXL / 3FS / 文件系统
```

RadixAttention 使用 Radix Tree 组织 token 前缀，让相同前缀共享 KV。HiCache 则让这些 KV 被 GPU 淘汰后，仍可保存在 CPU 或外部存储中，并在后续命中时预取回 GPU。

常见写入策略：

| 策略 | 行为 | 特点 |
| --- | --- | --- |
| `write_through` | 新 KV 立即写入下层 | 命中机会高，但写入压力大 |
| `write_through_selective` | 达到热度阈值后再写入 | 只保留更可能复用的数据 |
| `write_back` | 上层淘汰时才写入 | 写流量小，但淘汰过程更重 |

常见 L3 预取策略：

| 策略 | 行为 |
| --- | --- |
| `best_effort` | 不阻塞 GPU，能取多少就使用多少 |
| `wait_complete` | 等待全部命中数据加载完成 |
| `timeout` | 等待完成或超时，在延迟和命中率之间折中 |

HiCache 更适合统一使用 SGLang，并希望缓存与 RadixAttention、调度器及 PD 分离能力原生协同的场景。

## 7. Mooncake

Mooncake 是面向大模型的数据传输与分布式缓存基础设施，主要包含两个部分：

- **Mooncake Transfer Engine**：负责高速搬运数据。
- **Mooncake Store**：负责对象的存储、定位、复制和淘汰。

在多级 KV Cache 中，Mooncake 可以承担两种角色：

```text
角色 1：传输层
Prefill GPU ──Mooncake TE──> Decode GPU

角色 2：远端缓存池
LMCache / HiCache ──> Mooncake Store
```

它==支持 GPU、CPU、NVMe 之间的数据传输，并可使用 RDMA、TCP、NVMe-oF 等路径。==Mooncake Store 还可以把多台机器的内存和 SSD 组织成共享缓存池。

需要注意：Mooncake 不负责理解 Prompt，也不决定前缀如何匹配。缓存键、命中规则和复用策略仍由 LMCache、HiCache 或推理引擎负责。

## 8. NIXL

NIXL（NVIDIA Inference Xfer Library）是统一的异构数据传输抽象层。

```text
LMCache / HiCache / vLLM / Dynamo
                 ↓
               NIXL
        ┌────────┼────────┐
        ↓        ↓        ↓
       UCX      GDS     其他插件
        │        │
      RDMA    NVMe → GPU
```

它主要负责：

- 统一描述 CPU、GPU、文件和块设备等不同存储位置；
- 注册内存并管理远端访问所需的元数据；
- 通过插件选择 UCX、GDS、POSIX、3FS、S3、Mooncake 等传输路径；
- 提供异步、点对点的数据传输接口。

NIXL 更像一套通用“搬运接口”，通常不负责前缀命中、token 哈希、KV 淘汰、副本策略和全局容量分配。

NIXL 可以把 Mooncake Transfer Engine 作为插件，因此两者既可以分别使用，也可以组合使用。

## 9. 四者的关系

```text
                     推理引擎
                  vLLM / SGLang
                         │
            ┌────────────┴────────────┐
            ↓                         ↓
         LMCache                   HiCache
      独立缓存管理层          SGLang 原生分层缓存
            │                         │
            └────────────┬────────────┘
                         ↓
              Mooncake Store / 3FS
                 分布式存储层
                         ↓
           NIXL / Mooncake Transfer Engine
                    数据传输层
                         ↓
           UCX / RDMA / GDS / TCP / NVMe
```

| 项目 | 主要定位 | 是否管理缓存语义 |
| --- | --- | --- |
| LMCache | 跨引擎 KV Cache 管理层 | 是 |
| HiCache | SGLang 原生分层缓存 | 是 |
| Mooncake Store | 分布式对象与 KV 存储池 | 部分，主要管理对象 |
| Mooncake Transfer Engine | 高性能数据传输引擎 | 否 |
| NIXL | 可插拔的异构数据传输抽象层 | 否 |

可以这样记忆：

- **LMCache 与 HiCache 决定“缓存什么、何时命中、何时淘汰”**，通常按推理引擎和架构需求二选一。
- **Mooncake Store 和 3FS 解决“数据存在哪里”**。
- **NIXL 与 Mooncake Transfer Engine 解决“数据如何高效搬运”**。
- NIXL 和 Mooncake 可以作为不同传输方案，也可以通过插件组合使用。

