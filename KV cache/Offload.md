# 多级 KV Cache Offload 学习笔记

## 1. 为什么需要 KV Cache Offload

大模型推理包含两个阶段：

- **Prefill**：处理整个输入 Prompt，计算量大，并生成每个 token 的 KV Cache。
    
- **Decode**：逐 token 生成，反复读取已有 KV Cache，通常受显存容量和带宽限制。
    

单 token 的 KV Cache 大小近似为：

$$
2 \times 层数 \times KV\ Head数 \times Head维度 \times 数据类型字节数  
$$

其中 `2` 代表 Key 和 Value。

长上下文和高并发会迅速耗尽 GPU 显存。如果直接删除 KV，后续请求只能重新执行 Prefill。因此，可以把冷 KV 转移到更大、更慢的介质中，需要时再加载回来。

这就是 **KV Cache Offload**。

---

## 2. 多级 KV Cache 的基本结构

```
L1：GPU HBM/VRAM
    活跃请求、热点前缀
          │
          ▼
L2：CPU DRAM
    近期使用、等待复用的 KV
          │
          ▼
L3：NVMe / 分布式内存 / 分布式文件系统
    大容量、跨实例共享的 KV
          │
          ▼
L4：S3 等对象存储
    冷数据或持久数据
```

各层特点：

| 层级  | 介质               | 优点         | 缺点          |
| --- | ---------------- | ---------- | ----------- |
| L1  | GPU HBM          | 延迟最低、可直接计算 | 容量小、昂贵      |
| L2  | CPU DRAM         | 容量较大、延迟适中  | PCIe 传输有开销  |
| L3  | NVMe、远端 DRAM、3FS | 容量大、可跨节点共享 | I/O 和网络延迟较高 |
| L4  | S3               | 容量和持久性强    | 延迟最高        |

多级缓存的基本原则是：

$$
加载KV的成本 < 重新执行Prefill的成本  
$$

==如果加载比重算更慢，Offload 就没有收益。==

---

## 3. 两种不同的使用场景

### 3.1 跨请求 KV 复用

例如多个请求包含相同的 System Prompt：

```
请求1：公共System Prompt + 问题A
请求2：公共System Prompt + 问题B
请求3：公共System Prompt + 问题C
```

首次请求计算公共前缀的 KV，后续请求直接加载复用，只计算不同的问题部分。

适合：

- 公共 System Prompt。
    
- 多轮对话。
    
- Agent 工作流。
    
- RAG 中反复使用的文档。
    
- 相同模型上的重复长前缀。
    

这是 LMCache 和 HiCache 的主要应用。

### 3.2 Prefill-Decode 分离

```
Prefill Worker
    │ 计算Prompt并产生KV
    │
    │ KV传输
    ▼
Decode Worker
    │
    └── 开始逐token生成
```

这种场景主要需要高速 KV 传输，Mooncake Transfer Engine 和 NIXL 都可以作为底层传输方案。

它和跨请求缓存的区别是：

- PD 传输通常是一次性的请求交接。
    
- 多级缓存需要查找、复用、淘汰和长期管理。
    

---

## 4. 一次缓存请求的完整流程

```
1. 请求到达推理引擎
2. 对Prompt token进行分块或前缀匹配
3. 依次查找GPU、CPU和外部存储
4. 将命中的KV加载到GPU
5. 跳过命中部分的Prefill
6. 只计算未命中的token
7. 新KV异步写入CPU或外部存储
8. 上层空间不足时淘汰冷KV
```

对应的数据流：

```
                         缓存命中
外部存储 ──预取──> CPU ──加载──> GPU
                                      │
Prompt ───────────────────────────────>│ 推理
                                      │
外部存储 <──异步写入── CPU <──Offload─┘
```

---

# 5. LMCache

LMCache 是相对独立的 **KV Cache 管理层**，通过 Connector 接入 vLLM、SGLang 等推理引擎。

```
vLLM / SGLang
       │
       ▼
    LMCache
       │
       ├── CPU DRAM
       ├── 本地NVMe
       ├── Mooncake
       ├── NIXL
       ├── Redis
       └── S3等存储
```

它的主要价值可以概括为三点：
第一，GPU 显存不足时，可以把冷 KV 异步转移到 CPU 或外部存储；
第二，新请求到来时，可以根据 token 分块和哈希查找已有 KV，跳过重复的 Prefill；
第三，它对上层推理引擎和下层存储相对解耦，可以连接 Mooncake、NIXL、Redis、S3 等不同后端。

因此，LMCache 更适合以 vLLM 为主、存在多种推理引擎，或者需要构建跨节点共享 KV Cache 服务的环境。除了普通前缀复用，LMCache 还可以通过 CacheBlend 复用 Prompt 中非前缀位置的文本，不过这种复用需要选择性重算，以修复上下文和位置变化产生的误差。

---

# 6. HiCache

HiCache 是 SGLang 原生的多级 KV Cache 系统，是 RadixAttention 的扩展。

```
SGLang RadixAttention
         │
         ▼
L1：GPU KV Pool
         │
         ▼
L2：Host Memory Pool
         │
         ▼
L3：Mooncake / NIXL / 3FS / File等
```

RadixAttention 使用 Radix Tree 组织 token 前缀。相同前缀可以指向相同 KV 节点。HiCache 进一步让这些节点在被 GPU 淘汰后，仍然保留在 CPU 或外部存储中。

HiCache 是 SGLang 原生的多级 KV Cache 机制，可以看作 RadixAttention 从 GPU 显存向 CPU 内存和外部存储的扩展。RadixAttention 使用 Radix Tree 组织 token 前缀，使相同前缀能够共享 KV；==HiCache 则让这些 KV 在被 GPU 淘汰后，仍然保留在 CPU 或外部存储中。==

HiCache 通常分为三级：**GPU KV Pool**保存活跃请求和热点前缀，**CPU Host Memory Pool** 保存从 GPU 降级的数据，**Mooncake、NIXL、3FS 或文件系统**等外部后端提供容量更大的共享存储。请求到达后，SGLang 会在各层查找匹配前缀，并把命中的 KV 预取回 GPU。

### 写入策略

|   |   |   |
|---|---|---|
|策略|行为|特点|
|`write_through`|新 KV 立即写入下层|命中机会高，I/O 压力大|
|`write_through_selective`|热度达到阈值后写入|只保留热点数据|
|`write_back`|上层淘汰时才写入|写流量小，淘汰路径更重|

### L3 预取策略

|   |   |
|---|---|
|策略|行为|
|`best_effort`|不阻塞 GPU，能取多少取多少|
|`wait_complete`|等待命中数据全部加载|
|`timeout`|等待到完成或超时，平衡延迟和命中率|

### 适用场景

- 推理引擎统一为 SGLang。
    
- 主要复用公共前缀和多轮对话。
    
- 希望缓存和 RadixAttention、调度器原生协同。
    
- 使用 SGLang 的 PD 分离和分层存储能力。
    

---

# 7. Mooncake

Mooncake 是一套面向大模型的数据传输与分布式缓存基础设施，主要包含：

- **Mooncake Transfer Engine**：负责高速搬运数据。
    
- **Mooncake Store**：负责存储、定位、复制和淘汰对象。
    

在多级 KV Cache 中，它主要承担两种角色：

```
角色1：传输层
Prefill GPU ──Mooncake TE──> Decode GPU

角色2：远端缓存池
LMCache / HiCache ──> Mooncake Store
```

主要特点：

- 支持 RDMA、TCP、NVMe-oF 等路径。
    
- 支持 GPU、CPU 和 NVMe 之间的数据传输。
    
- 支持多网卡并行和拓扑感知。
    
- Mooncake Store 可以提供分布式 DRAM/SSD KV Cache 池。
    

==Mooncake 不负责理解 Prompt 语义或 Radix 前缀；缓存键、命中规则和复用策略仍由 LMCache、HiCache或推理引擎决定。==

---

# 8. NIXL

NIXL，即 NVIDIA Inference Xfer Library，是统一的异构数据传输抽象层。

```
LMCache / HiCache / vLLM / Dynamo
                 │
                 ▼
               NIXL
        ┌────────┼─────────┐
        ▼        ▼         ▼
       UCX      GDS       其他插件
        │        │
      RDMA    NVMe→GPU
```

主要作用：

- 统一描述 CPU、GPU、文件、块设备和对象存储。
    
- 注册内存并管理远端访问元数据。
    
- 通过插件选择 UCX、GDS、POSIX、3FS、S3、Mooncake 等路径。
    
- 提供异步点对点传输接口。
    

NIXL 更接近“搬运库”，通常不负责：

- Prefix Cache 命中。
    
- KV 淘汰策略。
    
- 缓存副本管理。
    
- Token 哈希。
    
- 全局缓存容量分配。
    

当前 NIXL 也可以将 Mooncake Transfer Engine 作为插件，因此两者不一定是互斥关系。

---

# 9. 四者的关系

```
                    推理引擎
                  vLLM / SGLang
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
         LMCache                   HiCache
      独立缓存管理层          SGLang原生分层缓存
            │                         │
            └────────────┬────────────┘
                         ▼
             Mooncake Store / 3FS
                 分布式存储层
                         │
                         ▼
           NIXL / Mooncake Transfer Engine
                    数据传输层
                         │
                         ▼
           UCX / RDMA / GDS / TCP / NVMe
```

核心定位：

|   |   |   |
|---|---|---|
|项目|主要定位|是否管理缓存语义|
|LMCache|跨引擎 KV Cache 管理层|是|
|HiCache|SGLang 原生分层缓存|是|
|Mooncake Store|分布式对象与 KV 存储池|部分，主要管理对象|
|Mooncake TE|高性能数据传输引擎|否|
|NIXL|可插拔异构数据传输抽象|否|

LMCache 与 HiCache 通常是替代关系。

Mooncake 和 NIXL 可以：

- 直接作为传输方案进行比较。
    
- 分别被 LMCache、HiCache 使用。
    
- 组合使用，例如 NIXL 通过 Mooncake 插件调用 Mooncake TE。
    

---