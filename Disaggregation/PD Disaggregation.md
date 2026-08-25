
> 口径：本文讨论大模型推理服务中的 Prefill/Decode 分离，简称 PD 分离或 P/D Disaggregation。示例参数以 SGLang 为主；具体传输 backend、拓扑和功能组合随版本变化。
## 1. 一句话结论

PD 分离把一次请求的两个阶段放到不同的 worker pool：

```text
Prefill workers：处理 Prompt，生成 KV Cache
Decode workers： 接收 KV Cache，持续生成输出 token
```

它不是按 `B/S/H` 切 Tensor，而是按**推理阶段和服务角色**切系统：

```text
请求 ──> P节点：Prefill ──传输KV──> D节点：Decode ──> 输出tokens
```

主要目标：

- 消除长 Prefill 对 Decode 流式生成的干扰；
- 分别优化 TTFT 和 TPOT/ITL；
- Prefill 与 Decode 独立扩缩容；
- 为两个阶段选择不同 batch、kernel、并行策略和硬件；
- 让 MoE/DeepEP 等组件分别使用吞吐模式与低延迟模式。

主要代价：

- 必须跨实例传输 KV Cache；
- TTFT 新增握手、D 侧预分配和传输开销；
- P、D 两套 worker 都要加载模型权重；
- 路由、缓存一致性、容错和容量规划更复杂。

---

## 2. 为什么 Prefill 和 Decode 适合分开

### 2.1 Prefill 的计算特征

输入长度为 `S` 时，Prefill 一次处理整个 Prompt：

```text
输入 Query 数：S
Dense Attention主要计算：随S²增长
典型瓶颈：计算、Tensor Core、长序列Attention
关键指标：TTFT、input tokens/s
```

Prefill 往往适合：

- 大 token batch；
- 高吞吐 kernel；
- PCP、较大 TP/EP；
- 较大的 chunked-prefill size；
- DeepEP normal/high-throughput dispatch；
- 对 CUDA Graph 的依赖相对较低。

### 2.2 Decode 的计算特征

Decode 每一步通常只产生一个或少量新 token，却要读取完整历史 KV：

```text
新 Query 数：通常1
历史 KV 长度：S_kv，逐步增长
典型瓶颈：KV容量、HBM带宽、collective延迟
关键指标：TPOT/ITL、output tokens/s、最大并发
```

Decode 往往适合：

- 大并发 request batch；
- 低延迟 kernel 与 CUDA Graph；
- DPA、DCP、MTP/投机解码；
- FP8/量化 KV Cache；
- DeepEP low-latency dispatch；
- 稳定、低抖动的调度。

两个阶段的最佳配置明显不同，因此让同一 worker 同时兼顾二者，经常不是最优。

---

## 3. 合并部署的问题

传统 colocated engine 中：

```text
同一批GPU
├─ 接收新的Prefill请求
└─ 维护正在生成的Decode请求
```

主要问题：

### 3.1 Prefill 打断 Decode

```text
长Prefill进入GPU
      ↓
Decode请求等待下一次调度
      ↓
ITL出现尖峰，P95/P99 TPOT变差
```

Chunked Prefill 可以缩短单次阻塞，但不能彻底消除阶段间的资源竞争。

### 3.2 DP Attention 负载不均

不同 Attention DP ranks 可能处于不同模式：

```text
rank 0：长Prefill
rank 1：Decode
rank 2：idle
```

进入共享 MoE/EP 同步点时，较快的 Decode rank 可能等待较慢的 Prefill rank。

### 3.3 两套 kernel/通信模式冲突

以 DeepEP 为例：

```text
Prefill：倾向normal/high-throughput dispatch
Decode： 倾向low-latency dispatch + CUDA Graph
```

同一通信组难以同时针对两种 workload 选择最优模式。PD 分离后，P pool 和 D pool 可以固定采用各自适合的配置。

---

## 4. 系统架构

典型 PD 系统包含四部分：

```text
                         ┌────────────────────┐
Client ──> Router/Gateway│ 请求路由、负载均衡  │
                         └──────┬───────┬─────┘
                                │       │
                     选择P worker│       │选择D worker
                                ▼       ▼
                       ┌───────────┐  ┌───────────┐
                       │ Prefill P │  │ Decode D  │
                       │ 计算Prompt│  │ 预留KV空间 │
                       └─────┬─────┘  └─────▲─────┘
                             │              │
                             └── KV传输 ────┘
                                RDMA/NVLink
                                Mooncake/NIXL
```

### 4.1 Router/Gateway

负责：

- 选择 Prefill worker；
- 选择 Decode worker；
- 协调 P/D 配对与 bootstrap 信息；
- 使用 cache-aware、power-of-two 等策略做负载均衡；
- 汇合 Prefill metadata 与 Decode 流式输出；
- 健康检查、失败重试、超时和背压。

### 4.2 Prefill pool

- 接收 Prompt token；
- 运行所有 Transformer 层的 Prefill；
- 生成每层 KV Cache；
- 将 KV 数据和必要 metadata 发送给 D worker；
- 完成交接后释放该请求的临时状态。

### 4.3 Decode pool

- 在 Prefill 开始前或传输前预留 KV block；
- 接收 P worker 生成的 KV；
- 将请求加入 Decode running batch；
- 迭代生成 token，直到完成或取消；
- 长期持有该请求的 KV Cache。

### 4.4 KV transfer backend

负责高带宽、低 CPU 开销的数据搬运。SGLang 当前常见选项包括：

- Mooncake；
- NIXL；
- Ascend 专用 backend；
- 测试用 fake backend。

生产环境通常优先使用 GPU Direct/RDMA，避免经 CPU staging 带来的额外复制。

---

## 5. 一条请求的完整生命周期

```text
1. Client把请求发给Gateway
2. Gateway选择一个P worker和一个D worker
3. P与D建立bootstrap/传输连接
4. D根据Prompt长度和预留输出长度分配KV blocks
5. D通知P可以开始或继续Prefill
6. P执行Prompt forward，生成各层KV
7. P通过传输backend把KV和metadata发送给D
8. D确认KV就绪，将请求加入Decode batch
9. D生成首token及后续tokens
10. 请求完成，D释放KV；Gateway结束stream
```

不同实现可能：

- Prefill 完成后一次性传输；
- 按层、按 chunk 或 staging buffer 分批传输；
- 让传输线程与 Prefill 计算重叠；
- 在 P 或 D 侧执行首 token sampling。

因此首 token 究竟在哪侧产生属于实现细节；不变的是：Decode 必须在所需 KV 到达 D 侧后才能正确继续生成。

---

## 6. KV Cache 传输量

标准 Attention 中，一条请求的 KV 体积近似为：

```text
KV_bytes ≈
2 × L × S_prompt × Nkv × Dh × bytes_per_element
```

其中：

```text
2   = K + V
L   = 层数
Nkv = KV head数
Dh  = head_dim
```

例如：

```text
L=32
S=32768
Nkv=8
Dh=128
BF16/FP16=2 bytes

KV ≈ 4 GiB / request
```

这 4 GiB 必须从 P 侧交给 D 侧。若网络有效带宽为 `BW`：

```text
纯传输时间下界 ≈ KV_bytes / BW
```

例如 50 GB/s 的有效带宽，4 GiB 的理想下界约为 80 ms，尚未计算连接、metadata、非连续内存和并发竞争。

MLA 使用压缩 KV latent，传输量通常显著小于普通 GQA/MHA；KV FP8 也可近似把相对 BF16 的数据量减半，但必须保证 P/D 两侧 dtype、布局和 kernel 兼容。

### 6.1 并行布局对传输的影响

如果 P/D 都使用 TP/PP，每个 rank 通常只传自己持有的 KV shard：

```text
P rank i 的KV shard -> 对应的D rank或重排后的目标rank
```

所有 ranks 的总传输量仍约等于完整请求 KV。若 P 与 D 的 TP、DPA、DCP 或 PP 布局不同，还需要 runtime 支持 KV layout mapping；不能假设任意拓扑都能直接互传。

---

## 7. 对性能指标的影响

箭头表示典型方向：

```text
↓ / ↑        = 主要影响
可能↓ / 可能↑ = 依赖负载、网络与调度
→            = 没有直接改变
```

| 指标 | 典型变化 | 原因 |
|---|---|---|
| TTFT | 可能↓或可能↑ | P队列/计算可独立扩展，但新增握手、D预分配和KV传输 |
| P95/P99 TTFT | 可能↓ | P pool与独立路由可减少混部排队长尾 |
| TPOT/ITL | **↓** | Decode不再被新Prefill打断，可固定使用低延迟模式 |
| P95/P99 ITL | **↓** | 阶段干扰和DP Attention模式不均衡减少 |
| 总吞吐 | **↑** | P/D分别使用适合的batch、kernel和并行策略 |
| P侧长期KV占用 | **↓** | P只暂存并发送KV，不承担持续Decode |
| D侧单请求KV | → | D仍需保存该请求完整或DCP分片后的Decode KV |
| 模型权重总占用 | **↑** | P pool和D pool都需要加载模型权重 |

### 7.1 TTFT 的新构成

PD 模式下：

```text
TTFT =
Gateway/Prefill Queue
+ P/D配对与握手
+ D侧KV预分配
+ Prefill计算
+ 暴露在关键路径上的KV传输
+ Decode入队
+ 首次Decode/sampling
+ 首token发送
```

若计算与传输充分重叠：

```text
暴露传输时间 < KV总传输时间
```

低负载、短 Prompt 时，新增握手和 KV 传输可能让 TTFT 上升；高负载或长 Prompt 时，独立 P pool、PCP 和更好的 Prefill batching 可能让 TTFT 下降。

### 7.2 TPOT/ITL

PD 分离后：

```text
TPOT ≈
D侧Decode调度等待
+ Decode Attention/KV读取
+ D侧TP/DCP/DPA/EP通信
+ sampling与stream
```

不再包含 colocated worker 上长 Prefill 带来的直接阻塞，因此 TPOT，尤其 P95/P99 ITL，通常是 PD 分离最稳定的延迟收益。

### 7.3 吞吐

端到端能力由最慢阶段决定：

```text
QPS_max = min(
  Prefill pool能力,
  KV传输能力,
  Decode pool能力,
  Gateway/调度能力
)
```

只扩 P 或只扩 D 都可能没有端到端收益；必须找到当前瓶颈。

---

## 8. P/D 容量配比

设：

```text
λ        = 请求到达率（req/s）
E[Tp]    = 一个请求占用P worker的平均服务时间
E[Td]    = 一个请求占用D worker的平均驻留/服务时间
ρ_target = 目标利用率，例如0.7～0.85
```

粗略容量条件：

```text
P容量：N_P >= λ × E[Tp] / ρ_target
D容量：N_D >= λ × E[Td] / ρ_target
```

也可以先用 token 速率估算：

```text
Prefill输入需求 ≈ QPS × 平均ISL
Decode输出需求  ≈ QPS × 平均OSL
```

但这只是第一步，因为：

- Prefill Attention 计算并不严格与 ISL 线性；
- Decode 每 token 成本会随上下文增长；
- MTP 一次 step 可产生多个 token；
- prefix cache 命中会降低 Prefill 计算和传输；
- batch size 会改变每 token 效率。

经验方向：

```text
长ISL、短OSL -> 增加P容量
短ISL、长OSL -> 增加D容量
长ISL、长OSL -> P、网络、D都可能成为瓶颈
```

Decode 请求驻留时间通常远长于 Prefill 服务时间，所以不能简单按 `P:D=1:1` 配置。

---

## 9. 与其他并行策略的关系

PD 分离是阶段级部署架构；PCP/DCP/DPA/TP/PP/EP 是阶段内部并行方式。

```text
P pool：TP + PP + PCP + EP
D pool：TP + PP + DPA/DCP + EP + MTP
```

### 9.1 PD 与 PCP

```text
PD： 把Prefill放到独立P workers
PCP：在一个Prefill请求内部沿S_q切分
```

二者可以组合。PCP 降低长 Prompt 计算时间，PD 防止这些 Prefill 计算影响 Decode。

### 9.2 PD 与 DCP

```text
PD： 把Decode放到独立D workers
DCP：在D workers内部沿S_kv切历史KV
```

DCP 可降低 D 侧单 rank KV 占用，但 P 传来的 KV 必须转换/写入正确的 DCP shard。

### 9.3 PD 与 DP Attention

DPA 让不同 Attention ranks 处理不同请求。合并部署时，某 rank Prefill、某 rank Decode 容易在 MoE 同步点互相等待；PD 分离后：

```text
P侧DPA ranks：只处理Prefill/extend
D侧DPA ranks：只处理Decode
```

负载模式更统一，kernel、CUDA Graph 和 DeepEP dispatch 更容易优化。

### 9.4 PD 与 EP

P 侧可使用面向吞吐的 EP dispatch，D 侧使用面向低延迟的 EP dispatch。两套 pool 仍各自加载相应模型/专家权重 shard。

---

## 10. SGLang 参数怎样读

### 10.1 Prefill worker

```bash
python -m sglang.launch_server \
  --model-path MODEL_PATH \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-bootstrap-port 8998 \
  --port 30000
```

### 10.2 Decode worker

```bash
python -m sglang.launch_server \
  --model-path MODEL_PATH \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --port 30001
```

### 10.3 Gateway

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://PREFILL_HOST:30000 8998 \
  --decode http://DECODE_HOST:30001 \
  --host 0.0.0.0 \
  --port 8000
```

参数含义：

| 参数 | 含义 |
|---|---|
| `--disaggregation-mode prefill` | 启动只负责 Prefill 的 worker |
| `--disaggregation-mode decode` | 启动只负责 Decode 的 worker |
| `--disaggregation-transfer-backend` | KV 传输 backend，常用 Mooncake/NIXL |
| `--disaggregation-bootstrap-port` | P/D 建立传输关系的 bootstrap 端口 |
| `--disaggregation-ib-device` | 指定 KV 传输使用的 InfiniBand/RoCE 设备 |
| Router `--pd-disaggregation` | 让 Gateway 按 PD 流程组合 P/D workers |

生产配置通常还会分别设置 TP、DP Attention、EP、KV dtype、CUDA Graph、max-running-requests 和显存比例。

---

## 11. 什么时候适合 PD 分离

### 适合

- 高并发在线服务，既有 Prefill 又有持续 Decode；
- 长 Prompt 经常导致 P95/P99 ITL 抖动；
- Prefill 与 Decode 的最佳 GPU 配比明显不同；
- 使用 DPA/大规模 EP，需要区分 Prefill 与 Decode 通信模式；
- 希望独立扩缩 P、D pool；
- KV 传输网络足够快，并可使用 RDMA/GPU Direct；
- TTFT 与 TPOT 有不同 SLO，需要分别治理。

### 不一定适合

- 单机、低 QPS、请求很短；
- KV Cache 很大但网络带宽不足；
- 模型较小，合并部署已能稳定满足 SLO；
- P/D 两套权重副本造成的 GPU 成本不可接受；
- 业务 prefix cache 命中率很高，但路由不能保持 cache affinity；
- 传输 backend 不支持目标模型的 KV/state 布局；
- 运维系统无法处理 P/D 独立故障、重试和背压。

---

## 12. 常见失败模式

### 12.1 D pool 预分配失败

D 侧没有足够 KV block，即使 P 侧有空闲算力，请求也不能安全开始交接。需要：

- admission control；
- 为未来 Decode token 预留空间；
- D pool 背压传递给 Gateway/P pool；
- 控制 max-running-requests 与 max-total-tokens。

### 12.2 KV 传输成为瓶颈

症状：

```text
Prefill forward很快
但TTFT随ISL近似按传输量增长
IB/RDMA链路接近饱和
```

优化方向：KV FP8、增加 NIC/rail、拓扑亲和、传输计算重叠、减少不必要的 cache miss。

### 12.3 P/D 数量比例错误

- P 太少：Prefill Queue、TTFT 上升，D 吃不饱；
- D 太少：Decode admission 排队、KV 满、TPOT/吞吐变差；
- P 太多：KV 瞬时涌入 D，形成 transfer/admission burst；
- D 太多：Decode batch 太小，GPU 利用率下降。

### 12.4 Cache locality 被破坏

共享 prefix 可能只存在于某些 P/D workers。纯随机路由会降低 cache hit，导致重新 Prefill 和重复 KV 传输。应考虑 cache-aware routing 和缓存状态协调。

### 12.5 P/D 配置或布局不兼容

- 模型版本、RoPE 配置不一致；
- KV dtype 不一致；
- TP/PP/DPA/DCP rank 映射不受支持；
- block/page size 不兼容；
- 混合 Attention/SSM 模型还有额外状态需要传输。

### 12.6 故障后的重复计算

P 完成但 D 失败，或 KV 传输中断时，请求可能必须重新 Prefill。生产系统需要幂等 request ID、超时、重试策略和失败清理。

---

## 13. 高频误区

1. **“PD 分离就是 PCP+DCP。”** 不是。PD 决定阶段放在哪些 worker；PCP/DCP 决定阶段内部如何多卡切序列。
2. **“PD 分离一定降低 TTFT。”** 不一定。它新增握手和 KV 传输，短 Prompt/低负载时可能更慢。
3. **“P 完成后只需把最后一层输出传给 D。”** 错。标准自回归 Decode 需要所有 Attention 层的历史 KV Cache。
4. **“只要增加 P 节点，系统吞吐就会提高。”** 错。端到端吞吐由 P、网络、D 中最慢的一段决定。
5. **“KV 传输完成后 P/D 都长期保留一份。”** 正常流程中 P 应释放临时请求状态，D 长期持有；交接窗口内可能有短暂双份。
6. **“D pool 越大越好。”** D 过多会让每卡 batch 过小，降低 kernel 利用率并增加成本。
7. **“同一个模型就一定能在任意 P/D 拓扑间传 KV。”** 错。KV dtype、block layout、并行 rank 映射和 backend 都必须兼容。

---

## 14. 自测题

**Q1：为什么 PD 分离通常更稳定地改善 TPOT，而不一定改善 TTFT？**  
A：Decode 不再被 Prefill 打断，所以 ITL 更稳定；但 TTFT 新增 D 预分配、握手和 KV 传输。

**Q2：P pool 很空闲，但 D pool KV 满，继续加 P 有用吗？**  
A：没有。瓶颈在 D admission/KV 容量，应扩 D、优化 KV 或降低请求驻留量。

**Q3：一条 Prompt 的 KV 为什么必须传给 D？**  
A：后续每个 Decode Query 都要关注所有历史 K/V；只有最后 hidden state 不足以继续标准自回归 Attention。

**Q4：PCP 和 PD 如何配合？**  
A：PD 将 Prefill 放到专用 P pool，PCP 在 P pool 内把单条长 Prompt 沿序列分给多卡。

**Q5：DPA 和 PD 如何配合？**  
A：P、D 两侧 DPA ranks 分别只处理 Prefill 或 Decode，避免混合模式在共享 MoE 同步点互相拖累。

**Q6：怎样判断 KV 网络是否成为瓶颈？**  
A：Prefill forward 已结束但 TTFT 仍随 KV 字节数增长，且 transfer tail/NIC 利用率高；端到端 QPS 受传输带宽而非 P/D GPU 限制。

---

## 15. 参考资料

- [SGLang：PD Disaggregation 文档](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md)
- [SGLang：PD 参数说明](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)
- [SGLang Model Gateway：PD 路由](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/README.md)
- [LMSYS：DeepSeek PD 分离与大规模 EP](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)

> 版本提示：Mooncake/NIXL、staging buffer、异构 TP、DPA/DCP 布局转换等路径仍在快速演进。部署前应以目标版本的参数、模型支持矩阵和实际 profiler 为准。
