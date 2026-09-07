# PD Disaggregation 原理

> 本文讨论大模型推理服务中的 Prefill/Decode 分离（PD 分离）。示例参数以 SGLang 为主，具体 backend、拓扑与功能组合以目标版本为准。

## 1. 核心结论

PD 分离将请求的两个阶段部署到不同 worker pool：

```text
请求 → P worker：Prefill → 传输 KV Cache → D worker：Decode → 输出 token
```

它是**按推理阶段拆服务**，不是沿张量维度切分。核心收益是==隔离两类 workload，使其可独立调度、优化和扩缩容：==

- P 侧面向计算吞吐和 TTFT；
- D 侧面向显存带宽、并发与 TPOT/ITL；
- 避免长 Prefill 阻塞流式 Decode，改善尾部 ITL；
- P/D 可采用不同的 batch、kernel、并行策略和硬件配比。

代价是跨实例传输 KV、复制模型权重，并引入配对、预分配、路由、背压和故障恢复等复杂度。因此 PD 通常更稳定地改善 TPOT/ITL，但不保证降低 TTFT。

## 2. 为什么要拆分

Prefill 一次处理整个 Prompt，Attention 计算随序列长度快速增长，通常偏计算密集，适合大 token batch、高吞吐 kernel、PCP 和较大的 chunked-prefill size。

Decode 每步只生成少量 token，但持续读取完整历史 KV，通常受 KV 容量、HBM 带宽和 collective 延迟限制，适合大并发 request batch、CUDA Graph、DPA/DCP、MTP/投机解码及低延迟通信。

在合并部署中，长 Prefill 会占用 GPU 调度窗口，造成 Decode ITL 尖峰；DPA/EP 场景下，不同 rank 分别执行 Prefill、Decode 或空闲，还可能在共享同步点互相等待。Chunked Prefill 能缩短单次阻塞，但不能消除阶段间竞争。

PD 分离后，两侧可固定使用不同策略。例如 MoE 场景下：

```text
P pool：高吞吐 dispatch
D pool：低延迟 dispatch + CUDA Graph
```

## 3. 架构与请求流程

系统通常由 Gateway、P pool、D pool 和 KV transfer backend 组成：

```text
Client → Gateway ─┬→ P worker ── KV ──→ D worker → token stream
                  └────── 配对、路由、背压与容错 ──────┘
```

一次请求的关键路径如下：

1. Gateway 选择 P、D worker，并协调传输连接；
2. D 根据上下文和输出预算预留 KV blocks；
3. P 执行 Prompt forward，生成各层 KV；
4. KV 通过 RDMA/GPU Direct 等路径传到 D；
5. D 在 KV 就绪后加入 Decode batch，持续生成 token；
6. 请求结束后 D 释放 KV，P 在交接完成后释放临时状态。

KV **可整批传输，也可按层、chunk 或 staging buffer** 传输，并**与 Prefill 计算重叠**。首 token 在 P 侧还是 D 侧生成属于实现细节；D 必须拿到继续解码所需的全部状态。

Gateway 除负载均衡外，还需要处理 cache affinity、admission control、超时、重试和背压。常见 KV backend 包括 Mooncake、NIXL 及硬件专用实现。

## 4. KV 传输成本

标准 Attention 下，一条请求的 KV 大小近似为：

```text
KV_bytes ≈ 2 × L × S_prompt × Nkv × Dh × bytes_per_element
```

其中 `2` 表示 K+V，`L` 为层数，`Nkv` 为 KV head 数，`Dh` 为 head dimension。

例如 `L=32、S=32768、Nkv=8、Dh=128、BF16` 时，KV 约为 `4 GiB/request`。若有效带宽为 50 GB/s，理想传输下界约 80 ms，尚未包含握手、非连续内存和并发竞争。

MLA 的压缩 KV latent 可显著减小传输量；KV FP8 相对 BF16 通常可近似减半，但 P/D 两侧必须兼容 dtype、布局与 kernel。

P/D 使用 TP、PP、DPA 或 DCP 时，各 rank 通常只传自己的 KV shard，但总传输量仍接近完整请求 KV。若两侧并行布局不同，runtime 必须支持 rank 和 KV layout 映射，不能假设任意拓扑都可直接互传。

## 5. 性能与容量

### 延迟

PD 模式下：

```text
TTFT = P排队与计算
     + P/D配对、D侧预分配
     + 关键路径上的KV传输
     + D入队与首次Decode
```

==短 Prompt、低负载时，握手和 KV 传输可能使 TTFT 上升；高负载或长 Prompt 场景中，独立 P pool 和更好的 batching 也可能降低 TTFT。==

==Decode 不再被新 Prefill 直接打断，因此平均 TPOT 以及 P95/P99 ITL 通常更稳定。这往往是 PD 最确定的收益。==

### 吞吐

端到端能力取决于==最慢环节==：

```text
QPS_max = min(P pool, KV network, D pool, Gateway)
```

只扩 P 或只扩 D 未必有效。

```text
长 ISL、短 OSL → P 压力更大
短 ISL、长 OSL → D 压力更大
长 ISL、长 OSL → P、网络、D 均需评估
```

D 的请求驻留时间通常远长于 P 的服务时间，因此不能默认 `P:D = 1:1`。

## 6. 与其他并行策略的关系

PD 是阶段级部署架构；TP、PP、EP、PCP、DCP、DPA 是阶段内部并行方式，可组合使用：

```text
P pool：TP + PP + PCP + EP
D pool：TP + PP + DPA/DCP + EP + MTP
```

## 7. SGLang 最小配置

```bash
# Prefill worker
python -m sglang.launch_server \
  --model-path MODEL_PATH \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-bootstrap-port 8998 \
  --port 30000

# Decode worker
python -m sglang.launch_server \
  --model-path MODEL_PATH \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --port 30001

# Gateway
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://PREFILL_HOST:30000 8998 \
  --decode http://DECODE_HOST:30001 \
  --host 0.0.0.0 \
  --port 8000
```

生产环境还需分别设置 TP/DPA/EP、KV dtype、CUDA Graph、显存比例、`max-running-requests` 和 token 上限，并核对目标版本的 backend 与模型支持矩阵。

## 8. 适用条件与主要风险

PD 更适合以下场景：

- **高并发在线推理，长 Prefill** 明显影响 P95/P99 ITL；
- P/D 的最优 GPU 配比、batch 或通信模式差异较大；
- 需要分别治理 TTFT、TPOT，或独立扩缩 P/D；
- KV 网络有足够带宽，并支持 RDMA/GPU Direct。

低 QPS、短请求、网络较慢，或双份模型权重成本不可接受时，合并部署通常更简单。

上线前重点检查：

- **D 侧 KV 不足**：必须在 P 开始或交接前完成 admission，背压应传到 Gateway；
- **网络成为瓶颈**：表现为 Prefill 已结束，但 TTFT 仍随 KV 字节数增长且链路饱和；
- **P/D 配比错误**：==P 少则 TTFT 排队，D 少则 admission/KV 拥塞，池过大则 batch 变小或产生传输突发；==
- **缓存亲和性丢失**：随机路由降低 prefix cache 命中，导致重复 Prefill 和 KV 传输；
- **布局不兼容**：模型版本、RoPE、KV dtype、page size 及 TP/PP/DPA/DCP 映射必须一致或可转换；
- **交接失败**：P 完成后 D 故障可能触发重算，需要幂等 ID、超时、清理和重试策略。

## 9. 结论

PD 分离的本质是用额外的 KV 传输和系统复杂度，换取 Prefill/Decode 的资源隔离与独立优化。是否值得采用，取决于三个问题：

1. 合并部署中的 Prefill 是否已显著影响 Decode 尾延迟；
2. KV 网络能否承受峰值传输量；
3. P、D、网络三段能否按真实 ISL/OSL 分布实现容量平衡。

评估时应同时观察 TTFT、P95/P99 ITL、P/D 队列、D 侧 KV 使用率、传输尾延迟和 NIC 利用率，而不是只比较平均吞吐。

## 参考资料

- [SGLang：PD Disaggregation](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md)
- [SGLang：Server Arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)
- [SGLang Model Gateway](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/README.md)
- [LMSYS：DeepSeek PD 分离与大规模 EP](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
