## 1. Profiling 的目标

推理 profiling 不是单纯观察“GPU 利用率高不高”，而是回答：

1. 当前服务能达到多少吞吐？
    
2. 延迟由排队、数据传输还是 GPU 计算造成？
    
3. GPU 受计算、显存带宽、通信还是 CPU 限制？
    
4. 最佳并发数、批大小和模型实例数是多少？
    
5. 性能达到拐点后，应该优化参数还是增加 GPU？
    

DCGM 和 Prometheus 主要完成“持续观测与对比”。如果要定位到具体 CUDA Kernel，则需要进一步使用 Nsight Systems 或 Nsight Compute。

---

## 2. 分层理解推理指标

推理 profiling 最好分四层进行：

|层次|关注内容|数据来源|
|---|---|---|
|客户端层|QPS、TTFT、TPOT、P99 延迟、tokens/s|压测工具|
|推理服务层|排队时间、批大小、KV Cache、计算时间|vLLM、Triton、TGI 等|
|GPU 层|SM、Tensor Core、显存带宽、功耗、显存|DCGM Exporter|
|Kernel 层|CUDA Kernel、同步、Memcpy、算子耗时|Nsight|

最重要的原则是：

> 业务指标告诉我们“性能是否有问题”，DCGM 指标帮助判断“问题可能在哪个硬件方向”。

DCGM profiling 指标是一个采样区间内的平均值，适合低开销持续监控，但不是 Kernel Trace，无法直接指出哪个算子或代码行导致问题。[NVIDIA DCGM Profiling 文档](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)

---

## 3. 推理场景的核心指标

### 3.1 通用推理指标

- Throughput：每秒请求数
    
- Latency：平均、P50、P95、P99 延迟
    
- Queue Time：请求排队时间
    
- Compute Time：模型实际执行时间
    
- Batch Size：实际执行批大小
    
- Error Rate：超时和失败比例
    

### 3.2 LLM 推理指标

- TTFT：Time To First Token，首 Token 延迟
    
- TPOT：Time Per Output Token，平均每个输出 Token 时间
    
- ITL：Inter-Token Latency，相邻 Token 间隔
    
- Input tokens/s：输入 Token 吞吐
    
- Output tokens/s：输出 Token 吞吐
    
- KV Cache Usage：KV Cache 使用率
    
- Prefix Cache Hit Rate：前缀缓存命中率
    
- Running/Waiting Requests：运行中和等待中的请求数
    

DCGM 无法直接获取 TTFT、TPOT 或 KV Cache 使用率，这些指标必须来自推理框架或客户端压测工具。

### 3.3 关键 DCGM 指标

|   |   |   |
|---|---|---|
|指标|含义|用途|
|`DCGM_FI_DEV_GPU_UTIL`|GPU 总体繁忙率|初步判断 GPU 是否有工作|
|`DCGM_FI_PROF_SM_ACTIVE`|SM 中存在活跃 Warp 的时间比例|判断计算单元是否持续工作|
|`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`|Tensor Core 活跃比例|判断矩阵计算利用程度|
|`DCGM_FI_PROF_DRAM_ACTIVE`|显存接口活跃比例|判断显存带宽压力|
|`DCGM_FI_PROF_SM_OCCUPANCY`|SM Warp 占用率|辅助分析并行度|
|`DCGM_FI_DEV_FB_USED`|已使用显存|判断权重和 KV Cache 空间压力|
|`DCGM_FI_DEV_POWER_USAGE`|GPU 功耗|判断功耗与降频问题|
|`DCGM_FI_DEV_SM_CLOCK`|SM 时钟频率|判断是否发生降频|
|`DCGM_FI_DEV_GPU_TEMP`|GPU 温度|判断温度限制|
|`DCGM_FI_PROF_PCIE_RX_BYTES`|PCIe 接收流量|分析 Host 到 GPU 传输|
|`DCGM_FI_PROF_PCIE_TX_BYTES`|PCIe 发送流量|分析 GPU 到 Host 传输|
|`DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL`|NVLink 吞吐|分析多 GPU 通信|

不同 GPU、驱动和 DCGM 版本支持的 profiling 指标可能不同；Exporter 配置了某个指标，也不代表运行时一定能采集到。[DCGM Exporter 指标说明](https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html)

---

## 4. 监控架构

```
压测客户端
    │
    ├── 请求延迟、TTFT、TPOT、tokens/s
    │
推理服务 ── /metrics ─────────┐
                             │
GPU → DCGM → DCGM Exporter ──┼→ Prometheus → Grafana
                             │                  │
节点 CPU、网络监控 ────────────┘                  └→ Alertmanager
```

Prometheus 至少需要抓取两个目标：

1. DCGM Exporter：提供 GPU 指标。
    
2. 推理服务：提供请求、排队、批处理和 Token 指标。
    

Kubernetes 中通常在每个 GPU 节点部署一个 DCGM Exporter DaemonSet；也可以通过 NVIDIA GPU Operator 部署。[DCGM Exporter 安装方式](https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html)

实验环境可以使用较短的抓取周期，例如 1～5 秒；生产环境则需要综合考虑瞬时可见性、时序数量和存储成本。

---

## 5. Profiling 实验步骤

### 第一步：确定性能目标

先定义本轮实验的评价标准，例如：

- P99 延迟不超过 2 秒
    
- TTFT 不超过 500 毫秒
    
- TPOT 不超过 50 毫秒
    
- 单卡输出吞吐最大化
    
- 错误率低于 0.1%
    
- 显存使用率低于安全阈值
    

如果没有 SLO，只追求 GPU 利用率，容易得到“GPU 很忙，但用户体验很差”的配置。

### 第二步：固定实验变量

每组实验必须记录并固定：

- GPU 型号和数量
    
- MIG 配置
    
- 模型和版本
    
- FP16、BF16、FP8、INT8 等精度
    
- Tensor Parallel、Pipeline Parallel 配置
    
- 输入长度和输出长度分布
    
- 并发数
    
- 最大批大小
    
- 模型实例数
    
- KV Cache 配置
    
- 软件、驱动和 CUDA 版本
    

不要用随机且不受控的输入长度比较两个配置，否则结果可能主要来自工作负载差异。

### 第三步：验证 DCGM 指标

先查看目标 GPU 支持哪些 profiling 指标：

```
dcgmi profile --list --entity-id gpu:0
```

短时间观察 GPU 0：

```
dcgmi dmon \
  --entity-id gpu:0 \
  --field-id 1001,1004,1005 \
  --delay 1000
```

其中：

- `1001`：Graphics/Compute Engine Activity
    
- `1004`：Tensor Activity
    
- `1005`：DRAM Activity
    

实际使用前，应根据 `dcgmi profile --list` 的输出选择兼容的指标组。部分硬件无法同时采集所有 profiling counter。

确认 Exporter 确实暴露了指标：

```
curl http://dcgm-exporter:9400/metrics
```

重点检查是否包含：

```
DCGM_FI_PROF_SM_ACTIVE
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
DCGM_FI_PROF_DRAM_ACTIVE
DCGM_FI_DEV_FB_USED
DCGM_FI_DEV_POWER_USAGE
```

### 第四步：预热

模型加载完成后，不应立刻记录正式结果。预热可以消除：

- CUDA Context 初始化
    
- Kernel/JIT 编译
    
- 内存池建立
    
- 权重和缓存预取
    
- 动态批处理尚未稳定
    

可以先执行固定数量的请求，直到延迟、吞吐和 GPU 指标进入稳定区间。

### 第五步：逐级增加负载

推荐使用阶梯式并发：

```
并发：1 → 2 → 4 → 8 → 16 → 32 → 64
```

每个并发级别：

1. 预热。
    
2. 稳定运行一段时间。
    
3. 记录吞吐、P99、TTFT、TPOT。
    
4. 同时保存 GPU、CPU、网络和服务指标。
    
5. 等待资源恢复后再测试下一组。
    

关注两个拐点：

- 吞吐不再明显增长。
    
- P99、TTFT 或排队时间开始快速上升。
    

第一个违反 SLO 的并发点不能直接作为生产容量；通常要给突发流量和工作负载变化保留余量。

---

## 6. PromQL 查询示例

以下标签需要根据实际环境调整。

### GPU 平均利用率

```
avg by (gpu) (
  avg_over_time(DCGM_FI_DEV_GPU_UTIL[1m])
)
```

`GPU_UTIL` 高只能说明 GPU 持续有任务，不能证明 Tensor Core、计算能力或显存带宽已经充分利用。

### SM 活跃比例

Profiling Ratio 通常为 0～1，因此乘以 100 转换为百分比：

```
100 * avg by (gpu) (
  avg_over_time(DCGM_FI_PROF_SM_ACTIVE[1m])
)
```

### Tensor Core 活跃比例

```
100 * avg by (gpu) (
  avg_over_time(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE[1m])
)
```

### 显存带宽活跃比例

```
100 * avg by (gpu) (
  avg_over_time(DCGM_FI_PROF_DRAM_ACTIVE[1m])
)
```

### 最大显存占用

```
max by (gpu) (
  max_over_time(DCGM_FI_DEV_FB_USED[5m])
)
```

### GPU 平均功耗

```
avg by (gpu) (
  avg_over_time(DCGM_FI_DEV_POWER_USAGE[5m])
)
```

### 多卡负载差异

```
max(DCGM_FI_PROF_SM_ACTIVE)
-
min(DCGM_FI_PROF_SM_ACTIVE)
```

如果差异持续很大，需要检查：

- Tensor Parallel 是否均衡
    
- 不同 GPU 是否承担了不同角色
    
- 通信和同步等待
    
- 请求路由是否倾斜
    
- 某张卡是否发生降频
    

### 请求 P99 延迟

如果推理服务提供标准 Histogram：

```
histogram_quantile(
  0.99,
  sum by (le) (
    rate(inference_request_duration_seconds_bucket[5m])
  )
)
```

Histogram 的分位数应先聚合 bucket，再使用 `histogram_quantile()`；不要直接对各实例的 P99 求平均。[Prometheus Histogram 实践](https://prometheus.io/docs/practices/histograms/)

### Triton 每秒成功请求数

```
sum(
  rate(nv_inference_request_success[1m])
)
```

### Triton 平均排队时间

```
sum(rate(nv_inference_queue_duration_us[1m]))
/
sum(rate(nv_inference_request_success[1m]))
/
1000
```

结果单位为毫秒。Triton 会分别提供请求总时间、排队时间、输入处理、模型执行和输出处理等累计指标。[Triton Metrics 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2570/user-guide/docs/user_guide/metrics.html)

### 每 Token 能耗

如果推理引擎提供输出 Token Counter，可以近似计算：

```
sum(avg_over_time(DCGM_FI_DEV_POWER_USAGE[5m]))
/
sum(rate(generated_tokens_total[5m]))
```

因为瓦特等于焦耳/秒，所以结果约为焦耳/Token。

---

## 7. 如何判断瓶颈

|   |   |   |
|---|---|---|
|现象|可能原因|下一步|
|Tensor Active、SM Active 都很高|计算受限|FP8/INT8、算子融合、更高效引擎|
|DRAM Active 高，Tensor Active 较低|显存带宽受限|量化、连续批处理、减少权重/KV 访问|
|GPU 指标低，Queue Time 低|请求不足|增加并发或批大小|
|GPU 指标低，Queue Time 高|CPU、调度或同步瓶颈|查看 CPU、线程、批处理和锁等待|
|GPU 高，Queue Time 快速增加|GPU 容量饱和|扩容、限流或优化模型|
|PCIe 流量高，Input/Output Time 高|数据传输受限|Pinned Memory、减少复制、GPU Direct|
|显存接近耗尽|权重或 KV Cache 容量受限|量化、降低上下文长度、增加 GPU|
|功耗触顶、时钟下降|功耗或温度降频|检查功耗上限、散热和频率|
|多卡利用率差异大|并行策略或通信不均衡|检查拓扑、分片和请求路由|
|吞吐不增长但 P99 上升|已越过饱和点|回退并发并保留容量余量|

这些信号只能支持“瓶颈假设”，不能单靠一个指标得出结论。例如，高 Occupancy 不一定代表高性能；还需要结合 SM、Tensor、DRAM、吞吐和延迟一起判断。

---

## 8. LLM 的 Prefill 与 Decode

LLM 推理必须区分两个阶段：

### Prefill

- 一次处理多个输入 Token
    
- 通常矩阵运算规模较大
    
- 更容易利用 Tensor Core
    
- 主要影响 TTFT
    

实验负载可以采用：

```
长输入 + 短输出
```

例如固定较长 Prompt，只生成少量 Token。

### Decode

- 每次迭代通常只产生一个 Token
    
- 需要不断读取权重和 KV Cache
    
- 经常更容易受到显存带宽、Kernel Launch 和调度影响
    
- 主要影响 TPOT/ITL
    

实验负载可以采用：

```
短输入 + 长输出
```

在线连续批处理时，多个请求的 Prefill 和 Decode 会混合执行。DCGM 的秒级区间平均值无法自动分离它们，因此需要：

- 使用受控负载分别测试两个阶段；或
    
- 结合推理引擎 Trace；或
    
- 按压测时间窗口与 Prometheus 数据对齐。
    

---

## 9. 推荐实验记录表

|   |   |   |   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|---|---|
|配置|并发|输入/输出长度|吞吐|P99|TTFT|TPOT|SM|Tensor|DRAM|显存|功耗|
|BF16|1|512/128||||||||||
|BF16|8|512/128||||||||||
|BF16|32|512/128||||||||||
|FP8|8|512/128||||||||||

最终应输出：

- SLO 范围内的最大吞吐
    
- 推荐生产并发
    
- 推荐批大小
    
- 单卡 tokens/s
    
- 每 Token 能耗
    
- 性能瓶颈假设
    
- 下一步需要进行的深度分析
    

---

## 10. DCGM 不够时怎么办

当 DCGM 已经帮助定位到某个方向，但仍无法解释原因时：

- 使用 Nsight Systems 查看 CPU/GPU 时间线、同步、Memcpy 和 Kernel 间隙。
    
- 使用 Nsight Compute 分析具体 Kernel 的计算、访存和指令效率。
    
- 使用推理框架 Trace 分析调度、批处理、Prefill 和 Decode。
    
- 使用 CPU、网络和存储监控排除 GPU 之外的瓶颈。
    

DCGM profiling counter 可能与 Nsight 的硬件计数器发生冲突。运行 Nsight 前应暂停 DCGM profiling：

```
dcgmi profile --pause
```

分析结束后恢复：

```
dcgmi profile --resume
```

暂停操作会影响整个 DCGM Host Engine，而不只是某张 GPU。[NVIDIA Profiling 注意事项](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)

---

## 11. 总结

完整的推理 profiling 方法是：

```
确定 SLO
  → 固定工作负载
  → 采集客户端指标
  → 采集推理服务指标
  → 使用 DCGM 观察 GPU
  → 做并发和批大小扫描
  → 找到吞吐与延迟拐点
  → 建立瓶颈假设
  → 必要时使用 Nsight 深入验证
```

需要特别记住：

1. GPU Utilization 高不等于 GPU 性能已经充分发挥。
    
2. DCGM 负责发现硬件利用模式，不能替代 Kernel Profiler。
    
3. 吞吐必须与 P99、TTFT、TPOT 和错误率一起评价。
    
4. Prefill 和 Decode 应分开测试。
    
5. 最佳生产配置通常位于饱和点之前，而不是吞吐绝对最高点。