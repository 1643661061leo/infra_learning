## 1. Profiling 的目标

推理 profiling 不是单纯观察“GPU 利用率高不高”，而是回答：

1. 当前服务能达到多少吞吐？
    
2. 延迟由排队、数据传输还是 GPU 计算造成？
    
3. GPU 受计算、显存带宽、通信还是 CPU 限制？
    
4. 最佳并发数、批大小和模型实例数是多少？
    
5. 性能达到拐点后，应该优化参数还是增加 GPU？
    

DCGM 和 Prometheus 主要完成“持续观测与对比”。如果要定位到具体 CUDA Kernel，则需要进一步使用 [[Nsight Systems]] 或 [[Nsight Compute]]。

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

DCGM profiling 指标是一个采样区间内的==平均值==，适合==低开销持续监控==，但不是 Kernel Trace，无法直接指出哪个算子或代码行导致问题。[NVIDIA DCGM Profiling 文档](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)

---

## 3. 推理场景的核心指标

### 3.1 通用推理指标

- Throughput：每秒请求数
    
- Latency：平均、P50、P95、P99 延迟
    
- Queue Time：请求排队时间
    
- Compute Time：模型实际执行时间
    
- Batch Size：实际执行批大小
    
- Error Rate：超时和失败比例
    

### 3.2 [[LLM 推理指标]]

- TTFT：Time To First Token，首 Token 延迟
    
- TPOT：Time Per Output Token，平均每个输出 Token 时间
    
- ITL：Inter-Token Latency，相邻 Token 间隔
    
- Input tokens/s：输入 Token 吞吐
    
- Output tokens/s：输出 Token 吞吐
    
- KV Cache Usage：KV Cache 使用率
    
- Prefix Cache Hit Rate：前缀缓存命中率
    
- Running/Waiting Requests：运行中和等待中的请求数
    

DCGM 无法直接获取 TTFT、TPOT 或 KV Cache 使用率，这些指标必须来自==推理框架或客户端压测工具。==

### 3.3 DCGM

DCGM 全称 **NVIDIA Data Center GPU Manager**，是 NVIDIA 提供的数据中心 GPU 管理与诊断工具。

它主要提供：

- GPU 指标采集：利用率、显存使用量、温度、功耗、时钟频率等
- GPU 健康检查：PCIe、显存、温度、电源等
- 错误监控：ECC 错误、XID 错误、掉卡等
- GPU 诊断和压力测试
- 多 GPU、NVLink 和 MIG 监控
- GPU 策略与运行状态管理
### 3.4 关键 DCGM 指标(速查)

| 指标                                   | 含义                  | 用途                  |
| ------------------------------------ | ------------------- | ------------------- |
| `DCGM_FI_DEV_GPU_UTIL`               | GPU 总体繁忙率           | 初步判断 GPU 是否有工作      |
| `DCGM_FI_PROF_SM_ACTIVE`             | SM 中存在活跃 Warp 的时间比例 | 判断计算单元是否持续工作        |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`    | Tensor Core 活跃比例    | 判断矩阵计算利用程度          |
| `DCGM_FI_PROF_DRAM_ACTIVE`           | 显存接口活跃比例            | 判断显存带宽压力            |
| `DCGM_FI_PROF_SM_OCCUPANCY`          | SM Warp 占用率         | 辅助分析并行度             |
| `DCGM_FI_DEV_FB_USED`                | 已使用显存               | 判断权重和 KV Cache 空间压力 |
| `DCGM_FI_DEV_POWER_USAGE`            | GPU 功耗              | 判断功耗与降频问题           |
| `DCGM_FI_DEV_SM_CLOCK`               | SM 时钟频率             | 判断是否发生降频            |
| `DCGM_FI_DEV_GPU_TEMP`               | GPU 温度              | 判断温度限制              |
| `DCGM_FI_PROF_PCIE_RX_BYTES`         | PCIe 接收流量           | 分析 Host 到 GPU 传输    |
| `DCGM_FI_PROF_PCIE_TX_BYTES`         | PCIe 发送流量           | 分析 GPU 到 Host 传输    |
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | NVLink 吞吐           | 分析多 GPU 通信          |

不同 GPU、驱动和 DCGM 版本支持的 profiling 指标可能不同；Exporter 配置了某个指标，也不代表运行时一定能采集到，不用背，需要的时候直接查。[DCGM Exporter 指标说明](https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html)

### 3.5 Prometheus

Prometheus 是一个开源的监控和告警系统，特别适合云原生、容器和 Kubernetes 环境。

它的核心能力包括：
- 定期从目标服务拉取指标
- 按时间序列存储数据
- 使用 PromQL 查询和聚合指标
- 根据规则计算新的指标
- 通过 Alertmanager 发送告警
- 与 Grafana 配合展示监控面板

Prometheus 通常采用 **Pull 模型**：主动访问目标的 `/metrics` 地址，而不是等待目标把数据推过来。

---

## 4. 监控架构

### 二者如何配合

在 GPU 或 AI 集群中，各组件通常这样分工：

|组件|职责|
|---|---|
|DCGM|读取和诊断 NVIDIA GPU 状态|
|DCGM Exporter|把 DCGM 指标暴露为 Prometheus 格式|
|Prometheus|采集、存储和查询指标|
|Alertmanager|根据规则发送告警|
|Grafana|展示监控图表和仪表盘|

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

## 5. 运行流程

- 写Prometheus 配置 
```
global:

	scrape_interval: 1s
	
	evaluation_interval: 1s

  
scrape_configs:

	- job_name: dcgm-exporter

		static_configs:
		
			- targets: ['localhost:9400']


- job_name: sglang

	metrics_path: /metrics

	static_configs:

		- targets: ['localhost:8000'] # your_port
```
    - 抓取 DCGM Exporter `:9400`
    - 抓取 SGLang `:8000/metrics`


```
cd /home/ubuntu/your_test
```

启动 DCGM：

```
sudo systemctl enable --now nvidia-dcgm.service
sudo systemctl status nvidia-dcgm.service --no-pager
```

首次拉取并启动 Exporter：

```
docker pull nvcr.io/nvidia/k8s/dcgm-exporter:latest

docker run -d \
  --name leantune-dcgm-exporter \
  --restart unless-stopped \
  --gpus all \
  --net host \
  --cap-add SYS_ADMIN \
  nvcr.io/nvidia/k8s/dcgm-exporter:latest \
  -r tcp://localhost:5555
```

验证：

```
curl -fsS http://localhost:9400/metrics | head
```

启动Prometheus 镜像：

```
docker volume create leantune-prometheus-data

docker run -d \
  --name leantune-prometheus \
  --restart unless-stopped \
  --network host \
  -v /home/ubuntu/your_test/prometheus_sglang_dcgm.yml:/etc/prometheus/prometheus.yml:ro \
  -v leantune-prometheus-data:/prometheus \
  prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus
```

验证：

```
curl -fsS http://localhost:9090/-/ready
curl -fsS http://localhost:9090/api/v1/targets
```

---
## 6. 如何判断瓶颈

|                                |                   |                               |
| ------------------------------ | ----------------- | ----------------------------- |
| 现象                             | 可能原因              | 下一步                           |
| Tensor Active、SM Active 都很高    | 计算受限              | FP8/INT8、算子融合、更高效引擎           |
| DRAM Active 高，Tensor Active 较低 | 显存带宽受限            | 量化、连续批处理、减少权重/KV 访问           |
| GPU 指标低，Queue Time 低           | 请求不足              | 增加并发或批大小                      |
| GPU 指标低，Queue Time 高           | CPU、调度或同步瓶颈       | 查看 CPU、线程、批处理和锁等待             |
| GPU 高，Queue Time 快速增加          | GPU 容量饱和          | 扩容、限流或优化模型                    |
| PCIe 流量高，Input/Output Time 高   | 数据传输受限            | Pinned Memory、减少复制、GPU Direct |
| 显存接近耗尽                         | 权重或 KV Cache 容量受限 | 量化、降低上下文长度、增加 GPU             |
| 功耗触顶、时钟下降                      | 功耗或温度降频           | 检查功耗上限、散热和频率                  |
| 多卡利用率差异大                       | 并行策略或通信不均衡        | 检查拓扑、分片和请求路由                  |
| 吞吐不增长但 P99 上升                  | 已越过饱和点            | 回退并发并保留容量余量                   |

这些信号只能支持“瓶颈假设”，不能单靠一个指标得出结论。例如，高 Occupancy 不一定代表高性能；还需要结合 SM、Tensor、DRAM、吞吐和延迟一起判断。

---

## 7. LLM 的 [[Prefill 与 Decode]]

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

## 8. DCGM 不够时怎么办

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

## 9. 总结

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