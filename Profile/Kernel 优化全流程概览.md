
> 面向 GPU 系统初学者。以 LLM 推理、PyTorch Profiler 和 SGLang 为例。

## 1. 一句话结论

**从真实请求的 GPU 时间线找到热点，按实际 shape 测量优化收益，再把新实现放回服务验证。**

```text
整模 Profile → 按阶段找热点 → 采集真实 shape 和调用次数
            → 修改实现 → 单算子对照 → 整模 A/B 复测
```

## 2. 对整模做 Profile

一个 **算子**是模型中的一次运算，例如线性层。一个 **kernel** 是提交给 GPU 执行的程序。一次线性层调用可以先启动量化 kernel，再启动矩阵乘 kernel。

PyTorch 同时记录 CPU 侧的算子调用和 GPU 上的 kernel 执行时间。下面假设模型和输入已在 GPU 上，并已预热。

```python
import torch
from torch.profiler import profile, ProfilerActivity

with torch.inference_mode(), profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as p:
    model(**inputs)

print(p.key_averages(group_by_input_shape=True).table(
    sort_by="self_cuda_time_total", row_limit=10
))
p.export_chrome_trace("model.trace.json")
```

如果模型已部署在 SGLang，保持代表性流量，向服务端发采集请求。`num_steps` 指引擎执行步数。

```bash
curl -X POST http://127.0.0.1:30000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"num_steps":20,"output_dir":"/tmp/sglang-profile"}'
```

在 Perfetto 里打开 trace，按 **prefill**（处理输入）、**decode**（逐步生成）和混合批次分别看。先按 **累计 GPU 时间**找候选，再看它是否拖住该阶段完成。通信可以单独统计，原始时间线要保留，以便判断计算和通信是否重叠。

## 3. 找出最值得优化的 shape

在 trace 里沿着 **kernel → CUDA launch → 算子调用点**回查。普通 PyTorch 算子可先看 `Input Shapes`。自定义算子信息不足时，在实际 launch 的 wrapper 里采样记录 shape、数据类型和布局。CUDA Graph 回放可记录捕获档位的执行 shape，再统计各档位的回放次数。

以 GEMM 为例，`X[M,K] × W[K,N] → Y[M,N]`。**`M` 通常随本轮处理的 token 数变化，`N、K` 主要由模型层和并行切分决定。** 流量、缓存命中、调度共同影响实际遇到哪些 `M`，以及各出现多少次。

用 **调用次数 × 单次耗时**排序。一个每次只花 `20 µs`、调用 1,000 次的 shape，比一个每次 `500 µs`、只调用 10 次的 shape 更耗时。

## 4. 针对瓶颈优化

| 瓶颈        | 可以尝试                      |
| --------- | ------------------------- |
| 显存搬运多     | 融合算子，减少中间结果写回，评估量化        |
| GPU 并行度不足 | 调整 tile（数据分块）、warp 和线程块组织 |
| 搬运或通信造成等待 | 让搬运与计算异步重叠                |

对代表性和边界 shape 先验算输出，再预热并重复测量。用 CUDA events 测 GPU 时间，分别记录 **kernel 本体**与**完整算子**的耗时。后者要计入量化、布局转换和其他必要开销。改变数值精度时还要验证模型质量。

## 5. 用真实调用次数加权，再复测整模

对比同一批调用。令 `n(s)` 为 shape `s` 的基线调用次数，`t(s)` 为单次耗时。

```text
优化前总耗时 = Σ n(s) × 优化前 t(s)
优化后总耗时 = Σ n(s) × 优化后 t(s)
耗时降幅     = 1 − 优化后总耗时 / 优化前总耗时
```

例如，小 shape 调用 1,000 次，单次从 `20 → 15 µs`。大 shape 调用 10 次，单次从 `500 → 600 µs`。大 shape 退化，但整组调用从 `25 → 21 ms`，**累计耗时下降 16%**。直接平均每种 shape 的百分比会丢掉调用频率与基线耗时的信息。

最后关闭 profiler 和详细埋点，保持模型、引擎配置和压测负载一致，对替换前后做整模 A/B。看 **TTFT（首 token 延迟）**、**TPOT（平均每个后续 token 的耗时）和 吞吐**。调用分布和计算重叠可能随实现改变，端到端收益以服务实测为准。

---

## 参考资料

- [PyTorch Profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe)
- [SGLang Benchmark and Profiling](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)
- [NVIDIA Nsight Compute 瓶颈诊断指南](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html)
