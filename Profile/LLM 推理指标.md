## 1. 先理解[[推理的两个阶段]]

一次自回归推理可以粗略分为：

1. **Prefill：处理输入 Prompt**
    
    - 一次处理大量输入 Token。
        
    - 通常偏==计算密集。
        
    - 主要影响 ==TTFT、输入吞吐量。
        
2. **Decode：逐步生成输出**
    
    - 每一步通常只生成一个或少数几个 Token。
        
    - 频繁读取模型权重和 KV Cache，通常偏==显存带宽密集。
        
    - 主要影响 ==TPOT、ITL、输出吞吐量。
        

可以记成：

> ==长 Prompt 看 Prefill，长输出看 Decode；低并发看单请求速度，高并发看调度、批处理和 KV Cache。

---

## 2. 常用推理服务指标

笔者理解：low latency & high throughput 是一对对偶的概念。
### 2.1 延迟指标

| 指标          | 含义                                              | 重点场景            |
| ----------- | ----------------------------------------------- | --------------- |
| TTFT        | Time To First Token，从发送请求到收到第一个token            | 聊天、搜索、实时交互      |
| TPOT        | Time Per Output Token，首 Token 后平均生成每个 Token 的时间 | 持续生成速度          |
| ITL         | Inter-Token Latency，相邻两次流式输出块之间的时间              | 输出是否流畅、是否卡顿     |
| E2E Latency | 从发送请求到完整输出结束                                    | 非流式任务、Agent 子任务 |
| Queue Time  | 请求进入服务后等待调度的时间                                  | 高并发、过载分析        |

vLLM 对 TPOT 的定义为：

$$
TPOT=\frac{E2E-TTFT}{N_{output}-1}  
$$

因此可以近似认为：

$$
E2E \approx TTFT+(N_{output}-1)\times TPOT  
$$

若 TPOT 使用毫秒，则单请求解码速度约为：

$$
GenerationSpeed \approx \frac{1000}{TPOT}\quad token/s  
$$

vLLM 特别提醒：**ITL 按流式输出块统计，TPOT 按实际 Token 数摊销**。使用投机解码时，一个流式块可能包含多个 Token，因此 ITL 和 TPOT 可能明显不同。[vLLM 延迟指标定义](https://docs.vllm.ai/en/latest/benchmarking/cli/)

### 2.2 吞吐量指标

|   |   |   |
|---|---|---|
|指标|计算方式|用途|
|Request Throughput|完成请求数 / 时间|API 服务容量|
|Input Token Throughput|输入 Token 数 / 时间|Prefill 能力|
|Output Token Throughput|输出 Token 数 / 时间|Decode 能力|
|Total Token Throughput|输入与输出 Token 总数 / 时间|离线批处理|
|Peak Concurrency|同时处于系统中的最大请求数|容量与 KV Cache 分析|

注意：

- `req/s` 强烈依赖每个请求的输入、输出长度。
    
- 比较框架时，必须同时固定输入长度和输出长度。
    
- 总 Token 吞吐量高，不代表用户感受到的 TTFT 或 ITL 好。
    

### 2.3 Goodput

Goodput 指：

> 单位时间内，满足延迟 SLO 的成功请求数。

例如同时要求：

- TTFT ≤ 1 秒；
    
- TPOT ≤ 50 毫秒；
    
- E2E ≤ 15 秒；
    
- 请求成功。
    

即使某配置总吞吐量很高，如果大量请求超时，它的 Goodput 仍然可能很低。vLLM Benchmark 可通过 `--goodput ttft:1000 tpot:50 e2el:15000` 设置阈值。[vLLM Goodput 参数](https://docs.vllm.ai/en/latest/cli/bench/serve/)

### 2.4 容量与稳定性指标

生产环境还应关注：

- P50、P95、P99 TTFT/TPOT/E2E；
    
- 最大可持续 RPS；
    
- 排队请求数；
    
- 失败率、超时率；
    
- KV Cache 使用率；
    
- KV Cache 抢占、重算或请求回退次数；
    
- Prefix Cache 命中率；
    
- GPU 显存、计算利用率和功耗。
    

“最大可持续 RPS”通常定义为：在满足 P99 延迟和错误率要求的前提下，系统能够长期承载的最高请求速率。

### 2.5 质量与成本指标

性能测试不能脱离模型质量：

- 任务准确率、Exact Match、Pass@k；
    
- Perplexity；
    
- 结构化输出成功率；
    
- 工具调用成功率；
    
- 量化前后的精度下降；
    
- GPU 秒/请求；
    
- 每百万输出 Token 成本。
    

成本可粗略计算为：

$$
Cost/MToken=  
\frac{GPU数量\times GPU小时单价\times 测试时间}  
{输出Token数}\times 10^6  
$$

---

## 3. ==vLLM 与 SGLang 核心参数对照

### 3.1 显存和 KV Cache

|             |                            |                               |              |
| ----------- | -------------------------- | ----------------------------- | ------------ |
| 作用          | vLLM                       | SGLang                        | 主要影响         |
| GPU 内存预算    | `--gpu-memory-utilization` | `--mem-fraction-static`       | KV 容量、并发、OOM |
| 精确指定 KV 容量  | `--kv-cache-memory-bytes`  | `--max-total-tokens`          | 最大缓存 Token 数 |
| KV 精度       | `--kv-cache-dtype`         | `--kv-cache-dtype`            | KV 容量、带宽、质量  |
| KV 分块粒度     | `--block-size`             | `--page-size`                 | 内存碎片、缓存匹配粒度  |
| CPU KV 分层缓存 | `--kv-offloading-size`     | `--enable-hierarchical-cache` | 容量增加，但引入传输延迟 |

两个内存比例参数不能简单等价：

- vLLM 的 `--gpu-memory-utilization` 是当前实例可使用的 GPU 内存比例；如果显式设置 `--kv-cache-memory-bytes`，**后者会覆盖前者对 KV Cache 大小的推导。
    
- SGLang 的 `--mem-fraction-static` 主要决定“**模型权重 + KV Cache 池**”的静态内存比例。提高它通常能==增加 KV Cache 和并发==，但留给==激活值、CUDA Graph 的空间会变少==。[vLLM Cache 参数](https://docs.vllm.ai/en/latest/configuration/engine_args/)，[SGLang 内存与调度参数](https://docs.sglang.io/docs/advanced_features/server_arguments)
    

调优方向：

- KV Cache 太小：并发和峰值吞吐受限。
    
- KV Cache 太大：运行时激活或 CUDA Graph 可能 OOM。
    
- FP8 KV Cache：通常可以显著提升容量，但必须做质量回归；SGLang 对 FP8 KV Cache 建议提供正确的量化缩放参数，否则可能出现精度问题。
    

### 3.2 上下文与并发

|               |                            |                          |
| ------------- | -------------------------- | ------------------------ |
| 作用            | vLLM                       | SGLang                   |
| 最大上下文         | `--max-model-len`          | `--context-length`       |
| 单轮最大序列数       | `--max-num-seqs`           | `--max-running-requests` |
| 单轮 Token 预算   | `--max-num-batched-tokens` | `--max-prefill-tokens`   |
| 最大 Prefill 分块 | 由 Token 预算触发分块             | `--chunked-prefill-size` |

注意这些参数只是相似，并非**完全等价：

- vLLM 的 `max-num-batched-tokens` 限制一次调度迭代处理的总 Token 数。
    
- SGLang 的 `max-prefill-tokens` 主要限制 Prefill Batch，`chunked-prefill-size` 限制单个 Prefill Chunk。
    
- `max-num-seqs` 是单次迭代可处理的序列数，而 `max-running-requests` 是同时运行的请求上限。
    

增大并发或 Batch 通常可以提高峰值吞吐，但也可能：

- 增加排队和尾延迟；
    
- 占用更多 KV Cache；
    
- 增加 Prefill 对 Decode 的干扰；
    
- 导致抢占、回退或 OOM。
    

### 3.3 [[Chunked Prefill]]

|        |                                                                        |
| ------ | ---------------------------------------------------------------------- |
| 框架     | 参数                                                                     |
| vLLM   | `--enable-chunked-prefill`、`--max-num-batched-tokens`                  |
| SGLang | `--chunked-prefill-size`、`--max-prefill-tokens`、`--enable-mixed-chunk` |

vLLM V1 会在可用时默认启用 Chunked Prefill，并优先调度 Decode：

- 较小的 `max-num-batched-tokens` 通常更==有利于 ITL；
    
- 较大的值可以==加速 Prefill、改善 TTFT和峰值吞吐；
    
- 数值过大可能让 Prefill ==长时间占用 GPU，影响 Decode 流畅度==。[vLLM Chunked Prefill 调优](https://docs.vllm.ai/en/latest/configuration/optimization/)
    

SGLang 中：

- ==Prefill OOM== 时可降低 `--chunked-prefill-size`；
    
- 较小的 Chunk 节省峰值显存，但长 Prompt 的 Prefill 会变慢；
    
- ==Decode OOM== 时优先降低 `--max-running-requests`。[SGLang 调优指南](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)
    

### 3.4[[ Prefix Cache]]

|   |   |
|---|---|
|vLLM|SGLang|
|`--enable-prefix-caching`|Radix Cache 默认开启；`--disable-radix-cache` 用于关闭|
|`--block-size`、`--prefix-match-unit`|`--page-size`、`--schedule-policy lpm`|

适用场景：

- 固定 System Prompt；
    
- 多轮对话；
    
- Few-shot Prompt；
    
- RAG 中共享长模板；
    
- Agent 请求共享工具描述。
    

主要收益是减少重复 Prefill，因此通常==显著降低 TTFT、提高输入吞吐量；对纯 Decode 的 TPOT 没有直接加速。

SGLang 的 `lpm` 表示 Longest Prefix Match，会重新排序请求以增加缓存命中，但会增加调度开销。[SGLang 调优指南](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)

### 3.5 调度策略

|   |   |
|---|---|
|vLLM|SGLang|
|`--scheduling-policy fcfs/priority`|`--schedule-policy fcfs/lpm/priority/...`|
|`--watermark`|`--schedule-conservativeness`|
|`--async-scheduling`|默认重叠调度；`--disable-overlap-schedule` 关闭|

SGLang 的 `--schedule-conservativeness`：

- 降低：接纳更多请求，提高 KV Token 使用率和吞吐；
    
- 过低：容易填满 KV Cache，导致请求 Retract；
    
- 提高：保留更多安全余量，减少 Retract；
    
- 过高：GPU 和 KV Cache 利用不足。
    

官方建议观察日志中的 `token usage`、排队请求数和 Retract 警告后再调整，而不是直接套固定值。[SGLang 调度调优](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)

### 3.6 精度、量化和计算后端

两者常见参数：

- `--dtype`
    
- `--quantization`
    
- `--kv-cache-dtype`
    
- Attention、Sampling、MoE Kernel Backend
    
- CUDA Graph / `torch.compile`
    

影响规律：

- 权重量化降低模型显存，可能让单卡容纳更大的 Batch 或更多 KV Cache。
    
- 是否真正提速取决于 **GPU 架构和量化 Kernel**；更低位宽不一定更快。
    
- [[KV Cache]] 量化主要改善==长上下文和高并发容量。
    
- [[量化]]必须==重新测试任务准确率，尤其是数学、代码和长链推理。
    
- [[CUDA Graph ]]通常降低小 Batch Decode 的启动开销，但会额外占用显存。
    

vLLM 通过 `--compilation-config` 控制 `torch.compile` 与 CUDA Graph；SGLang 可用 `--cuda-graph-config`、`--cuda-graph-max-bs-decode` 等参数。SGLang 官方指出，提高 CUDA Graph 最大 Batch 可能改善大并发性能，但需要为 Graph Buffer 预留更多显存。[vLLM Compilation 参数](https://docs.vllm.ai/en/latest/configuration/engine_args/)，[SGLang CUDA Graph 参数](https://docs.sglang.io/docs/advanced_features/server_arguments)

### 3.7 [[并行策略]]

|   |   |   |   |
|---|---|---|---|
|策略|vLLM|SGLang|特点|
|Tensor Parallel|`--tensor-parallel-size`|`--tp-size`|用通信换模型切分|
|Pipeline Parallel|对应 PP 配置|`--pp-size`|适合超大模型，但有流水线气泡|
|Data Parallel|`--data-parallel-*` 或独立副本|`--dp-size`/Router|显存足够时适合提高吞吐|
|Expert Parallel|`--enable-expert-parallel`|`--ep-size`|MoE 模型|
|Context Parallel|PCP/DCP|`--attn-cp-size`、`--dcp-size`|长上下文模型|

经验：

- 模型单卡放不下：先考虑 TP。
    
- 模型每卡都能放下、目标是吞吐：优先多个 DP 副本。
    
- TP 过大可能因 All-Reduce 通信降低单请求性能。
    
- MoE 模型需要结合 EP、DP Attention 和 All-to-All 后端单独测试。
    
- vLLM 当前原生 DP 参数更偏向 MoE 部署；普通 Dense 模型常使用多个独立实例和外部负载均衡。
    

### 3.8 [[投机解码]]

|   |   |
|---|---|
|vLLM|SGLang|
|`--speculative-config`|`--speculative-algorithm`|
|支持 Draft、EAGLE、MTP、N-Gram、Suffix 等|支持 EAGLE、EAGLE3、NEXTN、NGRAM 等|

主要目标：

- 降低 TPOT；
    
- 提升低至中等 QPS 下的单请求生成速度。
    

关键观测指标：

- Draft Token 接受率；
    
- 每轮接受 Token 数；
    
- TPOT；
    
- 输出吞吐量；
    
- Draft 模型额外显存；
    
- 高 QPS 下是否反而降低总吞吐。
    

投机长度不是越大越好：接受率低时，生成和验证 Draft Token 的成本可能高于收益。vLLM 官方也将其主要定位在低至中等 QPS、Decode 受显存带宽限制的场景。[vLLM 投机解码](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

### 3.9 流式输出

两者均有 `--stream-interval`：

- 较小：更快地向客户端发送 Token，ITL 更平滑；
    
- 较大：减少 CPU、序列化和网络开销，吞吐可能提高；
    
- 较大时客户端测得的 ITL 会被“输出分块”放大。
    

因此比较 ITL 时，必须保证两个框架使用相同的 Streaming Interval。

---

## 4. 请求级参数对指标的影响

|                                 |                                |
| ------------------------------- | ------------------------------ |
| 参数                              | 影响                             |
| `max_tokens` / `max_new_tokens` | 决定最坏 Decode 时间、KV 占用和成本        |
| `min_tokens` / `min_new_tokens` | 强制延长输出，增加 E2E 和负载              |
| `stop` / `stop_token_ids`       | 提前结束可显著降低平均输出长度                |
| `ignore_eos`                    | 常用于固定长度 Benchmark，但会制造非真实负载    |
| `n`                             | 一次生成多路结果，显著增加 Decode 计算与 KV 占用 |
| `temperature`                   | 主要影响随机性和质量；`0` 常用于可复现测试        |
| `top_p` / `top_k` / `min_p`     | 影响采样分布，通常只带来较小的采样开销差异          |
| `logprobs` / `prompt_logprobs`  | 增加 Logits 处理、显存、CPU 和网络传输开销    |
| JSON Schema/Regex/Grammar       | 增加约束解码开销，但可能减少重试和无效输出          |

SGLang 原生 API 使用 `max_new_tokens`，OpenAI 兼容 API 通常使用 `max_tokens`；vLLM 使用 `max_tokens`。两者都支持 temperature、top-p、top-k、停止条件和惩罚参数。[vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)，[SGLang Sampling Parameters](https://docs.sglang.io/docs/basic_usage/sampling_params)

---

## 5. 按目标选择调优方向

|现象|更可能的原因|
|---|---|
|Queue Time 高，TTFT 高，TPOT 正常|系统过载、KV 容量不足、请求排队|
|Queue Time 正常，TTFT 高|Prefill 慢、长输入、PCP 通信或 KV 传输|
|TTFT 正常，TPOT 和 ITL 高|Decode/KV 带宽、DCP/TP/EP 通信|
|TPOT 正常，P95 ITL 高|Prefill 干扰、调度抖动、batch 切换|
|P95 TTFT 高但平均 TTFT 正常|少量长 Prompt、排队长尾或负载不均|
|TTFT、TPOT 都正常，E2E 高|输出长度大，或者客户端/网络/收尾开销|
|开 PCP 后 TPOT 下降|通常是 Prefill 阻塞减少，不一定是 Decode kernel 变快

### 目标 A：降低交互延迟

优先关注：

1. P99 TTFT；
    
2. P99 ITL；
    
3. TPOT；
    
4. Goodput。
    

常用方向：

- 启用 Prefix Cache；
    
- 使用 Chunked Prefill，避免长 Prompt 阻塞 Decode；
    
- 限制最大并发和输出长度；
    
- 使用 CUDA Graph；
    
- 低至中 QPS 时测试投机解码；
    
- 避免过大的 Prefill Batch；
    
- 保持较小的 Stream Interval。
    

### 目标 B：提高离线吞吐量

常用方向：

- 逐步提高并发和 Batch Token 预算；
    
- 增大 KV Cache，但保留运行时显存余量；
    
- 显存足够时优先增加 DP 副本；
    
- 使用合适的权重/KV 量化；
    
- MoE 模型测试 EP、DP Attention；
    
- 允许适当牺牲 TTFT 和 ITL；
    
- 保证客户端提交速度足够，服务端始终有待处理请求。
    

### 目标 C：处理 OOM

vLLM 可依次检查：

- 降低 `--max-num-batched-tokens`；
    
- 降低 `--max-num-seqs`；
    
- 降低 `--gpu-memory-utilization`；
    
- 缩短 `--max-model-len`；
    
- 使用 KV/权重量化；
    
- 最后再考虑 CPU Offload。
    

SGLang 可依次检查：

- Prefill OOM：降低 `--chunked-prefill-size`；
    
- Decode OOM：降低 `--max-running-requests`；
    
- 降低 `--mem-fraction-static`；
    
- 减小 CUDA Graph 捕获 Batch；
    
- 使用 KV/权重量化。
    

CPU Offload 通常是“用延迟换容量”，不应视为免费显存。

---

## 8. 最重要的结论

1. **不存在脱离工作负载的“最优参数”。**
    
2. TTFT、ITL/TPOT、吞吐量通常构成一条 Pareto 曲线。
    
3. 提高并发一般能提高吞吐，但会恶化排队和尾延迟。
    
4. Prefix Cache 主要优化重复 Prefill，而不是 Decode。
    
5. Chunked Prefill 的核心作用是协调长 Prefill 与在线 Decode。
    
6. KV Cache 容量决定长上下文服务的并发上限。
    
7. 调优时优先观察 P99 和 Goodput，不要只看平均吞吐量。
    
8. Benchmark 中最容易造成误判的变量是输出长度、`ignore_eos`、缓存命中率和请求到达方式。