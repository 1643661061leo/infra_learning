# EP 与 MoE 推理：原理、通信与框架应用

> 更新口径：2026-09-09。聚焦 MoE 推理；框架参数变化较快，部署时以实际版本的 `--help`、启动日志和官方文档为准。

## 1. 核心结论

- **MoE 是模型结构**：每个 token 只激活少量 experts。
- **EP 是并行策略**：不同 GPU 保存不同 experts，token ==根据 Router 结果==跨 GPU 流动。
- EP 分摊 expert 权重和计算，但每层通常增加两次通信：**Dispatch** 和 **Combine**。
- EP 不直接切 KV Cache；MLA/MoE 推理常使用 `DP Attention + EP`。
- EP size 越大，每卡 expert 权重越少，但单卡 GEMM 更小、All-to-All 更宽、负载长尾更明显。
- Prefill token 多，通常追求高吞吐；Decode 每步 token 少，通常追求低延迟与 CUDA Graph 兼容。

---

## 2. MoE 到底怎样计算：Tensor Shape 逐步变化

### 2.1 符号

```text
B = batch / request 数
S = 每个请求本轮参与计算的 token 数
T = B × S，本轮总 token 数
H = hidden size
I = expert intermediate size
E = routed expert 总数
K = 每个 token 选择的 expert 数，即 top-k
A = T × K，token-expert assignment 总数
```

Transformer 传入 MoE 层的 hidden states：

```text
X: [B,S,H] -> reshape -> [T,H]
```

### 2.2 Router 计算

Router 权重：

```text
W_router: [H,E]
```

计算全部 expert 分数：

```text
router_logits = X @ W_router

[T,H] @ [H,E] -> [T,E]
```

再对 expert 维做 top-k：

```text
topk_ids:     [T,K]
topk_weights: [T,K]
```

`topk_ids[t,j]` 表示 token `t` 的第 `j` 个目标 expert；`topk_weights[t,j]` 是该 expert 输出最终占的权重。

Router 的主要计算量近似为：

```text
Router FLOPs ≈ 2 × T × H × E
```

### 2.3 从 token 变成 token-expert assignments

每个 token 要送到 `K` 个 experts，因此在逻辑上将 token 复制 `K` 份：

```text
X:              [T,H]
X_expanded:     [T,K,H] -> [A,H]
expert_ids:     [T,K]   -> [A]
expert_weights: [T,K]   -> [A]
source_ids:     [T,K]   -> [A]
```

这里不是把 hidden size 扩大，而是把同一个 `[1,H]` token 创建 `K` 个 expert assignments。

例如 `T=4, K=2`：

```text
token 0 -> expert 0, expert 2
token 1 -> expert 2, expert 3
token 2 -> expert 1, expert 2
token 3 -> expert 0, expert 3
```

展开后：

```text
assignment 0: token 0 -> expert 0
assignment 1: token 0 -> expert 2
assignment 2: token 1 -> expert 2
assignment 3: token 1 -> expert 3
assignment 4: token 2 -> expert 1
assignment 5: token 2 -> expert 2
assignment 6: token 3 -> expert 0
assignment 7: token 3 -> expert 3

X_expanded: [8,H]
expert_ids: [8]
```

### 2.4 按 expert 分组

设 expert `e` 收到 `n_e` 个 assignments：

```text
Σ n_e = A = T × K
```

把 `X_expanded` 按 `expert_ids` 排序/permute：

```text
expert 0 input: X_0 [n_0,H]
expert 1 input: X_1 [n_1,H]
...
expert E-1 input: X_E-1 [n_E-1,H]
```

上例中：

```text
n_0 = 2
n_1 = 1
n_2 = 3
n_3 = 2
```

### 2.5 每个 expert 怎样计算

常见 gated MLP expert 有三块权重：

```text
W_gate,e: [H,I]
W_up,e:   [H,I]
W_down,e: [I,H]
```

对 expert `e`：

```text
G_e = X_e @ W_gate,e     # [n_e,H] @ [H,I] -> [n_e,I]
U_e = X_e @ W_up,e       # [n_e,H] @ [H,I] -> [n_e,I]
Z_e = SiLU(G_e) * U_e    # [n_e,I]
Y_e = Z_e @ W_down,e     # [n_e,I] @ [I,H] -> [n_e,H]
```

有些实现把 `W_gate` 与 `W_up` 合并：

```text
W_gate_up,e: [H,2I]
X_e @ W_gate_up,e -> [n_e,2I]
```

==这不改变数学含义，只减少 kernel 和内存访问开销。==

所有 experts 的主要计算量近似为：

```text
Expert FLOPs ≈ K × 6 × T × H × I
```

三个矩阵乘各按一次乘加为 2 FLOPs 估算，忽略激活与加权求和。

由于不同 `n_e` 大小不一样，工程上通常使用 **Grouped GEMM**，而不是为每个 expert 单独启动一个小 GEMM。

### 2.6 恢复 token 顺序并做 top-k 加权

Experts 输出按 assignment 排列：

```text
Y_expert_order: [A,H]
```

先 inverse-permute 回原 token/top-k 顺序：

```text
Y_assignments: [A,H] -> [T,K,H]
```

乘 topk_weights 并沿 K 求和：

```text
Y = Σ_K topk_weights[:,:,None] × Y_assignments

[T,K,1] × [T,K,H] -> sum(dim=1) -> [T,H]
```

最后 reshape 回 Transformer 需要的 shape：

```text
[T,H] -> [B,S,H]
```

完整的数学 shape 流程：

```text
[B,S,H]
  -> [T,H]
  -> Router [T,E]
  -> TopK ids/weights [T,K]
  -> assignments [A,H], A=T×K
  -> 按 expert 分组 [n_e,H]
  -> expert MLP [n_e,H]
  -> assignments [T,K,H]
  -> 加权求和 [T,H]
  -> [B,S,H]
```

---

## 3. EP 到底怎样做：Dispatch、All-to-All 与 Combine

### 3.1 Expert 权重怎样放置

设 `E=8, EP=4`，最简单的线性放置是：

```text
rank 0: experts 0,1
rank 1: experts 2,3
rank 2: experts 4,5
rank 3: experts 6,7
```

建立映射：

```text
expert_to_rank[e] = expert e 所在的 EP rank
```

每 rank 的 expert 权重理想约降为总 expert 权重的 `1/EP`。Attention、embedding、dense/shared FFN 和 LM head 如何放置，由 TP、DP、PP 等其他策略决定。

### 3.2 每个来源 rank 先做本地 Router

在 DPA + EP 中，不同 ranks 有不同请求。设 rank `r` 本地有：

```text
X_r: [T_r,H]
```

Router 输出：

```text
topk_ids_r:     [T_r,K]
topk_weights_r: [T_r,K]
```

展开后本地 assignment 数：

```text
A_r = T_r × K
X_expanded_r: [A_r,H]
```

### 3.3 Dispatch 第一步：计算发给每个 rank 的数量

根据 `expert_to_rank`，得到每个 assignment 的目标 rank：

```text
dest_rank[a] = expert_to_rank[expert_ids[a]]

例如
e0,e1 在 rank 0
e2,e3 在 rank 1

a/e0 -> dest_rank 0
a/e2 -> dest_rank 1
```
然后统计：

```text
send_counts_r[p] = rank r 要发给目标 rank p 的 assignment 数
```

例如 rank 0 有 6 个 assignments：

```text
send_counts_0 = [2,4]
```

表示 2 个留给/发给 rank 0，4 个发给 rank 1。

### 3.4 Dispatch 第二步：按目标 rank 打包

将 hidden states 按 `dest_rank` 排列：

```text
send_x:          [A_r,H]
send_expert_ids: [A_r]
send_metadata:   [A_r,...]
```

“按 `dest_rank` 排列 hidden states”是==把目标相同的行放到连续区域，向量内容不变：==

```text
打包前： [a/e2, b/e0, c/e3, d/e1]   dest=[1,0,1,0]
打包后： [b/e0, d/e1 | a/e2, c/e3]
            rank 0 | rank 1
                    
```

### 3.5 Dispatch 第三步：All-to-All-V

EP 通常需要 **变长 All-to-All**，因为每个 rank 发给其他 ranks 的 token 数不同。

==All-to-All 就是每个 rank 按目标 expert 所在位置，把自己的 token 分别发送给所有对应 ranks，同时接收其他 ranks 发给自己本地 experts 的 token。  ==

Dispatch 用 All-to-All 把 token 送到 expert，Combine 再用一次反向 All-to-All 把计算结果送回 token 原来的 rank。

对 rank `r`：

```text
输入： send_x [A_r,H]
输出： recv_x [R_r,H]

R_r = Σ_i send_counts_i[r]
```

例如

```text
                 发到 rank 0  发到 rank 1
来源 rank 0           1            3
来源 rank 1           1            1
```

==All-to-All-V 按这个矩阵一次完成所有 rank 之间的交换：==

```text
rank 0 收到：[a/e0, c/e1]                 -> recv_x_0 [2,4]
rank 1 收到：[a/e2, b/e2, b/e3, c/e3]     -> recv_x_1 [4,4]
```

`R_r` 就是 **Dispatch 后 rank `r` 总共收到多少行 assignment**，也就是发送矩阵第 `r` 列的和：

```text
R_0 = 1 + 1 = 2
R_1 = 3 + 1 = 4

recv_x_r: [R_r,H]
```

这些行来自所有 source ranks，但它们选择的 experts 都位于 rank `r`。移动的是 MoE 层的 activation 副本，不是整个请求或 KV Cache。

### 3.6 具体数字例子

假设：

```text
EP=2, H=8, K=2
rank 0: T_0=3 -> A_0=6
rank 1: T_1=2 -> A_1=4

rank 0 持有 experts 0,1
rank 1 持有 experts 2,3
```

路由结果：

```text
rank 0:
token a -> expert 0,2
token b -> expert 2,3
token c -> expert 1,2

rank 1:
token d -> expert 0,3
token e -> expert 1,3
```

各来源 rank 的发送数量：

```text
rank 0 -> rank 0: 2 个  # a/e0, c/e1
rank 0 -> rank 1: 4 个  # a/e2, b/e2, b/e3, c/e2

rank 1 -> rank 0: 2 个  # d/e0, e/e1
rank 1 -> rank 1: 2 个  # d/e3, e/e3

                 发到 rank 0  发到 rank 1
来源 rank 0           2            4
来源 rank 1           2            2
```

所以：

```text
# 这里表示的就是上述矩阵
rank 0 send_counts = [2,4] 
rank 1 send_counts = [2,2]

Dispatch 后：
rank 0 recv_x shape = [2+2,8] = [4,8] # 发到rank 0 有 4 个assignments
rank 1 recv_x shape = [4+2,8] = [6,8] # 发到rank 1 有 6 个assignments
```

==注意：rank 0 原来有 6 个 assignments，Dispatch 后却计算 4 个；rank 1 原来有 4 个，Dispatch 后计算 6 个。这正是 EP 动态负载不均的来源。==

### 3.7 本地 permute 与 Grouped GEMM

Dispatch 只保证 token 到达正确 rank，收到的数据还需要==按本地 expert 分组：==

```text
recv_x: [R_r,H]
  -> local permute
expert 0: [n_0,H]
expert 1: [n_1,H]
...
```

==Grouped GEMM 把一个 rank 上本轮有 token 输入的多个本地 experts 的矩阵乘，组织到一个或少量 kernel 中执行。==本地 Grouped GEMM 输出：

```text
local_expert_output: [R_r,H]
```

然后 inverse local permute，恢复为 Dispatch 接收顺序。

### 3.8 Combine：第二次 All-to-All-V

Expert 结果需要返回 token 原来的来源 rank，因此 Combine 基本是 Dispatch 的==逆过程==：

```text
expert rank
  -> 根据 dispatch handle / source mapping 打包
  -> All-to-All-V
  -> source rank 收回自己的 K 份 expert 输出
```

在上例中，刚刚是竖着求和，现在是横着求和：

```text
rank 0 最终收回 A_0=6 个结果: [6,8]
rank 1 最终收回 A_1=4 个结果: [4,8]
```

来源 rank 恢复 shape：

```text
rank 0: [6,8] -> [3,2,8] -> top-k 加权求和 -> [3,8]
rank 1: [4,8] -> [2,2,8] -> top-k 加权求和 -> [2,8]
```

因此一个标准 EP MoE 层通常有两次 A2A：

```text
Dispatch A2A：source rank -> expert rank
Combine A2A： expert rank -> source rank
```

### 3.9 AllGather + ReduceScatter 替代路径

并非所有 EP 都使用专用 A2A。通用实现可以这样做：

1. 每 rank 原始 token 为 `[T_r,H]`；
2. AllGather 后，每 rank 看到全局 token `[T_global,H]`；
3. 每 rank 只计算属于本地 experts 的 assignments；
4. 生成对全局 token 的局部 expert contribution；
5. ReduceScatter 求和并把结果切回各来源 rank `[T_r,H]`。

若各 rank 的 `T_r=T_local`：

```text
AllGather:
[T_local,H] -> [P×T_local,H]

Local experts:
[P×T_local,H] -> local partial output [P×T_local,H]

ReduceScatter:
[P×T_local,H] -> [T_local,H]
```

对比：

| 路径 | 通信对象 | 优点 | 缺点 |
|---|---|---|---|
| 专用 A2A | 只发送实际命中远端 expert 的 assignments | 通信稀疏，适合 Wide EP | 动态 shape、backend 和硬件要求高 |
| AG + RS | 收集全局 token，再聚合局部 expert 贡献 | 通用、规则、兼容 hybrid EP/TP | 可能发送许多本地 expert 不需要的数据 |

AG/RS 很适合作为 correctness 与性能 baseline；大规模 EP 通常更依赖 DeepEP 等专用 Dispatch/Combine。

---

## 4. 核心瓶颈

### 4.1 All-to-All 通信

Decode 每轮 token 少，A2A 启动和同步延迟可能高于 expert GEMM。EP 越宽，跨 rank 比例越高；跨节点时还受 RDMA、NIC/GPU 亲和性、NVLink 与网络拓扑影响。

典型症状：

- 单节点快、跨节点骤降；
- Dispatch/Combine 占 MoE 层大部分时间；
- EP size 增大后，每卡 expert 更少但吞吐反而下降；
- 通信 kernel 占用过多 SM，与 expert GEMM 争抢计算资源。

### 4.2 Expert 负载不均

设 rank `r` 上所有本地 experts 收到的 assignments 总数为：

```text
L_r = Σ n_e
```

可用下面的比例粗略观察均衡度：

```text
balancedness = mean(L_r) / max(L_r)
```

越接近 1 越好。MoE 层常由最忙 rank 决定完成时间：

```text
T_moe ≈ T_dispatch + max_r(T_expert,r) + T_combine
```

因此要观察逐层、逐 rank 的最大 token 数，而不是只看全局平均。热点 expert 还会同时增加目标 rank 的网络流量和 GEMM 时间。

---

## 5. 常用优化与前沿技术

| 技术                       | 解决什么                | 核心思路                                                      |
| ------------------------ | ------------------- | --------------------------------------------------------- |
| DeepEP / 专用 A2A          | Dispatch/Combine 慢  | 只传命中 expert 的 token，融合计数、permute、量化与通信                    |
| normal / high-throughput | Prefill 吞吐          | 动态连续布局，减少 padding，形成大 Grouped GEMM                        |
| low-latency / masked     | Decode 延迟           | 固定容量、CUDA Graph 友好，优化小 batch                              |
| FP8 / FP4 MoE            | 权重与通信带宽             | 低精度 expert 权重和 activation，配合 DeepGEMM/FlashInfer 等 kernel |
| EPLB                     | 长期 expert 热点        | 根据统计重排 experts，并复制热门 experts                              |
| Waterfill / LPLB         | 单 batch 仍不均         | shared expert 送到空闲 rank；或在同一 expert 的多个副本间动态分流            |
| TBO / DBO / SBO          | 通信不能隐藏              | 让一个 micro-batch/共享 expert 的计算与另一个 A2A 重叠                  |
| DPA + Wide EP            | KV 与 expert 权重占用高   | Attention 按请求 DP，experts 跨更宽 EP group 分布                  |
| PD 分离                    | Prefill/Decode 目标冲突 | P 侧用高吞吐路径，D 侧用低延迟路径并独立调规模                                 |
| Elastic EP               | 扩缩容和故障域             | 动态改变 ranks、通信组和 expert mapping，利用备份/冗余恢复                  |

DeepEP V2 进一步尝试用统一 `ElasticBuffer` 覆盖高吞吐和低延迟路径，并减少通信 SM 占用、扩大 EP 域。EPLB 通常只改变 logical expert 的物理位置或副本，不改变 Router 的 logical top-k；冗余 experts 会占用 HBM，可能挤压 KV Cache。

---

## 6. vLLM 与 SGLang：最常用配置

### 6.1 参数对照

| 目标 | vLLM | SGLang |
|---|---|---|
| 开启 EP | `--enable-expert-parallel` | `--ep-size N` |
| EP size | 自动为 `TP × DP` | 显式设置 `ep-size` |
| DPA + EP | DP engines 处理不同请求，experts 跨 `TP×DP` | `--dp-size` + `--enable-dp-attention` + `--ep-size` |
| 通用通信 | `allgather_reducescatter` | `--moe-a2a-backend none` |
| 专用 A2A | `--all2all-backend deepep_*` | `--moe-a2a-backend deepep` |
| Prefill/Decode 模式 | high_throughput / low_latency | normal / low_latency / auto |
| EPLB | `--enable-eplb --eplb-config ...` | `--enable-eplb --ep-num-redundant-experts ...` |
| 通信计算重叠 | `--enable-dbo` | `--enable-two-batch-overlap` / SBO |

### 6.2 vLLM 常用 DPA + EP

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel
```

此时：

```text
Attention DP = 8
Attention TP = 1
Expert EP    = TP × DP = 8
```

Prefill/Decode 分离时，可分别选择：

```bash
--all2all-backend deepep_high_throughput  # Prefill
--all2all-backend deepep_low_latency      # Decode
```

EPLB 示例：

```bash
--enable-eplb \
--eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2}' \
--enable-dbo
```

### 6.3 SGLang 常用 DPA + EP

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp-size 8 \
  --dp-size 8 \
  --ep-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --moe-runner-backend deep_gemm
```

此时：

```text
Attention TP = tp / dp = 1
Attention DP = 8
Expert EP    = 8
```

EPLB 与 overlap 常用参数：

```bash
--enable-eplb \
--ep-num-redundant-experts 8 \
--eplb-rebalance-num-iterations 1000 \
--enable-two-batch-overlap
```

当前许多 SGLang 专用 A2A backend 要求 `ep_size = tp_size`；若做 hybrid `ep_size < tp_size`，通常只能使用 `none` 的通用 collective 路径。

### 6.4 根据问题选择

| 问题 | 优先考虑 |
|---|---|
| expert 权重放不下 | 增大 EP；必要时 PP/ETP |
| MLA KV 重复、并发上不去 | DPA + EP |
| 单请求 KV OOM | DCP，不是 EP |
| Prefill MoE 吞吐低 | high-throughput/normal A2A、Grouped GEMM、TBO/DBO |
| Decode ITL 高 | low-latency A2A、masked kernel、CUDA Graph |
| 个别 ranks 过载 | EPLB、冗余 experts、Waterfill/LPLB |
| 跨节点扩展差 | 拓扑感知 placement、RDMA/NVLink、PD/PP |
| Wide EP 故障域过大 | Elastic EP、expert backup |

参考：[vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)、[SGLang Expert Parallelism](https://docs.sglang.io/docs/advanced_features/expert_parallelism)。

> 最终记忆：**MoE 把一个 token 变成 `K` 个 expert assignments；EP 用 Dispatch A2A 把 assignments 发到 expert rank，Grouped GEMM 计算，再用 Combine A2A 把结果送回来源 rank，恢复成 `[T,K,H]` 后沿 `K` 加权求和。**
