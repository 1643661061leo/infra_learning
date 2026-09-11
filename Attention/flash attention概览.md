# Online Softmax、LSE 与 FlashAttention 1–4 演进

> 以下公式默认对 Attention 的**每一行**分别计算，并使用自然指数 `exp` 和自然对数 `log`。这一块强烈建议阅读：https://blog.csdn.net/QcloudCommunity/article/details/161464930

## 1. 一句话结论

**Online Softmax 用==运行中最大值 `m`、指数和 `ℓ`、加权分子 `Z`==逐块计算完整 Softmax；LSE 把 `m` 和 `ℓ` 合并成一个稳定的归一化量。FA1 发明了这套分块 Attention 用法，FA2简化输出递推，FA3主要改变执行流水，FA4进一步减少重新缩放和指数计算的成本。**

```text
普通Attention：先生成完整分数矩阵，再做Softmax
       ↓
FA1：分块 + Online Softmax，避免保存完整矩阵
       ↓
FA2：保存未归一化分子，最后只除一次
       ↓
FA3：公式基本不变，让GEMM与Softmax重叠
       ↓
FA4：有条件地rescale，并让普通计算单元分担exp
```

---

## 2. 从普通 Softmax 开始

对某一个 Query，其与所有 Key 的分数为：

$$
s_j=\frac{q^\top k_j}{\sqrt d}+M_j
$$

- `d`：head dimension；
- `M_j`：mask，允许关注时为 `0`，禁止关注时为 $-\infty$；
- 被 mask 的位置有 $e^{-\infty}=0$，因此概率为 0。

普通 Softmax 定义为：

$$
p_j=\frac{e^{s_j}}{\sum_k e^{s_k}}
$$

Attention 输出是 Value 的加权和：

$$
O=\sum_j p_jv_j
$$

### 为什么要减最大值

若分数很大，直接计算 $e^{s_j}$ 容易溢出。令：

$$
m=\max_j s_j
$$

则：

$$
p_j
=\frac{e^{s_j-m}}{\sum_k e^{s_k-m}}
$$

分子、分母同时乘 $e^{-m}$，结果不变；同时所有 $s_j-m\leq0$，指数不会向上溢出。

定义减去最大值后的指数和：

$$
\ell=\sum_j e^{s_j-m}
$$

于是：

$$
p_j=\frac{e^{s_j-m}}{\ell}
$$

注意：`ℓ` 读作 ell，不是数字 1～

---

## 3. LSE 的定义与推导

LSE 是 **LogSumExp**：先对指数求和，再取对数。

$$
L=\operatorname{LSE}(s)
=\log\sum_j e^{s_j}
$$

把最大值 $m$ 提出来：

$$
\sum_j e^{s_j}
=e^m\sum_j e^{s_j-m}
=e^m\ell
$$

两边取对数：

$$
\boxed{L=m+\log\ell}
$$

所以三者的区别是：

| 符号 | 含义 | 每个 Query 行保存多少 |
|---|---|---:|
| $m$ | 这一行的最大分数 | 1 个数 |
| $\ell$ | 以 $m$ 为基准的指数和 | 1 个数 |
| $L$ | 原始分数的 LogSumExp，即 $m+\log\ell$ | 1 个数 |

LSE 还能直接写出 Softmax：

$$
\boxed{p_j=e^{s_j-L}}
$$

因为 $e^L=\sum_k e^{s_k}$。这正是 backward 能用 LSE 按块重新生成概率、而不必保存完整 `P` 矩阵的原因。

### 一个数值例子

设分数为 `[1000, 1001]`。直接算 $e^{1000}$、$e^{1001}$ 很容易溢出；稳定计算为：

$$
m=1001,\qquad \ell=e^{-1}+1
$$

$$
L=1001+\log(1+e^{-1})\approx1001.313
$$

$$
P\approx[0.269,\ 0.731]
$$

LSE 不是概率，也不是 loss；在这里它只是每行 Softmax 的稳定归一化信息。

---

## 4. Online Softmax 的推导

### 4.1 为什么普通 Softmax 不方便分块

若只看到第一块分数，暂时不知道后面的块是否会出现更大的值，也不知道整行的最终分母。因此，旧块的统计量必须能够在发现新最大值后换到新的基准。

设已处理前 `t-1` 块，保存：

$$
m^{(t-1)},\qquad
\ell^{(t-1)}=\sum_{j\in\text{旧块}}e^{s_j-m^{(t-1)}}
$$

第 `t` 块的分数记为 $S^{(t)}$。新的最大值为：

$$
m^{(t)}=max\left(m^{(t-1)},\operatorname{rowmax}(S^{(t)})\right)
$$

旧指数和原本以 $m^{(t-1)}$ 为基准。换成新基准 $m^{(t)}$：

$$
e^{s_j-m^{(t)}}
=e^{m^{(t-1)}-m^{(t)}}e^{s_j-m^{(t-1)}}
$$

定义重新缩放系数：

$$
\alpha^{(t)}=e^{m^{(t-1)}-m^{(t)}}
$$

因此得到 Online Softmax 的核心递推：

$$
\boxed{
\ell^{(t)}
=\alpha^{(t)}\ell^{(t-1)}
+\operatorname{rowsum}\left(e^{S^{(t)}-m^{(t)}}\right)
}
$$

初始化为：

$$
m^{(0)}=-\infty,\qquad \ell^{(0)}=0
$$

处理完所有块后：

$$
\boxed{L=m^{(T)}+\log\ell^{(T)}}
$$

### 4.2 Attention 输出也能同样合并

定义尚未除以分母的 Value 加权分子：

$$
Z^{(t)}
=\sum_{j\in1\ldots t}e^{s_j-m^{(t)}}v_j
$$

它与 `ℓ` 使用同一个基准，所以递推完全平行：

$$
\boxed{
Z^{(t)}
=\alpha^{(t)}Z^{(t-1)}
+e^{S^{(t)}-m^{(t)}}V^{(t)}
}
$$

最终输出：

$$
\boxed{O=\frac{Z^{(T)}}{\ell^{(T)}}}
$$

这说明每一块只需保留 `m`、`ℓ`、`Z`，不需要保存整行的所有分数或概率。

### 4.3 两个块也可以直接合并

若 A、B 两块分别有 $(m_A,\ell_A,Z_A)$ 和 $(m_B,\ell_B,Z_B)$，令：

$$
m=\max(m_A,m_B)
$$

则：

$$
\ell=e^{m_A-m}\ell_A+e^{m_B-m}\ell_B
$$

$$
Z=e^{m_A-m}Z_A+e^{m_B-m}Z_B
$$

$$
O=Z/\ell,\qquad L=m+\log\ell
$$

逐块扫描，本质上就是反复执行这个合并过程。

---

## 5. FA1 与 FA2 在公式上的区别

### 5.1 FA1：每一块都维护已归一化输出

FA1 保存的是当前已经归一化的 $O^{(t-1)}$。读入新块后：

$$
\boxed{
O^{(t)}=
\frac{
\alpha^{(t)}\ell^{(t-1)}O^{(t-1)}
+e^{S^{(t)}-m^{(t)}}V^{(t)}
}{\ell^{(t)}}
}
$$

为什么旧输出要乘 $\ell^{(t-1)}$？因为 $O^{(t-1)}$ 已经除过旧分母，要先恢复成加权分子，再换基准、加入新块、除以新分母。

### 5.2 FA2：循环中维护未归一化分子

FA2 直接保存 $Z$：

$$
\boxed{
Z^{(t)}
=\alpha^{(t)}Z^{(t-1)}
+e^{S^{(t)}-m^{(t)}}V^{(t)}
}
$$

直到所有块处理结束，才归一化一次：

$$
\boxed{O=Z^{(T)}/\ell^{(T)}}
$$

并把两个 backward 统计量压成一个：

$$
\boxed{L=m^{(T)}+\log\ell^{(T)}}
$$

最简对比：

```text
FA1：状态是 m、ell、已归一化O；每块都有一次除法/归一化
FA2：状态是 m、ell、未归一化Z；最后才归一化，并保存LSE
```

两者的最终数学结果相同。FA2 减少了除法和输出缩放等非矩阵运算；此外它还改变 thread block 与 warp 的工作划分，但那属于执行调度，不属于 Softmax 公式。

---

## 6. FA1 → FA4 的演进

| 版本                         | Online Softmax / LSE 的变化                            | 更主要的优化对象                              |
| -------------------------- | --------------------------------------------------- | ------------------------------------- |
| **[[flash attention v1]]** | 首次把分块 Online Softmax 与 Attention 输出递推结合；逐块维护归一化 `O` | 避免把完整 `S`、`P` 写入 HBM                  |
| **[[flash attention v2]]** | 循环中改存未归一化 `Z`，最后只除一次；保存 `LSE` 而非分别保存 `m`、`ℓ`        | 减少非矩阵运算，提高并行度，减少 warp 通信              |
| **[[flash attention v3]]** | 数学递推基本沿用 FA2，仍更新 `m`、`ℓ`、`Z` 和最终 `LSE`              | 把数据搬运、`QKᵀ`、Softmax、`PV` 异步重叠         |
| **[[flash attention v4]]** | 保留同一分块恒等式，但允许最大值小幅变化时暂不 rescale；部分指数改由多项式近似         | 缓解 Blackwell 上 `exp` 与 rescale 成为的新瓶颈 |

### FA3：改变时间安排，不改变数学目标

FA3 仍使用：

$$
m_t,\quad
\ell_t=\alpha_t\ell_{t-1}+\operatorname{rowsum}(e^{S_t-m_t}),\quad
Z_t=\alpha_tZ_{t-1}+e^{S_t-m_t}V_t
$$

它的关键是把不同块交错执行：

```text
Tensor Core： QKᵀ(t+1) ───────→ PV(t)
普通计算单元：       Softmax(t) ───────→
```

因此 FA3 的主要变化是“何时执行公式”。FP16/BF16 路径仍以完整 Attention 为目标；FP8 路径因量化会产生额外数值误差。

### FA4：不是每次最大值变化都立即 rescale

令 `r` 表示当前实际采用的指数基准，`b` 是当前块带来的候选新最大值。传统做法只要 `b>r`，就把旧 `ℓ` 和 `Z` 乘 $e^{r-b}$。

FA4 引入阈值 $\tau$：

- 若 $b-r>\tau$：把基准更新为 `b`，并 rescale 旧 `ℓ`、`Z`；
- 若 $b-r\leq\tau$：暂时保持基准 `r`，新块直接按 $e^{S-r}$ 累加。

保持基准时：

$$
\ell' = \ell+\operatorname{rowsum}(e^{S-r}),\qquad
Z'=Z+e^{S-r}V
$$

更换基准为 $r'$ 时：

$$
\ell'=e^{r-r'}\ell+\operatorname{rowsum}(e^{S-r'}),
$$

$$
Z'=e^{r-r'}Z+e^{S-r'}V
$$

只要 `ℓ` 与 `Z` 始终使用同一个基准，最终仍有：

$$
O=Z/\ell,\qquad L=r+\log\ell
$$

FA4 实现主要使用 $2^x$。论文常用的阈值是 $\tau_2=\log_2(256)=8$；换成本文的自然指数记法，相当于 $\tau_e=\ln256\approx5.545$。

FA4 还把约 `10%–25%` 的 $2^x$ 交给 FMA 单元用多项式近似，其余仍由专用指数单元计算。因此其数学目标不变，但实现结果不保证逐 bit 相同。

---

## 7. 代码和总结

[[flash_attention_v1_v2代码]]包含：

- `flash_attention_v1_forward`：K/V 块外循环，每轮更新已归一化 `O`；
- `flash_attention_backward_from_lse`：利用 LSE 按块重算 `P`；
- 与普通 Attention 的输出、LSE 和梯度校验。

核心代码对应下面的公式：

```python
m_new = maximum(m_old, rowmax(scores))
alpha = exp(m_old - m_new)
p = exp(scores - m_new)

ell_new = alpha * ell_old + rowsum(p)
out = (alpha * ell_old * out_old + p @ value_block) / ell_new
lse = m_new + log(ell_new)
```
$$
\boxed{
m'=\max(m,\max S),\quad
\ell'=e^{m-m'}\ell+\sum e^{S-m'},\quad
Z'=e^{m-m'}Z+e^{S-m'}V
}
$$

$$
\boxed{O=Z/\ell,\qquad LSE=m+\log\ell,\qquad P=e^{S-LSE}}
$$

```text
FA1：让Online Softmax支持分块Attention
FA2：把逐块归一化改成最后归一化
FA3：让同一套公式异步流水执行
FA4：减少rescale，并给exp增加第二条计算通道
```

---

## 参考资料

- [FlashAttention（NeurIPS 2022）](https://arxiv.org/html/2205.14135)
- [FlashAttention-2（ICLR 2024）](https://arxiv.org/html/2307.08691)
- [FlashAttention-3（NeurIPS 2024）](https://arxiv.org/html/2407.08608)
- [FlashAttention-4（2026）](https://arxiv.org/html/2603.05451)
