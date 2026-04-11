---
{"dg-publish":true,"permalink":"/infra/KV-cache 初探/","title":"KV-cache","tags":[""],"created":"2026-04-11T16:20:32.251+08:00","updated":"2026-04-11T16:58:37.619+08:00","dg-note-properties":{"title":"KV-cache","tags":[""]}}
---

## 概述

由于 Decoder-only 的 CasualLM 模型在 inference 时是自回归生成的，在注意力层有大量冗余计算。

我们记：
$$Q=\begin{bmatrix}
q_1\\
q_2\\
q_3\\
\cdots \\
q_n
\end{bmatrix} K=\begin{bmatrix}
k_1\\
k_2\\
k_3\\
\cdots \\
k_n
\end{bmatrix} V=\begin{bmatrix}
v_1\\
v_2\\
v_3\\
\cdots \\
v_n
\end{bmatrix}
$$
此处忽略掉 $\operatorname{softmax}$ 的 $\sqrt{d}$ 。

则：

$$ \begin{align*}
\operatorname{Attn}(Q,K,V) 
&= \operatorname{softmax} \left( \begin{bmatrix} 
q_1 k_1^\top & 0 & \cdots & 0 \\ 
q_2 k_1^\top & q_2 k_2^\top & \cdots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
q_n k_1^\top & q_n k_2^\top & \cdots & q_n k_n^\top 
\end{bmatrix} \right) 
\begin{bmatrix} v_1\\ v_2\\ \vdots\\ v_n \end{bmatrix} \\
\\
\\
&= \begin{bmatrix} 
\operatorname{softmax}(q_1k_1^\top)v_1\\ 
\operatorname{softmax}(q_2k_1^\top)v_1 + \operatorname{softmax}(q_2k_2^\top)v_2\\ 
\vdots \\ 
\operatorname{softmax}(q_nk_1^\top)v_1 + \operatorname{softmax}(q_nk_2^\top)v_2 + \cdots + \operatorname{softmax}(q_nk_n^\top)v_n 
\end{bmatrix}
\end{align*}$$

由于注意力掩码的限制，在从 $n$ 计算 $n+1$ token 时，最终的输出前 $n-1$ 行是完全不变的，只有最后一行的计算。

可以发现 $k,v$ 有一种对称关系，如果缓存了前 $n$ 步的 $k$ 矩阵和 $v$ 矩阵，可与做到只使用 $q_n$ 计算出最后一行，这是 KV cache，缓存了 KV 以及先前的计算结果。

计算一下复杂度，假设隐藏层维度为 $d$，原始 attention 单次预测时间复杂度 $O(n^2d)$ ，引入 KV cache 后，由于 Q 变成一维了，时间复杂度为 $O(nd)$ ，引入额外空间复杂度 $O(nd)$ ，时间上少一个数量级。

并且，原始计算下，需要存储 $N*N$ 的矩阵，现在只需要缓存两个 $N*d$ 的矩阵即可，空间上也有优化，可以说是不用白不用。

## 限制

KV cache 可行依赖于因果律，即先前的 token 不会被后到来的 token 影响，可以发现注意力掩码恰好把后续 token 移除了，满足了因果律。

并且，在 embedding 阶段，一些绝对位置编码会根据当前序列长度做位置编码，这样会破坏原先的 KV 不变性。在做 KV cache 时，我们需要保证先前 token 的 K 向量和 V 向量是不发生改变的，否则就没有缓存的必要了。常见的旋转位置编码 RoPE 不会影响先前 token embedding 的数值，是可以应用的。

## Q cache？

为什么只有 KV cache，没有 Q* Cache？

聚焦于只计算结果最后一行的向量：

$$\operatorname{softmax}(q_nk_1^\top)v_1 + \operatorname{softmax}(q_nk_2^\top)v_2 + \cdots + \operatorname{softmax}(q_nk_n^\top)v_n $$
$\operatorname{softmax}$ 的内部实则为：

$$q_n [k_1^\top,k_2^\top,\cdots,k_n^\top]$$
可以发现最终层的计算中，$K,V$ 是要反复利用的，但是 $Q$ 只需要使用 $q_n$ 。

所以缓存 $Q$ 没有什么意义。