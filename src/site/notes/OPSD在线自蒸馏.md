---
{"dg-publish":true,"permalink":"/OPSD在线自蒸馏/","title":"OPSD","tags":[""],"created":"2026-04-16T14:35:56.118+08:00","updated":"2026-04-16T23:35:01.444+08:00","dg-note-properties":{"title":"OPSD","tags":[""]}}
---

 On-Policy Self-Distillation

~~写的像综述~~
## Distillation

知识蒸馏应该可以和 RLVR 做一个比较，传统的 RL 训练是轨迹级别的，并将 advantage 同等应用到每一个 token 上，这就造成了 稀疏奖励 的问题，并且梯度消失问题也很明显（如果一批数据奖励没区别，advantage 直接变成 $0$，梯度直接一脚踢死）。

知识蒸馏常用的为 teacher-forcing ，更像一种在教师分布上的 SFT，但知识蒸馏提供了更广泛的 logits 用于计算 KL，教师的 token logits 比纯静态数据的独热码往往信息量更大。

话虽如此，这就涉及到常说的正反 KL 的问题，原因在于教师 token logits 往往不满足一个单峰分布。在一个双高 token logits 分布下，正向 KL 就会简单得取平均，两个方向的 rollout 取一个均值；反向 KL 直接抛弃其中一个 rollout，多样性也不足。

或者使用正反手 `Jensen-Shannon divergence` ，即 $\beta D_{KL}(A||B)+(1-\beta)D_{KL}(B||A)$ ，但我没试验过效果（消融实验说直接正向 KL 更好，ee）。

## On-policy / Off-policy

先抛开自蒸馏的概念，简要说离线蒸馏和在线蒸馏。

通常的蒸馏中，会取一个更强大的闭源模型刷数据，此时的蒸馏是 off-policy 的，因为我们固定了采样策略，而待训练的模型策略是一直更新的。

Off-policy distillation 有一些自生的问题，其会限制模型的性能，也会招致一些 hacking 现象。正如上文得 off-policy distillation，其本质还是一个离线 SFT，SFT 得共性问题很难避免。

于是有人提出了 On-policy，应该追溯到 24 年 [这篇](https://arxiv.org/abs/2306.13649)，总的来说，看是否为 on-policy 的关键在于你取教师的预测还是学生的预测做 teacher-forcing，前者由于你要优化学生分布，二者不一致时 off-policy 的，而后者往往是 on-policy 的。

## Self-distillation

蒸馏依赖于一个教师模型，论文认为一个模型是有能力自己教自己的，像自学一样，于是提出在线自蒸馏。

![source/opsd.png](/img/user/source/opsd.png)

目标为最小化散度：

$$\mathbb{E}_{(x,y^*)\sim S}[\mathbb{E}_{\hat{y}\sim p_S(\cdot | x)}[D(p_T \| p_S) (\hat{y} \mid x)]]$$

算法的流程为：

- 对于一个问题 $(x,y^*)$ 学生拿到 $x$ ，生成一条轨迹 $\hat{y} = (\hat{y}_1, \dots, \hat{y}_{|\hat{y}|}) \sim p_S(\cdot | x)$
- 对于该轨迹，教师学生同时做逐 token logits 预测，得到 $P_S(y_n| x,\hat{y}_{<n})$  和 $P_T(y_n| x,y^*,\hat{y}_{<n})$ ，注意到教师模型同时获得了最终一个期待输出。
- 计算散度 $\ell(x, y^*) \leftarrow D(p_T \| p_S) (\hat{y} \mid x) = \frac{1}{|\hat{y}|} \sum_{n=1}^{|\hat{y}|} D(p_T(\cdot \mid \hat{y}_{<n}, x, y^*)) \| p_S(\cdot \mid \hat{y}_{<n}, x)\rangle$
- 一个 batch 的散度求均值作为损失。

这里 $D$ 可以是任何一个散度符号，例如 $D_{KL}$ 。

### 改进

一个严重的问题是 $D_{KL}$ 通常需要遍历一整个词汇集，算一遍散度计算量爆炸。

从 RL 角度来讲，可以使用一个经典的蒸馏式奖励：

$$A_n(x,\hat{y})=\log_{p_T}(\hat{y_n})-\log_{p_S}(\hat{y_n})$$
$$\ell(x, y^*)=-\mathbb{E}[\sum_{n=1}^{|\hat{y}|}A_n\times \log_{P_S}(\hat{y_n}|x,y_{<n})]$$
在消融实验中，该简化方法相比全词表散度降低了 2-3 个点，大致能说明全词表比较是有其用处的。

总之，需要意识到的是，这个奖励信号是 token 级别的奖励，相比序列级奖励有更密集的更新。

## 可能的改进

- clip：作者发现风格词汇会影响教师和学生的分布，那就散度损失加个 clip。
- 结果奖励：可以发现对最终轨迹的结果是没有约束的，我们假定了教师模型偷看了结果后能引导学生模型给出正确答案，但这是存疑的。
- 需要逐步提升上限：对于很复杂的问题，往往当前的教师模型看了答案也做不了。

## 后话

刚接触这个领域，很多前置概念都是看的网上的一些总结，没有很仔细的调研，一些事实错误还请指出。

预告：下一个是 OPSD。。。








