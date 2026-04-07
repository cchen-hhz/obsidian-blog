---
{"dg-publish":true,"permalink":"/Social-R1/","title":"Social-R1","tags":[""],"created":"2026-03-26T14:24:22.592+08:00","updated":"2026-04-08T00:14:07.192+08:00","dg-note-properties":{"title":"Social-R1","tags":[""]}}
---






[原文](https://arxiv.org/abs/2603.09249)

一篇过程奖励强化学习，最终遗憾 ICLR 2026 拒稿，也有一些反思。

## 解读

### 前言

我感觉可以从这篇文章说起：

[Small LLMs Do Not Learn a Generalizable Theory of Mind via Reinforcement Learning](https://arxiv.org/abs/2507.15788)

其通过实验验证了基于结果的强化学习无法使得小模型 (7B) 泛化出心智推理能力（其实 7B 也不小了）。

并且多个文章表明心智推理能力是遵循 Scaling Law 的，小模型训练推理总感觉有点魔法。

感觉也能受到 [[DeL-ToM 的一些解读\|DeL-ToM 的一些解读]] 的一些启发，这一篇只将一个评分器用于 inference ，那是否可以运用到 LR 中，做一个过程奖励？

### 正文

总的来说，现有模型做不好复杂推理，其中一个原因是一个被称为 “推理寄生/逻辑翻转” 的概念，即大模型可能通过数据匹配 hacking 了一个答案出来，接着为了迎合 CoT 奖励对着答案写了一堆思考，在一些现有 SOTA 中也能看到这个现象，感觉模型上来先假定哪个答案是对的，然后开始一本正经思考。

那么过程奖励就是必要的，Social-R1 使用了三个奖励组合：

$$R=R_{fmt}(w_oR_{out}+\tau(w_sR_{struct}+w_cR_{content})\cdot R_{len}$$

这里 $R_{fmt}$ 采用了 deepseek 的格式验证（`<think> <answer>`），$R_{out}$ 是结果奖励，$R_{len}$ 是 token 长度限制奖励，采用了一种平滑的处理方式，使得 rollout 长度不在限制内不会突然爆 0 reward。

$R_{struct}$ 是 CoT 结构判别，其将一个思考过程 SIP 分解为 $4$ 个阶段：

1. Encoding Social Cues：提取事实。
2. Interpreting Cues：心理/意念分析。
3. Clarifying Goals：意图分析。
4. Response Generation：给出答案。

主要判别 CoT 是否按这个格式输出东西，那么这种诡异分类是如何判别的呢，直接用的 gpt-4o。。。

$R_{content}$ 是对每个过程的内容判断，防止瞎说，文章用了较大的笔墨来说这个评判机是怎么构建的：

1. 黄金输出：用 gpt-4o，通过传入 prompt 和 answer 生成的 CoT。
2. 接着使用 baseline 在不同 step 上的 checkpoint 的输出，通过另一个教师模型做评分。
3. 大概分了几档评分，接着有选择的做正负配对，训练一个二分类器。

主要贡献就是这么多，最终结论是 7B 模型比 70B 的准确率还要高。

## 锐评

来自 area chair：

1. the "anomalies" in Table 1 where explicit reasoning (CoT) actually degrades performance for several base models raise concerns about the benchmark's reliability.
2. The reliance on o3 and GPT-5 to generate gold trajectories and assign quality scores may result in overfitting of the stylistic preferences and internal biases rather than generalized social intelligence
3. The ToMBench-Hard distribution is skewed toward multi-choice format and specific categories like "Intention", which weakens the evaluation.

你猜怎么着，数据集不公开，代码不公开，有人提出了这个点，但似乎 rebuttal 效果不咋滴（）。

这篇文最大的问题是使用了一个闭源的黑箱教师模型，这和其要证明的理论是矛盾的：既然先有模型，甚至 SOTA 在这一领域做的都很烂，那你凭什么说你用教师模型生成的推理答案是正确的，此时用一个黑箱模型做评判是有损严谨的；另一方面，我在用一个 0样本的大模型对比精细训练过的模型，本身不太公平。

但话说回来，对于一个全世界都做不好的领域，很难完成一个很好很严谨的训练流程，不然就得像 [[DeL-ToM 的一些解读\|DeL-ToM 的一些解读]] 一样用符号语言表示，达到一个0噪声的效果。

比较有意思的是，设计一个奖励就能直接发 ICLR 也是挺厉害的，评分 8422 还不错（）。

不过话说回来，这也是一种尝试，有使用价值，说不定后面我这就要用到这个。




