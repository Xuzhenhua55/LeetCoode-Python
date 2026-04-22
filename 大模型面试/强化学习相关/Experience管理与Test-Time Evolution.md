# Experience 管理与 Test-Time Evolution

## 目录

1. [Experience Management（经验管理）](#一experience-management经验管理)
   - [Replay Buffer 基础](#11-replay-buffer-基础)
   - [On-Policy vs Off-Policy 数据管理](#12-on-policy-vs-off-policy-数据管理)
   - [Memory-Augmented 方法](#13-memory-augmented-方法)
2. [Test-Time Evolution（测试时演化）](#二test-time-evolution测试时演化)
   - [Test-Time Compute Scaling](#21-test-time-compute-scaling推理时计算扩展)
   - [Test-Time Training / Adaptation](#22-test-time-training--adaptation推理时训练)
   - [Self-Refinement / Self-Play](#23-self-refinement--self-play自我迭代改进)
3. [三者的关系与工程实践](#三三者的关系与工程实践)

---

## 一、Experience Management（经验管理）

### 1.1 Replay Buffer 基础

#### 什么是 Replay Buffer？

Replay Buffer（经验回放池）是强化学习中用于**存储历史交互数据**的数据结构。智能体与环境交互产生的 $(s, a, r, s')$ 元组被存入 buffer，训练时从中随机采样，打破数据的时序相关性。

```
智能体 π_θ  →  与环境交互  →  产生 (s, a, r, s')  →  存入 Replay Buffer
                                                              ↓
                                                    随机采样 mini-batch
                                                              ↓
                                                    更新策略 π_θ
```

#### 为什么需要 Replay Buffer？

| 问题 | 不用 Buffer | 用 Buffer |
|------|------------|----------|
| 数据相关性 | 连续帧高度相关，梯度方差大 | 随机采样打破相关性 |
| 数据利用率 | 每条数据只用一次 | 数据可被多次复用 |
| 训练稳定性 | 容易震荡 | 更稳定 |
| 适用算法 | On-policy（PPO、GRPO） | Off-policy（DQN、SAC） |

#### 经典 Replay Buffer：DQN 中的均匀采样

> **论文**：*Playing Atari with Deep Reinforcement Learning*（Mnih et al., 2013）

DQN 使用固定大小的循环队列，新数据覆盖最旧数据：

$$
\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^{N}
$$

训练时均匀随机采样：

$$
\text{sample} \sim \text{Uniform}(\mathcal{D})
$$

**缺点**：所有经验被等概率采样，重要的稀有经验被"淹没"。

---

#### 优先经验回放（Prioritized Experience Replay, PER）

> **论文**：*Prioritized Experience Replay*（Schaul et al., 2016）

**核心思想**：TD 误差大的经验更值得学习，应被更频繁地采样。

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
$$

其中：
- $p_i = |\delta_i| + \epsilon$，$\delta_i$ 是 TD 误差
- $\alpha$ 控制优先级程度（$\alpha=0$ 退化为均匀采样）

**重要性采样修正**（防止引入偏差）：

$$
w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta
$$

$\beta$ 从 0 退火到 1，训练后期完全修正偏差。

**工程实现**：使用 **Sum Tree** 数据结构，$O(\log N)$ 完成采样和更新。

```
Sum Tree 示意：
         42
        /  \
      29    13
     /  \  /  \
    13  16 3  10
```

---

#### Hindsight Experience Replay（HER）

> **论文**：*Hindsight Experience Replay*（Andrychowicz et al., 2017）

**核心思想**：即使没有达到目标，也可以把"实际到达的状态"当作虚构目标，从失败经验中学习。

**场景**：稀疏奖励任务（如机械臂抓取），大多数 episode 奖励为 0。

**做法**：对于轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$，除了存储原始目标 $g$，还额外存储以 $s_T$（实际终止状态）为目标的虚构经验：

$$
r'(s, a, g') = \mathbf{1}[s' = g']
$$

**效果**：将稀疏奖励问题转化为密集奖励，显著提升样本效率。

---

### 1.2 On-Policy vs Off-Policy 数据管理

#### 核心区别

| 维度 | On-Policy | Off-Policy |
|------|-----------|------------|
| 数据来源 | 必须来自当前策略 $\pi_\theta$ | 可来自任意策略 |
| 数据复用 | 每条数据只用一次（或少数几次） | 可大量复用 |
| 代表算法 | PPO、GRPO、A3C | DQN、SAC、TD3 |
| 修正方式 | 无需修正 | 需要重要性采样修正 |
| 样本效率 | 低 | 高 |
| 训练稳定性 | 高 | 相对低 |

#### On-Policy 的数据管理：PPO 的 Mini-Epoch

PPO 允许对同一批 rollout 数据做 $K$ 次梯度更新（mini-epoch），通过 clip 机制限制策略偏移：

$$
L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t A_t,\ \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t\right)\right]
$$

其中 $r_t = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$

**数据生命周期**：

```
rollout（采样 T 步）→ 计算 GAE 优势 → K 次 mini-batch 更新 → 丢弃数据 → 重新 rollout
```

#### Off-Policy 的数据管理：重要性采样

当数据来自旧策略 $\mu$ 而非当前策略 $\pi$ 时，需要修正：

$$
\mathbb{E}_{a \sim \pi}[f(a)] = \mathbb{E}_{a \sim \mu}\left[\frac{\pi(a)}{\mu(a)} f(a)\right]
$$

**问题**：当 $\pi$ 和 $\mu$ 差距过大时，重要性权重方差爆炸，训练不稳定。

**解决方案**：
- **截断重要性权重**（V-trace，IMPALA）：$\bar{\rho}_t = \min(\bar{\rho}, \rho_t)$
- **限制 buffer 中数据的"年龄"**：只保留最近 $K$ 个 epoch 的数据

#### 大模型 RL 中的特殊挑战

大模型 RL（如 RLHF）中，一条"经验"是一个完整的文本序列，而非单步 $(s, a, r, s')$：

```
问题 q → 生成完整回答 o = (t_1, t_2, ..., t_L) → 获得奖励 r(q, o)
```

**挑战**：
1. 序列长度不固定，难以批量存储
2. Token 级别的 credit assignment 困难
3. 奖励稀疏（只有序列末尾有奖励）

**GRPO 的解法**：对同一问题采样 $G$ 个回答，用组内相对奖励替代绝对奖励，完全绕开 Critic 网络：

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}
$$

---

### 1.3 Memory-Augmented 方法

#### 外部记忆（External Memory）

给模型配备可读写的外部存储，突破参数记忆的限制。

**Neural Turing Machine（NTM）**

> **论文**：*Neural Turing Machines*（Graves et al., 2014）

通过可微分的注意力机制读写外部记忆矩阵 $M \in \mathbb{R}^{N \times W}$：

$$
r_t = \sum_i w_t(i) M_t(i) \quad \text{（读操作）}
$$

$$
M_t(i) = M_{t-1}(i)(1 - w_t(i) e_t) + w_t(i) a_t \quad \text{（写操作）}
$$

**Differentiable Neural Computer（DNC）**

> **论文**：*Hybrid computing using a neural network with dynamic external memory*（Graves et al., 2016）

在 NTM 基础上增加了**动态内存分配**和**时序链接矩阵**，支持更复杂的读写模式。

#### RAG（Retrieval-Augmented Generation）作为 Experience 管理

> **论文**：*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*（Lewis et al., 2020）

将外部知识库作为"经验池"，推理时检索相关经验：

```
输入 query → 检索器（Dense Retrieval）→ 召回 Top-K 文档 → 拼接到 prompt → 生成答案
```

**在 RL 中的应用**：将历史成功轨迹存入向量数据库，推理时检索相似问题的解法作为 few-shot 示例。

#### Episodic Memory in RL

> **论文**：*Neural Episodic Control*（Pritzel et al., 2017）

将每个 episode 的 $(s, Q)$ 对存入可微分的字典，推理时用 $k$-NN 检索相似状态的 Q 值：

$$
Q^{\text{NEC}}(s, a) = \frac{\sum_i K(h, h_i) V_i}{\sum_i K(h, h_i)}
$$

其中 $K$ 是核函数（如 RBF 核），$h$ 是状态的嵌入表示。

**优势**：能在极少样本下快速学习（one-shot learning），因为直接记忆了成功经验。

---

## 二、Test-Time Evolution（测试时演化）

### 2.1 Test-Time Compute Scaling（推理时计算扩展）

#### 核心思想

**训练时 Scaling**（更多参数、更多数据）之外，**推理时 Scaling**（更多计算）也能显著提升性能。

> **论文**：*Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters*（Snell et al., 2024）

**关键发现**：在某些任务上，一个小模型用更多推理计算，可以超过大模型的单次推理。

$$
\text{性能} \propto f(\text{模型参数}, \text{推理计算量})
$$

---

#### Best-of-N（BoN）采样

最简单的推理时扩展方法：生成 $N$ 个候选答案，用 reward model 选最好的。

$$
o^* = \arg\max_{o_i \sim \pi_\theta} r(q, o_i), \quad i = 1, \ldots, N
$$

**性能上界**（Pass@N）：

$$
\text{Pass@N} = 1 - (1 - p)^N
$$

其中 $p$ 是单次采样的正确率。

**缺点**：
- 计算量随 $N$ 线性增长
- 依赖 reward model 的质量，RM 有偏则选出的答案也有偏

---

#### Beam Search 与 Tree Search

**标准 Beam Search**：维护 $k$ 条最优路径，每步扩展并保留 top-$k$：

```
Step 1: [A, B, C]  →  扩展  →  [AA, AB, AC, BA, BB, BC, CA, CB, CC]
                    →  保留 top-3  →  [AA, AB, BA]
```

**MCTS（蒙特卡洛树搜索）在 LLM 中的应用**

> **论文**：*Scaling LLM Test-Time Compute with Inference-Time Monte Carlo Tree Search*（2024）

将 LLM 的推理过程建模为树搜索：

```
根节点（问题 q）
    ├── 思路 A（UCB 分数高）
    │     ├── 步骤 A1 → 模拟到终止 → 奖励 0.8
    │     └── 步骤 A2 → 模拟到终止 → 奖励 0.3
    └── 思路 B
          └── 步骤 B1 → 模拟到终止 → 奖励 0.9  ← 最优路径
```

**UCB 选择公式**：

$$
\text{UCB}(v) = \frac{Q(v)}{N(v)} + c\sqrt{\frac{\ln N(\text{parent}(v))}{N(v)}}
$$

**AlphaCode 2 / AlphaProof** 就是将 MCTS 与 LLM 结合用于代码和数学推理。

---

#### Process Reward Model（PRM）vs Outcome Reward Model（ORM）

推理时扩展的关键是**如何评分**：

| 类型 | 评分粒度 | 优点 | 缺点 |
|------|---------|------|------|
| ORM（结果奖励） | 整个答案 | 简单，易训练 | 无法指导中间步骤 |
| PRM（过程奖励） | 每个推理步骤 | 可以在搜索中剪枝 | 需要步骤级标注，成本高 |

> **论文**：*Let's Verify Step by Step*（Lightman et al., 2023，OpenAI）

PRM 在 MATH 数据集上显著优于 ORM，尤其在 Best-of-N 场景下。

**PRM 在 MCTS 中的作用**：

```
每扩展一步 → PRM 评分当前步骤 → 低分步骤提前剪枝 → 计算集中在有希望的路径
```

---

#### Compute-Optimal Inference

> **论文**：*Scaling LLM Test-Time Compute Optimally*（Snell et al., 2024）

**核心结论**：不同难度的问题，最优的推理计算分配策略不同：

- **简单问题**：Best-of-N 就够了，不需要复杂搜索
- **中等问题**：Beam Search + PRM 效果最好
- **困难问题**：MCTS 才能充分探索解空间

**自适应计算分配**：根据问题难度动态分配推理计算量，而非对所有问题用相同策略。

---

### 2.2 Test-Time Training / Adaptation（推理时训练）

#### Test-Time Training（TTT）

> **论文**：*Test-Time Training with Self-Supervision for Generalization under Distribution Shifts*（Sun et al., 2020）

**核心思想**：在测试时，用测试样本本身做自监督学习，让模型快速适应新分布。

```
训练阶段：主任务（分类）+ 辅助任务（旋转预测）→ 共享特征提取器
测试阶段：用测试样本做辅助任务 → 更新特征提取器 → 再做主任务预测
```

#### TTT 在大模型中的应用

> **论文**：*Test-Time Training on Nearest Neighbors for Large Language Models*（Shi et al., 2023）

对于每个测试样本，检索训练集中最相似的样本，在这些样本上做短暂微调：

```
测试输入 x → 检索 Top-K 相似训练样本 → 在这 K 个样本上微调 → 预测 x
```

**优点**：无需修改模型架构，即插即用。
**缺点**：每个测试样本都需要微调，推理延迟大。

#### TTT 用于长上下文（ARC 竞赛）

> **论文**：*Scaling Test-Time Compute with Open Models*（2024）

在 ARC（Abstraction and Reasoning Corpus）任务上，TTT 取得了显著效果：

1. 从测试任务的 few-shot 示例中提取规律
2. 用这些示例微调模型（仅需几步梯度更新）
3. 用微调后的模型预测测试答案

**关键发现**：TTT + 推理时搜索的组合，在 ARC 上达到了接近人类的水平。

---

#### In-Context Learning 作为隐式 TTT

> **论文**：*In-Context Learning through the Bayesian Prism*（Xie et al., 2022）

ICL 可以被理解为一种**无梯度的隐式适应**：

$$
P(y|x, \text{context}) = \int P(y|x, \theta) P(\theta|\text{context}) d\theta
$$

模型通过 attention 机制在 context 中"检索"相关模式，等价于在推理时做了隐式的贝叶斯更新。

---

### 2.3 Self-Refinement / Self-Play（自我迭代改进）

#### Self-Refinement（自我精炼）

> **论文**：*Self-Refine: Iterative Refinement with Self-Feedback*（Madaan et al., 2023）

**核心思想**：模型生成初始答案 → 自我评估 → 根据反馈修改 → 循环迭代。

```
初始答案 y_0
    ↓
自我评估：fb_1 = Feedback(y_0)  ← 同一个模型
    ↓
精炼：y_1 = Refine(y_0, fb_1)
    ↓
自我评估：fb_2 = Feedback(y_1)
    ↓
精炼：y_2 = Refine(y_1, fb_2)
    ↓
... 直到满足停止条件
```

**数学形式**：

$$
y_{t+1} = \pi_\theta(y_t, \text{Feedback}(y_t, q), q)
$$

**效果**：在代码生成、数学推理、文本摘要等任务上，迭代 3-5 次后性能显著提升。

**局限性**：模型无法发现自己不知道的错误（"不知道自己不知道"）。

---

#### Self-Play（自我博弈）

> **论文**：*Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models*（Chen et al., 2024，SPIN）

**SPIN（Self-Play Fine-Tuning）**：

- **玩家**：当前策略 $\pi_\theta$（主玩家）vs 上一轮策略 $\pi_{\theta_{t-1}}$（对手）
- **目标**：让主玩家的输出比对手更接近真实数据分布

$$
\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}}\left[\ell\left(\lambda \log \frac{\pi_\theta(y|x)}{\pi_{\theta_{t-1}}(y|x)} - \lambda \log \frac{\pi_\theta(\tilde{y}|x)}{\pi_{\theta_{t-1}}(\tilde{y}|x)}\right)\right]
$$

其中 $y$ 是真实答案，$\tilde{y} \sim \pi_{\theta_{t-1}}$ 是对手生成的答案。

**直觉**：每轮训练后，对手变强，主玩家需要继续提升才能超越对手，形成持续进化的动力。

---

#### Constitutional AI（CAI）中的自我改进

> **论文**：*Constitutional AI: Harmlessness from AI Feedback*（Bai et al., 2022，Anthropic）

**两阶段自我改进**：

**阶段一：SL-CAI（监督学习）**
1. 模型生成有害回答
2. 模型根据"宪法原则"批判自己的回答
3. 模型根据批判修改回答
4. 用修改后的回答做 SFT

**阶段二：RL-CAI（强化学习）**
1. 用 AI 反馈（而非人类反馈）训练 Preference Model
2. 用 PM 做 RLHF

$$
\text{AI Feedback}: \text{PM}(y_1 \succ y_2 | x) = \sigma\left(\text{PM}(x, y_1) - \text{PM}(x, y_2)\right)
$$

---

#### STaR（Self-Taught Reasoner）

> **论文**：*STaR: Bootstrapping Reasoning With Reasoning*（Zelikman et al., 2022）

**核心思想**：用模型自己生成的正确推理链来训练自己，形成自举（bootstrapping）。

```
迭代流程：
1. 用当前模型生成推理链 + 答案
2. 筛选答案正确的推理链
3. 对于答案错误的，提供正确答案让模型"合理化"推理过程（rationalization）
4. 用筛选/合理化后的推理链做 SFT
5. 重复
```

**与 RLHF 的区别**：STaR 不需要 reward model，直接用答案正确性作为信号。

---

#### Iterative Self-Play with RL（DeepSeek-R1 的思路）

> **论文**：*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*（2025）

DeepSeek-R1 的训练本质上是一种**迭代自我进化**：

```
阶段 1：冷启动 SFT（少量 CoT 数据）
    ↓
阶段 2：GRPO 强化学习（数学/代码任务，规则奖励）
    ↓
阶段 3：拒绝采样 SFT（用 RL 模型生成高质量数据）
    ↓
阶段 4：再次 GRPO（加入通用任务）
    ↓
最终模型 DeepSeek-R1
```

**关键洞察**：模型在 RL 阶段自发涌现出"思考更长时间"的行为（extended thinking），这是一种 test-time compute scaling 的内化形式。

---

## 三、三者的关系与工程实践

### 3.1 统一视角

```
┌─────────────────────────────────────────────────────────┐
│                    大模型能力提升路径                      │
├─────────────────┬───────────────────┬───────────────────┤
│  Experience 管理 │  Test-Time Compute │  Self-Evolution   │
│                 │                   │                   │
│ • Replay Buffer │ • Best-of-N       │ • Self-Refine     │
│ • PER           │ • MCTS + PRM      │ • SPIN            │
│ • HER           │ • Beam Search     │ • STaR            │
│ • RAG           │ • Adaptive Compute│ • DeepSeek-R1     │
│                 │                   │                   │
│ 训练时数据管理   │ 推理时计算扩展     │ 迭代自我提升      │
└─────────────────┴───────────────────┴───────────────────┘
```

### 3.2 关键权衡

| 方法 | 计算开销 | 数据需求 | 适用场景 |
|------|---------|---------|---------|
| Replay Buffer（PER） | 低（训练时） | 需要大量交互 | 游戏、机器人控制 |
| Best-of-N | 中（N 倍推理） | 需要 RM | 对话、代码生成 |
| MCTS + PRM | 高（树搜索） | 需要步骤级标注 | 数学推理 |
| TTT | 高（每样本微调） | 需要相关训练数据 | 分布偏移场景 |
| Self-Refine | 中（多轮推理） | 无需额外数据 | 代码、写作 |
| STaR / SPIN | 中（迭代训练） | 需要可验证答案 | 推理任务 |

### 3.3 面试高频考点总结

**Q1：PER 为什么需要重要性采样修正？**

> 因为 PER 改变了采样分布（高 TD 误差的样本被过采样），如果不修正，梯度估计会有偏。重要性权重 $w_i \propto \frac{1}{P(i)}$ 抵消了采样偏差。

**Q2：On-policy 方法为什么不能直接用 Replay Buffer？**

> On-policy 方法（如 PPO）的目标函数假设数据来自当前策略，如果用旧策略的数据，重要性采样比率 $\frac{\pi_\theta}{\pi_{\text{old}}}$ 会偏离 1，clip 机制会截断梯度，导致学不到东西。

**Q3：Test-Time Compute Scaling 的本质是什么？**

> 本质是用**推理时的计算换取更高的答案质量**。通过搜索（MCTS、Beam Search）或采样（Best-of-N）探索更大的解空间，再用 reward model 或 PRM 选出最优解。

**Q4：Self-Refine 和 RLHF 的区别？**

> Self-Refine 是**推理时**的迭代改进，不更新模型参数，依赖模型自身的评估能力；RLHF 是**训练时**的改进，通过梯度更新改变模型参数，依赖 reward model 的反馈。

**Q5：为什么 DeepSeek-R1 的 extended thinking 是 test-time compute scaling 的内化？**

> 模型通过 RL 学会了"遇到难题时生成更长的思维链"，这等价于在推理时分配更多计算（更多 token）。区别在于这种行为是模型**自主学会**的，而非外部强制的搜索策略。

---

## 参考文献

1. Mnih et al. (2013). *Playing Atari with Deep Reinforcement Learning*. DeepMind.
2. Schaul et al. (2016). *Prioritized Experience Replay*. ICLR 2016.
3. Andrychowicz et al. (2017). *Hindsight Experience Replay*. NeurIPS 2017.
4. Graves et al. (2014). *Neural Turing Machines*. arXiv.
5. Graves et al. (2016). *Hybrid computing using a neural network with dynamic external memory*. Nature.
6. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
7. Pritzel et al. (2017). *Neural Episodic Control*. ICML 2017.
8. Snell et al. (2024). *Scaling LLM Test-Time Compute Optimally*. arXiv.
9. Lightman et al. (2023). *Let's Verify Step by Step*. OpenAI.
10. Sun et al. (2020). *Test-Time Training with Self-Supervision*. ICML 2020.
11. Madaan et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023.
12. Chen et al. (2024). *Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models*. ICML 2024.
13. Bai et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. Anthropic.
14. Zelikman et al. (2022). *STaR: Bootstrapping Reasoning With Reasoning*. NeurIPS 2022.
15. DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*.
