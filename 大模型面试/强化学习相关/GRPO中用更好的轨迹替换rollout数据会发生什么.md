# 在 GRPO 的过程中，如果用更好的 traj 替换 rollout 出来的数据，训练会发生什么？

## 问题背景

GRPO 是一种 **on-policy** 的强化学习算法，其核心假设是：每轮更新所用的轨迹数据，必须来自**当前策略**的采样。如果用"更好的轨迹"（如专家数据、更强模型生成的数据）替换 rollout buffer 中的数据，会引发一系列问题。

---

## GRPO 的标准流程

```
当前策略 π_θ  →  采样 G 个输出（rollout）  →  组内归一化计算优势  →  PPO-clip 更新  →  新策略 π_θ'
```

GRPO 的目标函数为：

$$
\mathcal{L}_{\text{GRPO}} = \mathbb{E}\left[\sum_{i=1}^{G} \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i,\ \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)},\ 1-\varepsilon,\ 1+\varepsilon\right)\hat{A}_i\right)\right]
$$

其中 $\pi_{\theta_{\text{old}}}$ 是**生成当前 rollout 数据的策略**，即分母必须与数据来源一致。

---

## 替换数据后会发生什么

### 问题一：重要性采样比率错误（核心问题）

重要性采样比率为：

$$
r_i = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}
$$

当 $o_i$ 来自外部更好的策略 $\pi_{\text{expert}}$ 时，分母应该是 $\pi_{\text{expert}}(o_i|q)$，但代码中仍然用 $\pi_{\theta_{\text{old}}}(o_i|q)$ 计算，导致**重要性采样修正完全错误**，梯度估计有偏。

### 问题二：clip 机制截断梯度为零

"更好的轨迹"对当前策略而言往往是 **out-of-distribution** 的，即：

$$
\pi_{\theta_{\text{old}}}(o_i|q) \approx 0
$$

此时比率 $r_i \to \infty$，PPO 的 clip 机制会将其截断到 $[1-\varepsilon, 1+\varepsilon]$，**梯度被截断为零，模型完全学不到任何东西**。

### 问题三：组内归一化失效

GRPO 的优势估计依赖组内归一化：

$$
\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}
$$

这一归一化的前提是**所有 G 个输出来自同一策略**，组内奖励的均值和方差才有统计意义。混入外部高质量轨迹后，组内奖励分布被破坏，优势估计失去意义。

---

## 正确的解决方案：根据数据来源选择匹配的算法

| 场景 | 正确做法 | 原因 |
|------|---------|------|
| 当前策略采样 + 筛选高分轨迹 | RAFT / ReST（筛选后做 SFT） | 绕开 off-policy 问题 |
| 有外部高质量对比数据 | DPO | 专为 offline 偏好数据设计 |
| on-policy 但想过滤无效数据 | DAPO（动态过滤） | 保持 on-policy 同时提升数据质量 |
| 直接替换 GRPO rollout buffer | ❌ 不可行 | 重要性采样错误 + 组内归一化失效 |

---

## 相关论文与研究

### 1. RAFT / ReST —— 正确的"用好数据训练"方式

> **论文**：*RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment*（Dong et al., 2023）

**核心思想**：不在 RL 循环内替换数据，而是：
1. 用当前策略采样大量轨迹
2. 用 reward model 筛选高分轨迹
3. 用筛选后的数据做 **SFT**（而非 RL）

$$
\mathcal{D}_{\text{filtered}} = \{o_i \mid r(o_i) > \tau\}
$$

SFT 不需要重要性采样，因此完全绕开了 off-policy 问题。

---

### 2. ReST$^{\text{EM}}$ —— EM 框架下的迭代提升

> **论文**：*Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models*（Singh et al., 2023）

将训练理解为 **EM（期望最大化）**：
- **E 步**：用当前策略采样，筛选正确轨迹
- **M 步**：在筛选数据上做 SFT，最大化似然

$$
\theta^* = \arg\max_\theta \mathbb{E}_{o \sim \pi_{\theta_{\text{old}}}}[\mathbf{1}[r(o)=1] \log \pi_\theta(o|q)]
$$

---

### 3. DAPO —— on-policy 框架下的数据质量优化

> **论文**：*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*（ByteDance, 2025）

DAPO 提出 **Dynamic Sampling** 策略，在保持 on-policy 的前提下提升数据质量：
- 过滤掉组内**全对或全错**的 group（优势为 0，无梯度信号）
- 动态补充采样，确保每个 batch 都有有效的学习信号

这是"用更好数据"思想的**正确工程实现**：不是替换数据，而是**过滤无效数据**，始终保持 on-policy。

---

### 4. Dr. GRPO —— 揭示 GRPO 的内在偏差

> **论文**：*Dr. GRPO: Decomposed Reward-Guided Policy Optimization*（2025）

指出原始 GRPO 中组内归一化会引入 **length bias**（长输出倾向于获得更高优势）和 **difficulty bias**（简单题组内方差小导致梯度过大），提出去偏方法。

这与替换数据的问题密切相关：数据来源不同会进一步放大这些偏差，使训练更加不稳定。

---

### 5. DPO —— 有高质量对比数据时的最优选择

> **论文**：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*（Rafailov et al., 2023）

如果已有高质量的对比数据（好轨迹 vs 坏轨迹），**不应使用 on-policy RL**，直接用 DPO：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(o^+|q)}{\pi_{\text{ref}}(o^+|q)} - \beta \log \frac{\pi_\theta(o^-|q)}{\pi_{\text{ref}}(o^-|q)}\right)
$$

DPO 本质上就是"有更好轨迹时该怎么用"这个问题的正确答案。

---

## 总结

**在 GRPO 框架内直接替换 rollout 数据为"更好的轨迹"，会破坏重要性采样的正确性和组内归一化的统计假设，导致训练失效。** 正确的做法是根据数据来源选择匹配的算法框架：

- 想筛选高质量数据 → **RAFT / ReST**（SFT 方式）
- 想过滤无效 on-policy 数据 → **DAPO**（动态采样）
- 已有离线对比数据 → **DPO**
- 坚持用 GRPO → 必须保证数据来自当前策略的 rollout
