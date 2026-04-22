# DPO 原理与公式推导

---

## 一、Bradley-Terry 模型

Bradley-Terry（BT）模型由 R.A. Bradley 和 M.E. Terry 在 1952 年提出，是一种用于描述两个元素之间比较概率的统计模型，常用于排名系统、体育比赛胜负预测等场景。其核心公式如下：

$$
P(i \succ j) = \frac{\alpha_i}{\alpha_i + \alpha_j}
$$

- $P(i \succ j)$：表示第 $i$ 个元素战胜第 $j$ 个元素的概率。
- $\alpha_i$：表示第 $i$ 个元素的实力（能力值），通常是一个正实数。

> 📌 **数学注解**：这个公式的直觉很简单——两个人比赛，各自有一个"实力值"，胜率就是自己实力占总实力的比例。例如 $\alpha_A = 2, \alpha_B = 1$，则 A 胜 B 的概率为 $\frac{2}{2+1} = \frac{2}{3}$。

---

### 1.1 问题实例

对比较关系进行建模，问 B 战胜 C 的概率有多大？

| 对战  | 胜 | 负 |
|-------|----|----|
| A 对 B | 8  | 4  |
| A 对 C | 3  | 5  |

**推导步骤：**

1. 根据定义，$\alpha_i$ 战胜 $\alpha_j$ 的公式如下：

$$
P(A \succ B) = \frac{\alpha_A}{\alpha_A + \alpha_B}, \quad P(B \succ A) = \frac{\alpha_B}{\alpha_A + \alpha_B}
$$

$$
P(A \succ C) = \frac{\alpha_A}{\alpha_A + \alpha_C}, \quad P(C \succ A) = \frac{\alpha_C}{\alpha_A + \alpha_C}
$$

2. 整个数据的联合概率（似然函数）就是所有比赛胜负结果的联合发生概率：

$$
L = \left(\frac{\alpha_A}{\alpha_A+\alpha_B}\right)^8 \times \left(\frac{\alpha_B}{\alpha_A+\alpha_B}\right)^4 \times \left(\frac{\alpha_A}{\alpha_A+\alpha_C}\right)^3 \times \left(\frac{\alpha_C}{\alpha_A+\alpha_C}\right)^5
$$

> 📌 **数学注解**：联合概率 = 每场比赛概率的乘积（假设各场比赛独立）。A 赢了 8 次就乘 8 次 $P(A \succ B)$，B 赢了 4 次就乘 4 次 $P(B \succ A)$，以此类推。

3. 取对数则可以写出对数最大似然估计：

$$
\ln L = 8\ln\left(\frac{\alpha_A}{\alpha_A+\alpha_B}\right) + 4\ln\left(\frac{\alpha_B}{\alpha_A+\alpha_B}\right) + 3\ln\left(\frac{\alpha_A}{\alpha_A+\alpha_C}\right) + 5\ln\left(\frac{\alpha_C}{\alpha_A+\alpha_C}\right)
$$

> 📌 **数学注解**：取对数的目的是将连乘变成连加，方便求导。$\ln(a \times b) = \ln a + \ln b$，这是对数的基本性质。

4. 为了最大化似然函数，我们对 $\alpha_A$、$\alpha_B$、$\alpha_C$ 进行求偏导得到极值：

$$
\alpha_A : \alpha_B : \alpha_C = 1 : \frac{1}{2} : \frac{5}{3}
$$

5. 偏导以 $\alpha_A$ 为例，为了找到极值点，令 $\frac{\partial \ln L}{\partial \alpha_A} = 0$：

$$
8 \cdot \frac{\alpha_B}{\alpha_A(\alpha_A+\alpha_B)} - 4 \cdot \frac{1}{\alpha_A+\alpha_B} + 3 \cdot \frac{\alpha_C}{\alpha_A(\alpha_A+\alpha_C)} - 5 \cdot \frac{1}{\alpha_A+\alpha_C} = 0
$$

> 📌 **数学注解**：对 $\ln\left(\frac{\alpha_A}{\alpha_A+\alpha_B}\right) = \ln\alpha_A - \ln(\alpha_A+\alpha_B)$ 求关于 $\alpha_A$ 的偏导，利用 $\frac{d}{dx}\ln f(x) = \frac{f'(x)}{f(x)}$ 即可得到上式。

6. 最后，B 战胜 C 的概率是：

$$
P(B \succ C) = \frac{\alpha_B}{\alpha_B + \alpha_C} \approx 0.23
$$

---

### 1.2 BT 损失

#### （1）似然函数和对数似然函数

给定观测数据 $D = \{(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)\}$，其中每一对 $(x_i, y_i)$ 表示在第 $i$ 次比赛中对象 $x_i$ 胜了对象 $y_i$。

1. **似然函数**：

$$
L(\alpha) = \prod_{i=1}^{N} \frac{\alpha_{x_i}}{\alpha_{x_i} + \alpha_{y_i}}
$$

2. **对数似然函数**：

$$
\log L(\alpha) = \sum_{i=1}^{N} \log\left(\frac{\alpha_{x_i}}{\alpha_{x_i}+\alpha_{y_i}}\right) = \sum_{i=1}^{N} \left[\log \alpha_{x_i} - \log(\alpha_{x_i} + \alpha_{y_i})\right]
$$

---

#### （2）损失函数推导

在机器学习中，通常我们要最大化对数似然函数，等价于最小化负对数似然函数（Negative Log-Likelihood，NLL）。所以对应的损失函数（loss function）为：

$$
\text{Loss} = -\log L(\alpha) = -\sum_{i=1}^{N} \log\left(\frac{\alpha_{x_i}}{\alpha_{x_i}+\alpha_{y_i}}\right) = -N \cdot \mathbb{E}_{(\alpha_x, \alpha_y) \sim D}\left[\ln\left(\frac{\alpha_x}{\alpha_x + \alpha_y}\right)\right]
$$

> 📌 **数学注解**：$\mathbb{E}_{(\alpha_x,\alpha_y)\sim D}[\cdot]$ 表示对数据集 $D$ 中所有样本取期望（平均），即 $\frac{1}{N}\sum_{i=1}^N[\cdot]$，所以前面乘了 $N$。

---

#### （3）参数化

为了方便优化以及 $\alpha$ 可能为负数，常将参数 $\alpha$ 化为指数形式：$e^\alpha$，则：

$$
\text{Loss} = -\mathbb{E}_{(\alpha_x,\alpha_y)\sim D}\left[\ln\left(\frac{e^{\alpha_x}}{e^{\alpha_x}+e^{\alpha_y}}\right)\right]
$$

利用恒等式：

$$
\frac{\exp(a)}{\exp(a)+\exp(b)} = \frac{1}{1+\exp(b-a)}
$$

> 📌 **数学注解**：分子分母同除以 $\exp(a)$ 即可得到右边的形式。

继续化简：

$$
= -\mathbb{E}_{(\alpha_x,\alpha_y)\sim D}\left[\ln\left(\frac{1}{1+e^{\alpha_y - \alpha_x}}\right)\right]
= -\mathbb{E}_{(\alpha_x,\alpha_y)\sim D}\left[\ln\left(\frac{1}{1+e^{-(\alpha_x-\alpha_y)}}\right)\right]
$$

$$
= \mathbb{E}_{(\alpha_x,\alpha_y)\sim D}\left[\ln\sigma(\alpha_x - \alpha_y)\right]
$$

> 📌 **数学注解（logsigmoid）**：
> $$\text{logsigmoid}(x) = \log\left(\frac{1}{1+e^{-x}}\right) = -\log(1+e^{-x}) = \ln\left(\frac{1}{1+e^{-x}}\right)$$
> 这里 $\log$ 对应数学中的 $\ln$（自然对数）。sigmoid 函数 $\sigma(x) = \frac{1}{1+e^{-x}}$，所以 $\ln\sigma(x) = \ln\frac{1}{1+e^{-x}}$。

---

### 1.3 RM Loss

在强化学习中，大模型的输入 prompt 是 $x$，回答 $y$。回答 $y$ 的好坏（实力得分）由 Reward 模型评估。那通过 BT 模型建模，其中 $r(x, y)$ 表示 RM 输出得分，可能为负数，加上指数函数：

$$
P(y_w \succ y_l) = \frac{r(x, y_w)}{r(x, y_w) + r(x, y_l)} = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}
$$

通过综合上述，得到 reward model 损失函数：

$$
\text{Loss} = -\ln P(y_w \succ y_l)
= -\ln\sigma(r(x, y_w) - r(x, y_l))
= -\mathbb{E}_{(x, y_w, y_l)\sim D}\left[\ln\sigma(r(x, y_w) - r(x, y_l))\right]
$$

> 📌 **数学注解**：这里 $y_w$ 是 winner（偏好回答），$y_l$ 是 loser（拒绝回答）。Loss 的含义是：让模型对好回答的打分比坏回答高，差值越大，$\sigma$ 越接近 1，$-\ln\sigma$ 越接近 0，loss 越小。

---

## 二、DPO

### 2.1 训练目标

1. 奖励函数：$r(x, y)$，$x$: prompt，$y$: response
2. 基准模型：$\pi_{ref}(y|x)$，参数冻结
3. 训练模型：$\pi(y|x)$

DPO 中要最大化的目标函数：

1. **直接最大化策略 $\pi$ 的期望奖励**：通俗理解，对于给定输入 $x$，model 输出 $y$，这个 $y$ 要满足对于 reward model 得分尽可能高。实际上 DPO 并不显式使用 reward model，而是将 preference 数据转化为 loss 形式间接体现这一点，后续会有 loss 推导。DPO 的关键贡献之一：将偏好数据转化为一个可以直接优化的目标函数，绕过显式学习奖励函数。

2. **同时惩罚 $\pi$ 与参考策略 $\pi_{ref}$ 之间的 KL 散度**，以防止策略偏离太多：

$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}[r(x, y)] - \beta \mathbb{D}_{KL}[\pi(y|x) \| \pi_{ref}(y|x)]
$$

> 📌 **数学注解（KL 散度）**：$\mathbb{D}_{KL}[P \| Q] = \sum_y P(y) \log\frac{P(y)}{Q(y)}$，衡量分布 $P$ 与 $Q$ 的差异，值越大说明两个分布差得越远。这里用它来约束训练模型不要偏离参考模型太远，$\beta$ 控制约束强度。

---

### 2.2 损失推导

> **DPO 的公式推导，核心思想是：**
> 将奖励函数 $r(x, y)$ 转换为仅依赖 $\pi$ 和 $\pi_{ref}$ 的形式，从而绕过显式训练奖励模型。

#### 步骤 1：最大化目标函数 → 最小化损失函数

$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}[r(x, y)] - \beta \mathbb{D}_{KL}[\pi(y|x) \| \pi_{ref}(y|x)]
$$

展开 KL 散度（$\mathbb{D}_{KL}[P\|Q] = \mathbb{E}_P[\log\frac{P}{Q}]$）：

$$
= \max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}[r(x, y)] - \mathbb{E}_{x \sim D, y \sim \pi}\left[\beta \log\frac{\pi(y|x)}{\pi_{ref}(y|x)}\right]
$$

$$
= \max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}\left[r(x, y) - \beta \log\frac{\pi(y|x)}{\pi_{ref}(y|x)}\right]
$$

取负号转为最小化：

$$
= \min_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}\left[\log\frac{\pi(y|x)}{\pi_{ref}(y|x)} - \frac{1}{\beta}r(x, y)\right]
$$

> 📌 **数学注解**：最大化 $f$ 等价于最小化 $-f$，这里同时除以 $\beta$（正数）不改变最优解。

---

#### 步骤 2：提取配分函数（Partition Function）$Z(x)$

$$
= \min_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}\left[\log\frac{\pi(y|x)}{\pi_{ref}(y|x)} - \log\exp\left(\frac{1}{\beta}r(x,y)\right)\right]
$$

> 📌 **数学注解**：$\frac{1}{\beta}r(x,y) = \log\exp\left(\frac{1}{\beta}r(x,y)\right)$，利用 $\log e^a = a$ 将减法统一成对数形式，方便合并。

$$
= \min_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}\left[\log\frac{\pi(y|x)}{\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)}\right]
$$

引入配分函数 $Z(x) = \sum_y \pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$，分子分母同乘 $Z(x)$：

$$
= \min_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} - \log Z(x)\right]
$$

> 📌 **数学注解（配分函数）**：$Z(x)$ 是一个归一化常数，使得 $\frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 成为一个合法的概率分布（所有 $y$ 上求和为 1）。在给定输入 $x$ 的前提下，将参考策略 $\pi_{ref}(y|x)$ 和奖励 $r(x,y)$ 的加权组合进行归一化，使得最终结果形成一个合法的概率分布。

---

#### 步骤 3：化简为 KL 散度形式

定义最优策略：

$$
\frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) = \frac{\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)}{\sum_y \pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} = \pi^*(y|x)
$$

则目标函数变为：

$$
= \min_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}\left[\log\frac{\pi(y|x)}{\pi^*(y|x)} - \log Z(x)\right]
$$

由于 $\log Z(x)$ 与 $\pi$ 无关，最小化时可以忽略：

$$
= \min_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}\left[\log\frac{\pi(y|x)}{\pi^*(y|x)}\right]
$$

$$
= \min_{\pi} \mathbb{E}_{x \sim D}\left[\mathbb{D}_{KL}(\pi(y|x) \| \pi^*(y|x))\right]
$$

> 📌 **数学注解**：KL 散度 $\mathbb{D}_{KL}(P\|Q) \geq 0$，当且仅当 $P = Q$ 时取等号。所以最小化 KL 散度的最优解就是 $\pi = \pi^*$。

$$
\Rightarrow \quad \pi(y|x) = \pi^*(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

> 💡 **直觉理解**：我们希望当前策略 $\pi(y|x)$ 尽可能接近一个"理想最优策略" $\pi^*(y|x)$，其中 $\pi^*(y|x)$ 是根据偏好数据和参考策略构造出来的。

---

#### 步骤 4：推导 DPO Loss

在原目标函数的 $r(x,y)$ 的损失函数为 $-\ln\sigma(r(x, y_w) - r(x, y_l))$，从下面公式可以推导出 $r(x,y)$ 另一种表达式，再带入损失函数：

由最优策略公式反推 $r(x, y)$：

$$
\pi(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

$$
\Rightarrow \exp\left(\frac{1}{\beta}r(x,y)\right) = \frac{\pi(y|x)}{\pi_{ref}(y|x)} Z(x)
$$

$$
\bigstar \quad r(x,y) = \beta\ln\left(\frac{\pi(y|x)}{\pi_{ref}(y|x)}Z(x)\right)
$$

$$
\Rightarrow r(x,y) = \beta\ln\left(\frac{\pi(y|x)}{\pi_{ref}(y|x)}\right) + \beta\ln Z(x)
$$

> 📌 **数学注解**：两边取 $\beta \ln(\cdot)$，利用 $\ln e^a = a$ 和 $\ln(ab) = \ln a + \ln b$ 展开。

将 $r(x, y_w)$ 和 $r(x, y_l)$ 代入 RM Loss：

$$
\bigstar \quad -\ln\sigma(r(x, y_w) - r(x, y_l))
$$

$$
= -\ln\sigma\left(\beta\ln\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} + \beta\ln Z(x) - \beta\ln\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)} - \beta\ln Z(x)\right)
$$

$\beta\ln Z(x)$ 项相消：

$$
= -\ln\sigma\left(\beta\ln\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\ln\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\right)
$$

> 📌 **数学注解**：$Z(x)$ 只与输入 $x$ 有关，与 $y_w$、$y_l$ 无关，所以在 $r(x,y_w) - r(x,y_l)$ 中 $\beta\ln Z(x)$ 恰好抵消。这正是 DPO 的精妙之处——不需要显式计算 $Z(x)$！

---

### 2.3 最终 DPO Loss

$$
\boxed{\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)\sim D}\left[\ln\sigma\left(\beta\ln\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\ln\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]}
$$

**直觉理解**：
- $\ln\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)}$：当前策略相对参考模型，对好回答的"偏好程度"（log ratio）
- $\ln\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}$：当前策略相对参考模型，对坏回答的"偏好程度"（log ratio）
- Loss 希望好回答的 log ratio **大于**坏回答的 log ratio，差值越大，loss 越小
- $\beta$ 控制约束强度：$\beta$ 越大，越不允许策略偏离参考模型
