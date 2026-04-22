# Centered Kernel Alignment (CKA) 渐进式详解

> 本文以渐进顺序介绍 CKA，从直觉到数学推导再到代码实现，重点讲解**如何比较两个 M×N 矩阵的相似度**。

---

## 目录

1. [为什么要比较两个矩阵的相似度？](#1-为什么要比较两个矩阵的相似度)
2. [朴素方法及其问题](#2-朴素方法及其问题)
3. [核心直觉：比较"样本间关系结构"而非"特征本身"](#3-核心直觉比较样本间关系结构而非特征本身)
4. [Gram 矩阵：将特征空间折叠为样本空间](#4-gram-矩阵将特征空间折叠为样本空间)
5. [HSIC：用核方法衡量统计独立性](#5-hsic用核方法衡量统计独立性)
6. [CKA：归一化的 HSIC](#6-cka归一化的-hsic)
7. [线性 CKA 的简洁公式与推导](#7-线性-cka-的简洁公式与推导)
8. [核化 CKA（RBF 核等）](#8-核化-ckarbf-核等)
9. [完整计算步骤（手把手）](#9-完整计算步骤手把手)
10. [Python 代码实现](#10-python-代码实现)
11. [CKA 的三大关键性质](#11-cka-的三大关键性质)
12. [与其他相似度方法的对比](#12-与其他相似度方法的对比)
13. [CKA 的局限性与注意事项](#13-cka-的局限性与注意事项)
14. [典型应用场景](#14-典型应用场景)
15. [参考文献](#15-参考文献)

---

## 1. 为什么要比较两个矩阵的相似度？

在深度学习中，我们经常需要回答这样的问题：

- 两个不同架构的模型，处理同一批数据时，内部表示有多相似？
- 同一模型不同层之间的表示有什么关系？
- 模型 A 是不是模型 B 的微调版本？

这些问题的本质是：**给定两个矩阵 $X \in \mathbb{R}^{M \times d_1}$ 和 $Y \in \mathbb{R}^{M \times d_2}$（$M$ 个样本，维度可以不同），如何量化它们的相似度？**

- $M$：样本数量（两个矩阵必须行数相同，即同一批样本）
- $d_1, d_2$：特征维度（**可以不同**，这是 CKA 的一大优势）

---

## 2. 朴素方法及其问题

### 方法一：余弦相似度 / 点积

直接计算 X 和 Y 的列向量之间的余弦相似度。**问题：维度必须相同。**

### 方法二：线性回归（$R^2$）

**做法**：用 $X$ 去线性预测 $Y$，即寻找矩阵 $W$ 使得 $Y \approx XW$，然后用决定系数 $R^2$ 衡量预测的好坏。

$$R^2 = 1 - \frac{\|Y - XW\|_F^2}{\|Y - \bar{Y}\|_F^2}$$

**问题：**

1. **非对称**：用 $X$ 预测 $Y$ 和用 $Y$ 预测 $X$ 是两个不同的优化问题，得到的 $R^2$ 值不同。举例：若 $Y = 2X$（$Y$ 完全由 $X$ 决定），则 $X \to Y$ 的 $R^2 = 1$，但 $Y \to X$ 的 $R^2$ 也等于 1——看起来没问题。然而若 $Y = X_1$（$Y$ 只跟 $X$ 的第一列有关），则 $X \to Y$ 的 $R^2 = 1$，但 $Y \to X$ 的 $R^2$ 可能很低（因为 $Y$ 的 1 维无法重建 $X$ 的 $d_1$ 维）。一个对称的相似度度量不应该依赖"谁预测谁"。

2. **维度必须匹配**：线性回归 $Y \approx XW$ 要求 $W \in \mathbb{R}^{d_1 \times d_2}$，虽然 $d_1$ 和 $d_2$ 可以不同，但当 $d_1 < d_2$ 时，$X$ 的表达力不足以预测 $Y$ 的所有维度，$R^2$ 天然偏低——这不是因为表示不相似，而是因为回归问题本身欠定。反过来 $d_1 > d_2$ 则容易过拟合。更根本地说，如果 $d_1 \neq d_2$，两个方向的回归问题难度不对称，无法给出公正的相似度评分。

3. **对尺度敏感**：$R^2$ 的分子 $\|Y - XW\|_F^2$ 的量纲与 $Y$ 的尺度有关。如果将 $Y$ 放大 2 倍，回归系数 $W$ 也会放大 2 倍，残差也放大 2 倍，但分母 $\|Y - \bar{Y}\|_F^2$ 放大 4 倍，所以 $R^2$ 值不变——看起来 $R^2$ 对 $Y$ 的缩放不敏感。但问题在于：如果 $X$ 缩放 $\alpha$ 倍，回归系数 $W$ 会缩放 $1/\alpha$，残差不变，$R^2$ 不变。然而当我们说"两个表示相似"时，我们希望即使 $X$ 和 $Y$ 的激活值量级完全不同（比如一个网络用 ReLU，一个用 tanh），只要结构一致就应该判定相似。$R^2$ 在 $d_1 \neq d_2$ 时的不对称性使得它无法做到这一点。

### 方法三：CCA（典型相关分析）

**做法**：对于 $X \in \mathbb{R}^{M \times d_1}$ 和 $Y \in \mathbb{R}^{M \times d_2}$，CCA 寻找投影向量 $u \in \mathbb{R}^{d_1}$ 和 $v \in \mathbb{R}^{d_2}$，使得 $Xu$ 和 $Yv$ 的相关系数最大：

$$\rho_1 = \max_{u,v} \text{corr}(Xu, Yv)$$

然后在与第一对正交的约束下找第二对，以此类推，得到 $\rho_1, \rho_2, \ldots, \rho_{\min(d_1,d_2)}$。最终相似度通常取这些相关系数的均值。

**问题：**

1. **当 $d > M$ 时退化**：这是 CCA 最致命的问题。当 $d_1 > M$ 或 $d_2 > M$ 时，$X$ 的列空间已经覆盖了 $\mathbb{R}^M$（或接近覆盖），意味着对任意 $Yv$，总能找到某个 $u$ 使得 $Xu = Yv$（或非常接近），从而 $\rho = 1$。换句话说，**无论 $X$ 和 $Y$ 的表示是否真有关系，CCA 都会报告高度相关**。这在深度学习中很常见：现代网络的隐藏维度 $d$ 通常远大于样本数 $M$（例如 $d = 4096$，$M = 100$），此时 CKA 会给出虚高的相似度。数学上，这是因为当 $d > M$ 时，样本协方差矩阵是奇异的，CCA 的目标函数在无穷多组解上都达到最大值。

2. **不满足各向同性缩放不变性**：CCA 寻找的是"最大相关"方向，相关系数本身对缩放是鲁棒的（$\text{corr}(\alpha X, \beta Y) = \text{corr}(X, Y)$），所以单对投影的相关系数有缩放不变性。但当我们把多对 CCA 相关系数聚合成一个整体相似度时（比如取均值），不同方向的贡献权重与各方向的方差有关——缩放 $X$ 会改变各主成分的方差排序，进而改变 CCA 投影的优先级和聚合结果。因此 CCA 的整体相似度度量不满足 $s(\alpha X, Y) = s(X, Y)$。

### 方法四：直接比较特征值（SVD 奇异值）

**做法**：对 $X$ 和 $Y$ 分别做奇异值分解 $X = U_X \Sigma_X V_X^T$，$Y = U_Y \Sigma_Y V_Y^T$，然后比较 $\Sigma_X$ 和 $\Sigma_Y$ 的奇异值分布（如计算两条奇异值曲线之间的距离或相关性）。

**问题：丢失了方向信息，无法判断是否真的"对齐"。**

具体来说：奇异值 $\Sigma$ 只告诉你"沿着每个主成分方向，数据有多大的方差"，但不告诉你这些主成分指向哪里。考虑以下极端例子：

- $X = \begin{bmatrix} 1 & 0 \\ 0 & 100 \end{bmatrix}$，奇异值为 $\{100, 1\}$
- $Y = \begin{bmatrix} 0 & 100 \\ 1 & 0 \end{bmatrix}$（$X$ 的两列交换），奇异值仍为 $\{100, 1\}$

奇异值完全相同！但 $X$ 和 $Y$ 的第一主成分方向是正交的——它们捕捉的是完全不同的信息。如果模型 A 在第一个神经元上编码了"类别信息"，在第二个上编码了"噪声"，而模型 B 反过来，那奇异值分布看起来一模一样，但实际表示结构截然不同。

再举一例：$X$ 的前几个主成分编码了"语义信息"，$Y$ 的前几个主成分编码了完全不同的"位置信息"，但碰巧方差分布相似。SVD 比较法会误判为相似。

**总结**：奇异值分布只反映"能量在各维度上的分配"，不反映"每个维度实际编码了什么"。两个表示相似的前提是它们编码的**信息内容**一致，而不仅仅是**方差分布**一致。

### 核心矛盾

我们想要一个相似度度量，满足：
1. **维度无关**：$d_1 \neq d_2$ 也能用
2. **对称**：$s(X, Y) = s(Y, X)$
3. **正交变换不变**：如果 $Y = XQ$（$Q$ 为正交矩阵），则 $s(X, Y) = 1$
4. **缩放不变**：$s(\alpha X, \beta Y) = s(X, Y)$
5. **有界**：值在 $[0, 1]$ 之间

**CKA 满足以上所有条件。**

---

## 3. 核心直觉：比较"样本间关系结构"而非"特征本身"

CKA 的核心洞察是：

> **不比较"特征怎么表示"，而是比较"样本之间的相似性结构是否一致"。**

举个例子：
- 模型 A 用 64 维向量表示"猫""狗""鸟"
- 模型 B 用 128 维向量表示同样的"猫""狗""鸟"

直接比较维度没意义（维度不同）。但我们可以问：
- 在模型 A 中，"猫"和"狗"的相似度 vs "猫"和"鸟"的相似度
- 在模型 B 中，同样的关系是否保持一致？

**如果两个模型认为"哪些样本相似、哪些样本不同"的判断是一致的，那它们的表示就是相似的——不管维度差多少。**

---

## 4. Gram 矩阵：将特征空间折叠为样本空间

### 定义

给定 $X \in \mathbb{R}^{M \times d}$，其 Gram 矩阵为：

$$K = XX^T \in \mathbb{R}^{M \times M}$$

### 物理意义

- $K_{ij} = x_i \cdot x_j$ = 样本 $i$ 和样本 $j$ 在特征空间中的点积（相似度）
- Gram 矩阵编码了**所有样本对之间的相似关系**
- 维度从 $d$ 被折叠掉了，变成了 $M \times M$

### 关键优势

即使 $d_1 \neq d_2$：
- $K = XX^T$ 是 $M \times M$
- $L = YY^T$ 也是 $M \times M$

**现在两个矩阵维度相同了！可以比较了！**

---

## 5. HSIC：用核方法衡量统计独立性

### 动机

Gram 矩阵之间的相似度怎么衡量？HSIC（Hilbert-Schmidt Independence Criterion）提供了一种方式。

### HSIC 的含义

- HSIC 是一种**统计独立性检验**统计量
- HSIC(X, Y) = 0 意味着 X 和 Y 统计独立
- HSIC 越大，X 和 Y 越相关

### HSIC 的经验估计

给定核矩阵 $K, L$，HSIC 的经验估计需要对核矩阵进行**中心化**。下面先解释为什么叫核矩阵、为什么 HSIC 属于核方法，再解释中心化矩阵，最后给出 HSIC 公式。

#### 为什么叫"核矩阵"？为什么 HSIC 是"核方法"？

**核函数（Kernel Function）** 是一种衡量两个样本之间"相似度"的函数 $k(x_i, x_j)$，但它的强大之处在于：**不需要显式地将数据映射到高维空间，就能隐式地计算高维空间中的内积。**

用一个完整例子讲清楚核技巧：

**场景**：假设有 1 维数据，正类样本 $x = -2$ 和 $x = 2$，负类样本 $x = 0$。在 1 维数轴上，负类夹在两个正类中间——**无论怎么画一个分割点，都无法把两类分开**（选 1 分不开 -2 和 0，选 -1 分不开 0 和 2）。

**思路**：把数据映射到更高维的空间，让数据变得可分。比如映射 $\phi(x) = [x, x^2]$，把 1 维变成 2 维：

- $x = -2 \to \phi(-2) = [-2, 4]$（正类）
- $x = 0 \to \phi(0) = [0, 0]$（负类）
- $x = 2 \to \phi(2) = [2, 4]$（正类）

在 2 维空间中，正类在上方（$x^2=4$），负类在原点——画一条水平线 $x^2 = 2$ 就能轻松分开！这就是"升维后线性可分"的力量。

**问题来了**：映射后计算量会暴增。如果原始数据是 $d$ 维，多项式映射后可能变成 $d^2$ 维甚至更高。RBF 核对应的映射甚至是**无穷维**的——根本不可能显式算出 $\phi(x)$。

**核技巧的洞见**：在很多算法中（比如 SVM、HSIC），我们其实**不需要知道 $\phi(x)$ 长什么样**，只需要知道 $\phi(x_i) \cdot \phi(x_j)$ 的值——即映射后两个向量的内积。而核函数 $k(x_i, x_j)$ 恰好等于这个内积：

$$k(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$$

用上面的例子验证：

$$\phi(x_i) \cdot \phi(x_j) = [x_i, x_i^2] \cdot [x_j, x_j^2] = x_i x_j + x_i^2 x_j^2$$

所以 $k(x_i, x_j) = x_i x_j + x_i^2 x_j^2$。

**关键**：右边这个式子 $x_i x_j + x_i^2 x_j^2$ **只需要原始数据 $x_i, x_j$ 就能算出来**，完全不需要先算出 $\phi(2) = [2, 4]$ 和 $\phi(3) = [3, 9]$ 再做内积。但结果和"先映射再算内积"一模一样！

用数字验证（取 $x_i = -2, x_j = 2$）：
- **先映射再内积**：$\phi(-2) \cdot \phi(2) = [-2, 4] \cdot [2, 4] = -4 + 16 = 12$
- **直接用核函数**：$k(-2, 2) = (-2)(2) + (-2)^2(2)^2 = -4 + 16 = 12$

结果完全一致，但核函数的计算过程中**从未构造过 $[-2, 4]$ 和 $[2, 4]$ 这两个 2 维向量**。

**这就是核技巧**：用核函数在原始空间中计算，等价于在某个高维空间中做内积，但不需要真的去那个高维空间。当映射是无穷维（如 RBF 核）时，核技巧是**唯一**可行的计算方式。

常见的核函数有：

| 核函数 | 定义 | 直觉 |
|--------|------|------|
| **线性核** | $k(x_i, x_j) = x_i^T x_j$ | 原始空间内积，不做任何映射 |
| **RBF 核** | $k(x_i, x_j) = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$ | 映射到无穷维空间，捕捉任意非线性关系 |
| **多项式核** | $k(x_i, x_j) = (x_i^T x_j + c)^d$ | 映射到 $d$ 次多项式特征空间 |

**核矩阵（Kernel Matrix / Gram Matrix）** 就是用核函数计算所有样本对的相似度，排列成的矩阵：

$$K_{ij} = k(x_i, x_j)$$

当核函数取**线性核** $k(x_i, x_j) = x_i^T x_j$ 时，核矩阵就是前面介绍的 Gram 矩阵 $K = XX^T$。所以 **Gram 矩阵是核矩阵的特例**——它对应的是线性核。

**为什么 HSIC 属于"核方法"？**

HSIC 的全称是 Hilbert-Schmidt Independence Criterion（希尔伯特-施密特独立性准则）。它的核心思路是：

1. 用核函数 $k$ 将数据 $X$ 隐式映射到一个高维的**再生核希尔伯特空间（RKHS）** $\mathcal{F}$ 中
2. 用核函数 $l$ 将数据 $Y$ 隐式映射到另一个 RKHS $\mathcal{G}$ 中
3. 在这两个高维空间中计算**交叉协方差算子** $C_{XY}$，它衡量了 $X$ 和 $Y$ 在各自高维空间中的"对齐程度"
4. 用 $C_{XY}$ 的 Hilbert-Schmidt 范数 $\|C_{XY}\|_{HS}^2$ 作为独立性度量

关键在于：**第 3-4 步的计算只需要核矩阵，不需要显式构造高维映射 $\phi(x)$**。这就是为什么 HSIC 属于核方法——它利用核技巧，在高维空间中衡量独立性，但计算完全在核矩阵上完成，复杂度只与样本数 $n$ 有关，与映射后的维度无关（映射后可能是无穷维！）。

> **一句话总结**："核方法"的核心思想是用核函数隐式地做高维映射，从而捕捉非线性关系。HSIC 利用核方法将数据映射到高维空间后再检验独立性，使得它能捕捉 $X$ 和 $Y$ 之间**任意**的统计依赖关系（不只是线性的），而计算却只需要核矩阵 $K, L$ 上的简单操作。

#### 什么是中心化矩阵？

中心化矩阵定义为：

$$H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^T$$

其中 $I_n$ 是 $n \times n$ 单位矩阵，$\mathbf{1}$ 是全 1 列向量（$\mathbf{1} = [1, 1, \ldots, 1]^T \in \mathbb{R}^n$）。

展开来看：

$$H = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix} - \frac{1}{n}\begin{bmatrix} 1 & 1 & \cdots & 1 \\ 1 & 1 & \cdots & 1 \\ \vdots & & \ddots & \vdots \\ 1 & 1 & \cdots & 1 \end{bmatrix} = \begin{bmatrix} 1-\frac{1}{n} & -\frac{1}{n} & \cdots & -\frac{1}{n} \\ -\frac{1}{n} & 1-\frac{1}{n} & \cdots & -\frac{1}{n} \\ \vdots & & \ddots & \vdots \\ -\frac{1}{n} & -\frac{1}{n} & \cdots & 1-\frac{1}{n} \end{bmatrix}$$

**$H$ 的核心作用：左乘 $H$ 等价于减去行均值，右乘 $H$ 等价于减去列均值。**

用一个简单例子说明。假设有向量 $v = [3, 5, 7]^T$，其均值为 5，中心化后应为 $v - 5\cdot\mathbf{1} = [-2, 0, 2]^T$。用 $H$ 计算：

$$Hv = \left(I - \frac{1}{3}\mathbf{1}\mathbf{1}^T\right)v = v - \frac{1}{3}\mathbf{1}(\mathbf{1}^Tv) = v - \underbrace{\frac{3+5+7}{3}}_{=\text{均值}=5}\cdot\mathbf{1} = [-2, 0, 2]^T$$

**对核矩阵做中心化 $\tilde{K} = HKH$ 的含义**：$HK$ 先让每行减去行均值，$HKH$ 再让结果减去列均值。这等价于从核矩阵中移除了"每个样本与所有样本的平均相似度"的影响，使得 $\tilde{K}$ 的行和与列和都为 0——就像协方差矩阵中减去均值后只保留"围绕均值的波动"一样。

#### 行均值与列均值的直观意义

核矩阵 $K$ 是对称的（$K_{ij} = K_{ji}$），但行均值和列均值有着不同的直观解读：

**行均值**——"样本 $i$ 有多大众化？"

$$\text{行均值}_i = \frac{1}{n}\sum_{j=1}^n K_{ij} = \frac{1}{n}\sum_{j=1}^n x_i \cdot x_j$$

这衡量的是：样本 $i$ 与所有样本（包括自己）的平均相似度。如果行均值很高，说明样本 $i$ 跟谁都挺像——它是一个"大众化"的、缺乏辨识度的样本；如果行均值很低，说明样本 $i$ 跟大多数样本都不像——它是一个"异类"。减去行均值，就是剥离掉"这个样本本身有多大众化"的信息，只保留"它相对于平均水平，跟谁特别近、跟谁特别远"的独特关系。

**列均值**——"样本 $j$ 被多少样本觉得像？"

$$\text{列均值}_j = \frac{1}{n}\sum_{i=1}^n K_{ij} = \frac{1}{n}\sum_{i=1}^n x_i \cdot x_j$$

由于 $K$ 对称，列均值在数值上等于行均值。但直观意义稍有不同：列均值衡量的是"样本 $j$ 被所有其他样本视为邻居的程度"——即样本 $j$ 是不是一个"公共邻居"。减去列均值，就是剥离掉"样本 $j$ 作为公共邻居的贡献"，只保留"某个样本 $i$ 对样本 $j$ 的独特亲近感"。

**两次中心化的协同效果**

用具体例子说明。假设 3 个样本的 Gram 矩阵为：

$$K = \begin{bmatrix} 10 & 8 & 2 \\ 8 & 9 & 1 \\ 2 & 1 & 3 \end{bmatrix}$$

- 第 1 行均值 = $(10+8+2)/3 \approx 6.67$，第 2 行均值 $\approx 6$，第 3 行均值 $\approx 2$
- 样本 1 和 2 互相很相似（$K_{12}=8$），样本 3 是异类

$HK$（减去行均值）后：

$$HK \approx \begin{bmatrix} 3.33 & 1.33 & -4.67 \\ 2 & 3 & -5 \\ 0 & -1 & 1 \end{bmatrix}$$

行均值已变为 0。但这还不够——第 1 列的均值 $\approx 1.78$，第 2 列均值 $\approx 1.11$，第 3 列均值 $\approx -2.89$。这说明"被样本 1 和 2 觉得像"本身就是一件容易的事（它们跟谁都挺像），而"被样本 3 觉得像"则很难。$HKH$ 再减去列均值，把这些"公共易亲近性"也剥离掉：

$$\tilde{K} = HKH \approx \begin{bmatrix} 1.56 & 0.22 & -1.78 \\ 0.22 & 1.89 & -2.11 \\ -1.78 & -2.11 & 3.89 \end{bmatrix}$$

现在行和、列和都为 0。$\tilde{K}_{12} = 0.22$ 远小于原始的 $K_{12}=8$——因为样本 1 和 2 之间的相似度很大一部分来自"它们都是大众化样本"，真正独特的亲近感只有 0.22。而 $\tilde{K}_{33} = 3.89$ 相比原始 $K_{33}=3$ 反而变大了——因为样本 3 作为异类，它与自己相似这件事本身就是独特的信息。

> **一句话总结**：减行均值剥离"每个样本有多大众化"，减列均值剥离"每个样本被多少样本觉得像"，两次中心化后只保留"样本之间独特的亲近/疏远关系"。

> **直观类比**：原始 Gram 矩阵 $K$ 的元素 $K_{ij}$ 包含了两部分信息——"样本 $i$ 跟所有样本的平均相似度"和"样本 $i$ 与样本 $j$ 相对于平均水平的独特关系"。中心化剥离了前者，只保留后者。这样两个表示即使整体激活水平不同（一个网络所有相似度都偏高，另一个都偏低），只要它们"相对关系模式"一致，中心化后的核矩阵就会很接近。

#### HSIC 公式

$$\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \text{Tr}(\tilde{K}\tilde{L})$$

其中 $\text{Tr}$ 是**矩阵的迹（Trace）**，定义为矩阵主对角线元素之和：

$$\text{Tr}(A) = \sum_{i=1}^n A_{ii} = A_{11} + A_{22} + \cdots + A_{nn}$$

> **为什么 HSIC 公式里出现了 $\text{Tr}$？** 因为 $\text{Tr}(AB)$ 有一个等价写法：$\text{Tr}(AB) = \sum_{i,j} A_{ij}B_{ji}$，当 $B$ 对称时（$\tilde{L}$ 是对称矩阵），$B_{ji} = B_{ij}$，所以 $\text{Tr}(\tilde{K}\tilde{L}) = \sum_{i,j} \tilde{K}_{ij}\tilde{L}_{ij}$。这就是下面的"等价写法"。

其中 $\tilde{K} = HKH$，$\tilde{L} = HLH$ 是中心化后的核矩阵。

### 等价写法

$$\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \sum_{i,j} (\tilde{K})_{ij} (\tilde{L})_{ij}$$

即：**中心化后的 Gram 矩阵逐元素相乘再求和**（本质是两个向量化矩阵的内积）。

### HSIC 计算实例

沿用上面行均值/列均值解释中的 $K$，并构造一个 $L$，逐步演示 HSIC 的完整计算。

**输入**：3 个样本，两个表示空间各自产生 Gram 矩阵：

$$K = \begin{bmatrix} 10 & 8 & 2 \\ 8 & 9 & 1 \\ 2 & 1 & 3 \end{bmatrix}, \quad L = \begin{bmatrix} 9 & 7 & 1 \\ 7 & 8 & 2 \\ 1 & 2 & 4 \end{bmatrix}$$

$K$ 和 $L$ 的结构很相似（样本 1、2 互相亲近，样本 3 是异类），所以 HSIC 应该偏高。

**Step 1：构造中心化矩阵 $H$**（$n=3$）

$$H = I_3 - \frac{1}{3}\mathbf{1}\mathbf{1}^T = \begin{bmatrix} \frac{2}{3} & -\frac{1}{3} & -\frac{1}{3} \\ -\frac{1}{3} & \frac{2}{3} & -\frac{1}{3} \\ -\frac{1}{3} & -\frac{1}{3} & \frac{2}{3} \end{bmatrix}$$

**Step 2：计算中心化核矩阵 $\tilde{K} = HKH$**

先算 $HK$（每行减去行均值）：

$$HK = \begin{bmatrix} \frac{2}{3} & -\frac{1}{3} & -\frac{1}{3} \\ -\frac{1}{3} & \frac{2}{3} & -\frac{1}{3} \\ -\frac{1}{3} & -\frac{1}{3} & \frac{2}{3} \end{bmatrix}\begin{bmatrix} 10 & 8 & 2 \\ 8 & 9 & 1 \\ 2 & 1 & 3 \end{bmatrix} = \begin{bmatrix} \frac{10}{3} & \frac{4}{3} & -\frac{14}{3} \\ 2 & 3 & -5 \\ 0 & -1 & 1 \end{bmatrix}$$

验证：第 1 行 $10/3 + 4/3 - 14/3 = 0$ ✓ 行和为 0

再算 $\tilde{K} = (HK)H$（每列再减去列均值）：

$$\tilde{K} = HKH = \begin{bmatrix} \frac{10}{3} & \frac{4}{3} & -\frac{14}{3} \\ 2 & 3 & -5 \\ 0 & -1 & 1 \end{bmatrix}\begin{bmatrix} \frac{2}{3} & -\frac{1}{3} & -\frac{1}{3} \\ -\frac{1}{3} & \frac{2}{3} & -\frac{1}{3} \\ -\frac{1}{3} & -\frac{1}{3} & \frac{2}{3} \end{bmatrix} = \begin{bmatrix} \frac{14}{9} & \frac{2}{9} & -\frac{16}{9} \\ \frac{2}{9} & \frac{17}{9} & -\frac{19}{9} \\ -\frac{16}{9} & -\frac{19}{9} & \frac{35}{9} \end{bmatrix} \approx \begin{bmatrix} 1.56 & 0.22 & -1.78 \\ 0.22 & 1.89 & -2.11 \\ -1.78 & -2.11 & 3.89 \end{bmatrix}$$

验证：行和 $\approx 0$，列和 $\approx 0$ ✓

**Step 3：同样计算 $\tilde{L} = HLH$**

$$\tilde{L} = HLH \approx \begin{bmatrix} 1.56 & -0.11 & -1.44 \\ -0.11 & 1.22 & -1.11 \\ -1.44 & -1.11 & 2.56 \end{bmatrix}$$

**Step 4：计算 $\text{HSIC}(K, L) = \frac{1}{(n-1)^2}\text{Tr}(\tilde{K}\tilde{L})$**

$$\tilde{K}\tilde{L} \approx \begin{bmatrix} 1.56 & 0.22 & -1.78 \\ 0.22 & 1.89 & -2.11 \\ -1.78 & -2.11 & 3.89 \end{bmatrix}\begin{bmatrix} 1.56 & -0.11 & -1.44 \\ -0.11 & 1.22 & -1.11 \\ -1.44 & -1.11 & 2.56 \end{bmatrix}$$

主对角线元素（用于计算迹）：

- $(\tilde{K}\tilde{L})_{11} = 1.56 \times 1.56 + 0.22 \times (-0.11) + (-1.78) \times (-1.44) \approx 2.43 - 0.02 + 2.56 = 4.97$
- $(\tilde{K}\tilde{L})_{22} = 0.22 \times (-0.11) + 1.89 \times 1.22 + (-2.11) \times (-1.11) \approx -0.02 + 2.31 + 2.34 = 4.63$
- $(\tilde{K}\tilde{L})_{33} = (-1.78) \times (-1.44) + (-2.11) \times (-1.11) + 3.89 \times 2.56 \approx 2.56 + 2.34 + 9.96 = 14.86$

$$\text{Tr}(\tilde{K}\tilde{L}) \approx 4.97 + 4.63 + 14.86 = 24.46$$

$$\text{HSIC}(K, L) = \frac{24.46}{(3-1)^2} = \frac{24.46}{4} \approx 6.12$$

**Step 5（可选）：计算 $\text{HSIC}(K,K)$ 和 $\text{HSIC}(L,L)$ 以得到 CKA**

同理算得 $\text{HSIC}(K,K) \approx 9.72$，$\text{HSIC}(L,L) \approx 5.39$，则：

$$\text{CKA}(K, L) = \frac{6.12}{\sqrt{9.72 \times 5.39}} = \frac{6.12}{7.24} \approx 0.845$$

CKA $\approx 0.845$，说明 $K$ 和 $L$ 的结构很相似，符合直觉（它们都是"1、2亲近，3 异类"的模式）。

> **另一种等价算 HSIC 的方式**：用 $\sum_{i,j} \tilde{K}_{ij}\tilde{L}_{ij}$ 而非 $\text{Tr}(\tilde{K}\tilde{L})$，两者结果完全相同，但前者在代码中更直接——只需将两个矩阵逐元素相乘再求和。

### 为什么需要中心化？

中心化确保相似度度量不受常数偏移影响——这就像在计算协方差时要先减去均值一样。如果不做中心化，两个表示即使结构完全一致但整体激活值偏高，它们的 Gram 矩阵中所有元素都偏大，HSIC 的值就会被这种"整体偏移"主导，而非反映真正的结构相似性。

---

## 6. CKA：归一化的 HSIC

### HSIC 的问题

HSIC 不满足缩放不变性：$\text{HSIC}(\alpha X, Y) \neq \text{HSIC}(X, Y)$。直接用 HSIC 作为相似度，值域无界，难以跨场景比较。

### CKA 的定义

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

这就是 HSIC 的**余弦相似度**形式！类比向量 $a, b$ 的余弦相似度：

$$\cos(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}$$

CKA 就是将 Gram 矩阵展平为向量后的余弦相似度。

### CKA 的值域

- **范围 $[0, 1]$**
- $\text{CKA} = 1$：两个表示的样本间相似性结构完全一致
- $\text{CKA} = 0$：两个表示完全独立

---

## 7. 线性 CKA 的简洁公式与推导

### 最终公式

在线性核下（$K = XX^T$, $L = YY^T$），CKA 可以化简为极其简洁的形式：

$$\boxed{\text{CKA}(X, Y) = \frac{\|Y^TX\|_F^2}{\|X^TX\|_F \cdot \|Y^TY\|_F}}$$

### 完整推导

**目标：证明分子 $\|Y^TX\|_F^2 = \langle \text{vec}(XX^T), \text{vec}(YY^T) \rangle$**

$$
\begin{align*}
\|Y^TX\|_F^2 &= \text{Tr}\left((Y^TX)^T Y^TX\right) && \text{Frobenius范数定义} \\
&= \text{Tr}\left(X^TYY^TX\right) && \text{转置化简} \\
&= \text{Tr}\left(XX^TYY^T\right) && \text{迹的循环性质} \\
&= \sum_i (XX^TYY^T)_{ii} && \text{展开迹} \\
&= \sum_i \sum_j (XX^T)_{ij}(YY^T)_{ji} && \text{展开矩阵乘法} \\
&= \sum_i \sum_j (XX^T)_{ij}(YY^T)_{ij} && \text{Gram矩阵对称性：}(YY^T)_{ji}=(YY^T)_{ij} \\
&= \langle \text{vec}(XX^T), \text{vec}(YY^T)\rangle && \text{向量化后的点积}
\end{align*}
$$

**归一化项同理：** $\|X^TX\|_F^2 = \langle \text{vec}(XX^T), \text{vec}(XX^T) \rangle$

**代入 CKA 定义：**

$$
\begin{align*}
\text{CKA}(X,Y) &= \frac{\text{HSIC}(K,L)}{\sqrt{\text{HSIC}(K,K)\cdot\text{HSIC}(L,L)}} \\
&= \frac{\frac{1}{(n-1)^2}\|Y^TX\|_F^2}{\sqrt{\frac{1}{(n-1)^2}\|X^TX\|_F^2 \cdot \frac{1}{(n-1)^2}\|Y^TY\|_F^2}} \\
&= \frac{\frac{1}{(n-1)^2}\|Y^TX\|_F^2}{\frac{1}{n-1}\cdot\frac{1}{n-1}\|X^TX\|_F\|Y^TY\|_F} \\
&= \frac{\|Y^TX\|_F^2}{\|X^TX\|_F \cdot \|Y^TY\|_F}
\end{align*}
$$

**所有 (n-1) 项完全消去！** 最终公式只涉及三个 Frobenius 范数的计算。

---

## 8. 核化 CKA（RBF 核等）

### 为什么需要核化？

线性核只能捕捉线性关系。如果两个表示之间存在非线性关系，线性 CKA 可能给出低分。

### 核化 CKA 公式

$$\text{CKA}_{\text{kernel}}(X, Y) = \frac{\text{Tr}(\tilde{K}\tilde{L})}{\sqrt{\text{Tr}(\tilde{K}\tilde{K}) \cdot \text{Tr}(\tilde{L}\tilde{L})}}$$

其中：
- $K_{ij} = k(x_i, x_j)$（核函数，如 RBF）
- $L_{ij} = l(y_i, y_j)$
- $\tilde{K} = HKH$, $\tilde{L} = HLH$

### 常用核函数

| 核函数 | 公式 | 特点 |
|--------|------|------|
| **线性核** | $k(x,y) = x^Ty$ | 计算快，捕捉线性关系 |
| **RBF 核** | $k(x,y) = \exp(-\|x-y\|^2 / 2\sigma^2)$ | 捕捉非线性关系，需要选 σ |
| **多项式核** | $k(x,y) = (x^Ty + c)^d$ | 介于线性和 RBF 之间 |

### RBF 核的 σ 选择

实践中常用 **中位数启发式**：令 $\sigma = \sqrt{\text{样本对距离的中位数}}$，避免手动调参。

---

## 9. 完整计算步骤（手把手）

给定 $X \in \mathbb{R}^{M \times d_1}$ 和 $Y \in \mathbb{R}^{M \times d_2}$，计算线性 CKA 的步骤：

### Step 1：中心化

```python
X_centered = X - X.mean(axis=0, keepdims=True)  # 逐列减均值
Y_centered = Y - Y.mean(axis=0, keepdims=True)
```

### Step 2：计算 Gram 矩阵

```python
K = X_centered @ X_centered.T  # M×M
L = Y_centered @ Y_centered.T  # M×M
```

### Step 3：中心化 Gram 矩阵（对于核化 CKA 需要此步，线性 CKA 可跳过）

```python
H = np.eye(M) - np.ones((M, M)) / M
K_tilde = H @ K @ H
L_tilde = H @ L @ H
```

> **注意**：线性 CKA 的简洁公式已经隐含了中心化，可以直接用 Step 4 的公式。

### Step 4：计算 CKA

**线性核（简洁公式）：**

```python
numerator = np.sum((Y_centered.T @ X_centered) ** 2)  # ||Y^T X||_F^2
denominator = np.sqrt(np.sum((X_centered.T @ X_centered) ** 2)) * \
              np.sqrt(np.sum((Y_centered.T @ Y_centered) ** 2))
cka = numerator / denominator
```

**核化公式（通用）：**

```python
cka = np.sum(K_tilde * L_tilde) / np.sqrt(np.sum(K_tilde * K_tilde) * np.sum(L_tilde * L_tilde))
```

### Step 5：解读结果

| CKA 值 | 含义 |
|--------|------|
| $\approx 1.0$ | 两个表示的样本间结构几乎一致 |
| $\approx 0.5$ | 有一定相似性 |
| $\approx 0.0$ | 两个表示几乎独立 |
| $X$ 与自身 | 恒等于 $1.0$ |

---

## 10. Python 代码实现

```python
import numpy as np

def linear_CKA(X, Y):
    """
    线性 CKA：比较两个矩阵 $X \in \mathbb{R}^{M \times d_1}$ 和 $Y \in \mathbb{R}^{M \times d_2}$ 的相似度
    
    返回值在 $[0, 1]$ 之间，1 表示完全相似
    """
    # 中心化
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    # 简洁公式：||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    numerator = np.sum((Y.T @ X) ** 2)
    denominator = np.sqrt(np.sum((X.T @ X) ** 2)) * np.sqrt(np.sum((Y.T @ Y) ** 2))
    
    return numerator / denominator


def kernel_CKA(X, Y, sigma=None):
    """
    核化 CKA（RBF 核）：比较两个矩阵的相似度
    """
    def rbf_kernel(Z, sigma):
        # 计算平方欧氏距离矩阵
        sq_dist = np.sum(Z**2, axis=1, keepdims=True) + \
                  np.sum(Z**2, axis=1) - 2 * Z @ Z.T
        if sigma is None:
            # 中位数启发式选择 sigma
            mask = sq_dist != 0
            sigma = np.sqrt(np.median(sq_dist[mask]))
        return np.exp(-sq_dist / (2 * sigma**2))
    
    def center_kernel(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    
    K = rbf_kernel(X, sigma)
    L = rbf_kernel(Y, sigma)
    
    Kc = center_kernel(K)
    Lc = center_kernel(L)
    
    hsic_xy = np.sum(Kc * Lc)
    hsic_xx = np.sum(Kc * Kc)
    hsic_yy = np.sum(Lc * Lc)
    
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


# ========== 使用示例 ==========
if __name__ == "__main__":
    np.random.seed(42)
    M, d1, d2 = 100, 64, 128  # 100个样本，不同维度
    
    X = np.random.randn(M, d1)
    Y = np.random.randn(M, d2)
    
    # 随机矩阵：CKA 应该接近 0
    print(f"线性 CKA (随机 X vs Y): {linear_CKA(X, Y):.4f}")
    
    # 自比较：CKA 应该等于 1
    print(f"线性 CKA (X vs X):      {linear_CKA(X, X):.4f}")
    
    # 正交变换不变性验证
    Q, _ = np.linalg.qr(np.random.randn(d1, d1))  # 随机正交矩阵
    X_rotated = X @ Q
    print(f"线性 CKA (X vs XQ):     {linear_CKA(X, X_rotated):.4f}  (应≈1)")
    
    # 缩放不变性验证
    print(f"线性 CKA (X vs 3X):     {linear_CKA(X, 3*X):.4f}  (应≈1)")
    
    # RBF 核 CKA
    print(f"RBF 核 CKA (随机 X vs Y): {kernel_CKA(X, Y):.4f}")
```

### 计算复杂度

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 线性 CKA（简洁公式） | $O(M^2 d_1 + M^2 d_2 + M d_1 d_2)$ | $O(M d_1 + M d_2)$ |
| 核化 CKA | $O(M^2 d_1 + M^2 d_2 + M^2)$ | $O(M^2)$ |

当 $M$ 远小于 $d$ 时，简洁公式更高效；当 $d$ 远小于 $M$ 时，两者差别不大。

---

## 11. CKA 的三大关键性质

### 性质 1：维度无关性

d₁ ≠ d₂ 也能比较。因为 Gram 矩阵把特征维度折叠掉了，只保留样本间关系。

### 性质 2：正交变换不变性

如果 $Y = XQ$（$Q$ 为正交矩阵），则 $\text{CKA}(X, Y) = 1$。

**直觉**：正交变换只改变基的方向，不改变样本间的相对关系。就像旋转坐标系不会改变点之间的距离。

### 性质 3：各向同性缩放不变性

$\text{CKA}(\alpha X, \beta Y) = \text{CKA}(X, Y)$，对任意非零标量 $\alpha, \beta$ 成立。

**直觉**：两个表示的整体幅度（activation scale）不同不应影响相似性判断。

---

## 12. 与其他相似度方法的对比

| 方法 | 维度无关 | 对称 | 正交不变 | 缩放不变 | 值域 | 核心思想 |
|------|---------|------|---------|---------|------|---------|
| **CKA** | ✅ | ✅ | ✅ | ✅ | $[0,1]$ | 比较样本间相似性结构 |
| **SVCCA** | ✅(需SVD) | ✅ | ✅ | ❌ | $[0,1]$ | SVD降维 + CCA |
| **PWCCA** | ❌ | ❌ | 部分 | ❌ | $[0,1]$ | 加权投影的CCA |
| **线性回归 $R^2$** | ❌ | ❌ | ❌ | ❌ | $[0,1]$ | 用X预测Y |
| **RSA** | ✅ | ✅ | ✅ | ✅ | $[-1,1]$ | 基于距离矩阵的相关 |
| **直接余弦相似度** | ❌ | ✅ | ❌ | ✅ | $[-1,1]$ | 向量间夹角 |

### CKA vs SVCCA

- SVCCA 先对各自做 SVD 降维再算 CCA，**需要选择保留的维度数**，可能丢失信息
- CKA 直接比较 Gram 矩阵，无需选择超参数（线性核时）
- Kornblith 等人的实验表明 CKA 更能可靠地识别不同初始化网络之间的对应关系

### CKA vs 线性回归

- 线性回归**非对称**（$X \to Y$ 和 $Y \to X$ 结果不同）
- 线性回归**维度必须匹配**
- CKA 是 HSIC 的归一化，等价于 Gram 矩阵的余弦相似度

### CKA vs CCA

- CCA 当 $d > M$ 时会退化（总能找到完美相关的投影方向）
- CKA 基于 HSIC，不受此限制
- IJCAI 2024 的论文证明 CKA 本质上衡量的是**协方差算子的余弦相似度**

---

## 13. CKA 的局限性与注意事项

### 局限 1：对异常值敏感

CKA 基于 Gram 矩阵的点积，异常样本会过度影响结果。如果数据中有极端的离群点，建议先进行清洗。

### 局限 2：对保持线性可分性的变换敏感

如果对表示做某种变换后数据仍然线性可分（模型功能不变），CKA 可能给出与直觉不符的低分。这是因为 CKA 关心的是整体结构，而不仅是分类边界。

### 局限 3：CKA 值可被操纵

研究表明，可以在不改变模型功能行为的情况下，通过修改内部表示来改变 CKA 值。**CKA 值的变化不一定对应功能相似度的变化。**

### 局限 4：不对一般线性变换不变

CKA 只对正交变换和缩放不变，**不对任意可逆线性变换不变**。如果 $Y = XA$（$A$ 不是正交矩阵），CKA 值可能很低。

### 局限 5：只衡量全局相似性

CKA 给出一个标量值，不提供"哪些部分相似、哪些部分不同"的细粒度信息。

### 最佳实践

1. **样本数 $M$ 要足够大**（通常 $\geq 100$），否则统计估计不稳定
2. **使用同一批输入数据**计算两个矩阵的表示
3. **线性 CKA 通常是首选**——简单、快速、无需选超参数
4. **不要孤立地解读 CKA 值**——应结合具体应用场景和基线
5. **注意数据预处理**——中心化是必须的

---

## 14. 典型应用场景

### 场景 1：模型指纹 / 溯源

判断模型 B 是否是模型 A 的微调版本：提取两者各层的激活矩阵，计算层间 CKA 矩阵，高 CKA 值的对应关系揭示血缘关系。

### 场景 2：层间相似性分析

计算同一模型各层之间的 CKA 矩阵（热力图），观察表示的演化过程。

### 场景 3：跨架构比较

比较 CNN 和 Transformer 在同一数据上的内部表示，CKA 可以揭示哪些层扮演了相似的角色。

### 场景 4：训练动态监控

在训练的不同 checkpoint 计算各层与最终层的 CKA，观察表示的收敛过程。

### 场景 5：知识蒸馏

用 CKA 作为损失函数，使学生模型的内部表示对齐教师模型。

---

## 15. 参考文献

1. **Kornblith, S., Norouzi, M., Lee, H., & Hinton, G.** (2019). *Similarity of Neural Network Representations Revisited*. ICML 2019. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
2. **Davari, M., Horoi, S., Natik, A., Lajoie, G., Wolf, G., & Belilovsky, E.** (2022). *Reliability of CKA as a Similarity Measure in Deep Learning*. [arXiv:2210.16156](https://arxiv.org/abs/2210.16156)
3. **Cortes, C., Mohri, M., & Rostamizadeh, A.** (2012). *Algorithms for Learning Kernels Based on Centered Alignment*. JMLR.
4. **Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B.** (2005). *Measuring Statistical Dependence with Hilbert-Schmidt Norms*. ALT 2005.
5. IJCAI 2024: *Rethinking Centered Kernel Alignment in Knowledge Distillation*
