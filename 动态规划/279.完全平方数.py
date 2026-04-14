from math import sqrt

class Solution:
    def numSquares(self, n: int) -> int:
        dp = [float('inf')] * (n + 1)

        # 完全平方数只需要 1 个
        for i in range(1, int(sqrt(n)) + 1):
            dp[i * i] = 1

        # 对每个数尝试减去所有可能的完全平方数
        for i in range(2, n + 1):
            if dp[i] == 1: continue  # 完全平方数跳过
            for j in range(1, int(sqrt(i)) + 1):
                dp[i] = min(1 + dp[i - j * j], dp[i])

        return dp[-1]


# 感悟：
# 理论上针对任何一个数都可以拆成 1+n-1、2+n-2 这样的组合去计算 dp 最小值，但这样会超时。
# 仔细思考后发现：事实上任何一个数的最优拆分，必然是某几个完全平方数 + 若干没办法拆分的非完全平方数。
# 因此只需要尝试减去所有可能的完全平方数 j*j，取 min(dp[i-j*j] + 1) 即可。


# 解法二：完全背包问题视角
from math import isqrt, inf

class Solution:
    def numSquares(self, n: int) -> int:
        # 把问题看成完全背包问题：
        # 袋子大小为 n，物品是完全平方数（1, 4, 9, 16...）
        # 目标：刚好装满袋子，最少需要几个物品
        dp = [0] + n * [inf]

        for i in range(1, isqrt(n) + 1):
            # 完全背包：每种物品可以用无限次，正序遍历
            for j in range(i * i, n + 1):
                dp[j] = min(dp[j], dp[j - i * i] + 1)

        return dp[-1]