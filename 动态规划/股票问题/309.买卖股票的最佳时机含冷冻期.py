class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # 0 冷冻期的最大利润 1 持有股票的最大利润 2 普通不持有股票的最大利润
        # 核心在于 将 I II III IV中不持有股票的状态 拆分为冷冻期+普通不持有 使得持有股票只能从冷冻期状态转移
        dp = [[0, 0, 0] for _ in range(len(prices))]
        dp[0][1] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])
        return max(dp[-1])
