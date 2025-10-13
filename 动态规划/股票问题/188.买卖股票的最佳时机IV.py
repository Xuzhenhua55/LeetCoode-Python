class Solution(object):

    def maxProfit(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        """
        # 0 第一次持有股票的最大利润 1 第一次不持有股票的最大利润 2k-1 第k次持有股票的最大利润 2k 第k次不持有股票的最大利润
        dp = [[0] * (2 * k) for _ in range(len(prices))]
        for i in range(2 * k):
            if i % 2 == 0: dp[0][i] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], -prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
            for j in range(2, len(dp[0]), 2):
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i])
                dp[i][j + 1] = max(dp[i - 1][j + 1], dp[i - 1][j] + prices[i])
        return max(dp[-1])
