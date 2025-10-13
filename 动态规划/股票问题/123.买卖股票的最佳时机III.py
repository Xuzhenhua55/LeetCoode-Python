class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # 第一次持有股票 第一次不持有股票 第二次持有股票 第二次不持有股票
        dp = [[0, 0, 0, 0] for _ in range(len(prices))]
        dp[0][0], dp[0][2] = -prices[0], -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], -prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] - prices[i])
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] + prices[i])
        return max(dp[-1][1], dp[-1][3])
