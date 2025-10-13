class Solution(object):

    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        """
        # 0 持有股票的最大利润 1 不持有股票的最大利润
        dp = [[0, 0] for _ in range(len(prices))]
        dp[0][0] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i] - fee)
        # print(dp)
        return max(dp[-1])


Solution().maxProfit([1, 3, 2, 8, 4, 9], 2)
