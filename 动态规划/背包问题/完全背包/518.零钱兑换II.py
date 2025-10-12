class Solution(object):

    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        dp = [[0] * (amount + 1) for _ in range(len(coins))]
        for i in range(len(coins)):
            dp[i][0] = 1
        for j in range(coins[0], len(dp[0])):
            dp[0][j] = dp[0][j - coins[0]]
        for i in range(1, len(coins)):
            for j in range(0, len(dp[0])):
                if j >= coins[i]:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i]]
                else:
                    dp[i][j] = dp[i - 1][j]
        # print(dp)
        return dp[-1][amount]


Solution().change(5, [1, 2, 5])


class Solution(object):

    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        dp = [0] * (amount + 1)
        for j in range(len(dp)):
            if j % coins[0] == 0: dp[j] = 1
        for i in range(1, len(coins)):
            for j in range(len(dp)):
                if j >= coins[i]:
                    dp[j] = dp[j] + dp[j - coins[i]]
        return dp[amount]


Solution().change(5, [1, 2, 5])
