# dp[i]表示拆分该数字能够得到的最大的乘积
class Solution(object):

    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [-float('inf')] * (n + 1)
        dp[1] = 1
        dp[2] = 1
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):
                dp[i] = max(dp[i], max(j, dp[j]) * max(i - j, dp[i - j]))
        return dp[n]


Solution().integerBreak(8)
