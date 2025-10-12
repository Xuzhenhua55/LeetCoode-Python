class Solution(object):

    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        nums = []
        for i in range(1, n + 1):
            if i * i <= n: nums.append(i * i)
            else: break
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(len(nums)):
            for j in range(len(dp)):
                if j >= nums[i]:
                    dp[j] = min(dp[j], dp[j - nums[i]] + 1)
        # print(dp)
        return dp[-1]
