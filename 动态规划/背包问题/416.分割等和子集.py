class Solution(object):

    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        numSum = sum(nums)
        if numSum % 2 != 0: return False
        dp = [0] * (numSum // 2 + 1)
        for i in range(len(nums)):
            for j in range(len(dp) - 1, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return dp[-1] == numSum // 2


Solution().canPartition([1, 5, 11, 5])
