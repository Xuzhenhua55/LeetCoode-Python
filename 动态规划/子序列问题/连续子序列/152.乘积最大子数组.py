class Solution(object):

    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [[0] * 2 for _ in range(len(nums))]
        result = -float('inf')
        dp[0][0] = dp[0][1] = nums[0]
        for i in range(1, len(nums)):
            dp[i][0] = max(max(dp[i - 1][0] * nums[i], dp[i - 1][1] * nums[i]),
                           nums[i])
            dp[i][1] = min(min(dp[i - 1][0] * nums[i], dp[i - 1][1] * nums[i]),
                           nums[i])
        return max([max(x) for x in dp])
