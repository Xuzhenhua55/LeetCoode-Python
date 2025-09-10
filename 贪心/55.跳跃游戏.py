class Solution(object):

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        maxRight = 0
        for i in range(0, len(nums) - 1):
            if i <= maxRight:
                maxRight = max(maxRight, i + nums[i])
        return maxRight >= len(nums) - 1
