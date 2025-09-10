class Solution(object):

    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1: return 0
        left, maxRight = 0, 0
        step = 0
        while True:
            maxIndex = left
            for i in range(left, maxRight + 1):
                if i + nums[i] > maxRight:
                    maxRight = i + nums[i]
                    maxIndex = i
            left = maxIndex
            step += 1
            if maxRight >= len(nums) - 1: break
        return step
