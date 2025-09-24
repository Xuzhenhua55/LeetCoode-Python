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


class Solution(object):

    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1: return 0
        curMaxRight = nextMaxRight = step = 0
        for i in range(len(nums)):
            nextMaxRight = max(i + nums[i], nextMaxRight)
            if i == curMaxRight:
                step += 1
                curMaxRight = nextMaxRight
                if curMaxRight >= len(nums) - 1: break

        return step
