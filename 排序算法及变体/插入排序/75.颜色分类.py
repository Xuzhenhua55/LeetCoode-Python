class Solution(object):

    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        nextRedIndex, nextWhiteIndex = 0, 0
        for i, num in enumerate(nums):
            nums[i] = 2
            if num <= 1:
                nums[nextWhiteIndex] = 1
                nextWhiteIndex += 1
            if num == 0:
                nums[nextRedIndex] = 0
                nextRedIndex += 1
