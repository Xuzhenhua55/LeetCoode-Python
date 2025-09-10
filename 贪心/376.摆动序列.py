class Solution(object):

    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        isUp = None
        result = 1
        for i in range(1, len(nums)):
            if isUp != False and nums[i] < nums[i - 1]:
                result += 1
                isUp = False
            elif isUp != True and nums[i] > nums[i - 1]:
                result += 1
                isUp = True

        return result
