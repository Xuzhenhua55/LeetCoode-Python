class Solution(object):

    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curNum = nums[0]
        count = 0
        for num in nums:
            if count == 0:
                curNum, count = num, 1
            elif num == curNum:
                count += 1
            elif num != curNum:
                count -= 1
        return curNum
