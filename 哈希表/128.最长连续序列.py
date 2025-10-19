class Solution(object):

    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        numsSet = set(nums)
        result = 0
        for num in numsSet:
            if num - 1 in numsSet: continue
            nextNum = num + 1
            while nextNum in numsSet:
                nextNum = nextNum + 1
            result = max(result, nextNum - 1 - num + 1)
        return result
