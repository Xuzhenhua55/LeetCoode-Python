class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        indexDict=dict()
        for i in range(len(nums)):
            if target-nums[i] in indexDict:
                return [i,indexDict[target-nums[i]]]
            indexDict[nums[i]]=i
        return []