# 这题其实可以看成是滑动窗口的一个负面案例：即很容易想到要使用滑动窗口来做这道题，但是做着做着会发现不太对劲 这是因为滑窗只能解决 随着右边界扩展 整体“单调”的问题
class Solution(object):

    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        result = 0
        prefixSumList = [0] * (len(nums) + 1)
        from collections import defaultdict
        countDict = defaultdict(int)
        countDict[0] = 1
        for i, num in enumerate(nums):
            curSum = prefixSumList[i] + num
            result += countDict[curSum - k]
            prefixSumList[i + 1] = curSum
            countDict[curSum] += 1
        return result
