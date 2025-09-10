class Solution(object):

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curSum = 0
        result = -float('inf')
        for i in range(0, len(nums)):
            if nums[i] >= 0:
                curSum += nums[i]
                result = max(result, curSum)
            else:
                if curSum + nums[i] <= 0:
                    curSum = 0
                    # 这一句比较重要，因为有可能之前就已经是负数了，那么当前值有可能比之前的负数更大
                    result = max(result, nums[i])
                else:
                    curSum += nums[i]
        return result


class Solution(object):

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curSum = 0
        result = -float('inf')
        for i in range(0, len(nums)):
            curSum += nums[i]
            if curSum > result:
                result = curSum
            if curSum <= 0: curSum = 0
        return result
