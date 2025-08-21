class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        windowValue=0
        left,right=0,0
        result=float('inf')
        while right<len(nums):
            windowValue+=nums[right]
            while left<right and windowValue-nums[left]>=target:
                windowValue-=nums[left]
                left+=1
            if right-left+1<result and windowValue>=target:
                result=right-left+1
            right+=1
        if result!=float('inf'):
            return result
        else:
            return 0