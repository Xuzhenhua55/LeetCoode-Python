class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        result=[]
        left,right=0,len(nums)-1
        while left<=right:
            if abs(nums[left])<abs(nums[right]):
                result.append(nums[right]*nums[right])
                right-=1
            else:
                result.append(nums[left]*nums[left])
                left+=1
        result=sorted(result)
        return result