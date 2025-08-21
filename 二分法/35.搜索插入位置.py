# 第一个大于target的数
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left,right=0,len(nums)
        while left<right:
            mid=left+(right-left)/2
            if nums[mid]==target:
                return mid
            elif nums[mid]>target:
                right=mid
            else:
                left=mid+1
        return left