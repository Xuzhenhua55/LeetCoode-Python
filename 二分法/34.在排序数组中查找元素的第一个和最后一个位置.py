class Solution(object):
    # 第一个等于target的数
    def findLeft(self, nums, target):
        left, right = 0, len(nums)
        while (left < right):
            mid = left + (right - left) / 2
            if nums[mid] == target or nums[mid] > target:
                right = mid
            else:
                left = mid + 1
        if left < len(nums) and nums[left] == target:
            return left
        else:
            return -1

    # 第一个大于target的数
    def findRight(self, nums, target):
        left, right = 0, len(nums)
        while (left < right):
            mid = left + (right - left) / 2
            if nums[mid] == target or nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        if (left - 1 >= 0 and nums[left - 1] == target):
            return left - 1
        else:
            return -1

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        return [self.findLeft(nums, target), self.findRight(nums, target)]
