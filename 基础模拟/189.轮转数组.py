class Solution(object):

    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        nums.reverse()
        # 后k个必然会被移动到前k的位置并且按照原有顺序
        # 剩余的n-k个正常保持原有顺序即可
        nums[:k] = reversed(nums[:k])
        nums[k:] = reversed(nums[k:])


Solution().rotate([1, 2, 3, 4, 5, 6, 7], 3)
