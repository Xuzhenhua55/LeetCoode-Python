class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        def reverse(nums, left, right):
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        # 从右往左找到第一个升序位置（nums[i] < nums[i+1]）
        firstAscend = len(nums) - 2
        while firstAscend >= 0:
            if nums[firstAscend] < nums[firstAscend + 1]:
                break
            firstAscend -= 1

        # 如果没找到升序位置，说明整体递减，反转即可
        if firstAscend < 0:
            reverse(nums, 0, len(nums) - 1)
            return

        # 找到右边比升序位置大的最小值（从右往左第一个比它大的）
        firstGreater = len(nums) - 1
        while firstGreater > firstAscend:
            if nums[firstGreater] > nums[firstAscend]:
                break
            firstGreater -= 1

        # 交换
        nums[firstAscend], nums[firstGreater] = nums[firstGreater], nums[firstAscend]

        # 反转升序位置之后的序列（右侧递减 → 反转后递增）
        reverse(nums, firstAscend + 1, len(nums) - 1)