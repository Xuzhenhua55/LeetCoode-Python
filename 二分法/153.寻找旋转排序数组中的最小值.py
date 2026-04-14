class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums)

        while left < right:
            midIndex = (left + right) // 2
            if nums[midIndex] >= nums[0]:
                # mid 在左边上升段，最小值在右边
                left = midIndex + 1
            else:
                # mid 在右边下降段，最小值在左边（包含 mid）
                right = midIndex

        # left 是假设存在"悬崖"时的最低点
        # 如果 left 超出数组大小，说明数组未旋转（单纯上坡）
        return nums[left] if left < len(nums) else nums[0]