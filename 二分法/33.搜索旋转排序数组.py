class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)

        while left < right:
            midIndex = (left + right) // 2
            if nums[midIndex] == target:
                return midIndex

            if nums[midIndex] > target:
                # mid 值大于 target
                if nums[0] <= target:
                    # target 在左边上升段，最小值也在左边
                    right = midIndex
                elif nums[0] > target:
                    # target 在右边下降段
                    if nums[midIndex] > nums[-1]:
                        # mid 在左边上升段，target 在右边，需要向右找
                        left = midIndex + 1
                    else:
                        # mid 在右边下降段，向左找
                        right = midIndex

            else:
                # mid 值小于 target
                if nums[0] > target:
                    # target 在右边下降段，向右找
                    left = midIndex + 1
                elif nums[0] <= target:
                    # target 在左边上升段
                    if nums[midIndex] >= nums[0]:
                        # mid 在左边上升段，向右找
                        left = midIndex + 1
                    else:
                        # mid 在右边下降段，target 在左边，向左找
                        right = midIndex

        return -1