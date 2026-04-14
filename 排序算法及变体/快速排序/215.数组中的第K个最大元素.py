class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 快速选择解法：O(n) 平均时间复杂度
        def quickSelect(leftBound, rightBound):
            if leftBound >= rightBound:
                return nums[leftBound]

            targetNum = nums[rightBound]  # 选择右边界作为 pivot
            i, j = leftBound - 1, leftBound

            # partition：将小于 pivot 的元素放到左边
            while j < rightBound:
                if nums[j] < targetNum:
                    i += 1
                    nums[i], nums[j] = nums[j], nums[i]
                j += 1

            # 将 pivot 放到正确位置
            nums[i + 1], nums[rightBound] = nums[rightBound], nums[i + 1]
            pivotIdx = i + 1

            # 目标位置：第 k 大 = 第 len(nums) - k 小
            target = len(nums) - k

            if pivotIdx == target:
                return nums[pivotIdx]
            elif pivotIdx > target:
                return quickSelect(leftBound, pivotIdx - 1)
            else:
                return quickSelect(pivotIdx + 1, rightBound)

        return quickSelect(0, len(nums) - 1)