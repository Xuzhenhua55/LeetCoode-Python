class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        import random
        
        # 快速选择解法：O(N) 平均时间复杂度
        # 为什么是 O(N)？
        # 每次 partition 操作后，只需要根据 pivotIdx 和 target 的大小关系，选择其中一半继续递归，另一半直接丢弃。
        # 假设每次都能完美平分，时间消耗为：N + N/2 + N/4 + ... + 1 ≈ 2N，即 O(N)。
        # 最坏情况是 O(N^2)（例如数组已有序且每次选最右侧元素作为 pivot）。
        # 优化：在 partition 前随机选择一个元素与最右侧元素交换，打乱输入，确保数学期望上时间复杂度稳定在 O(N)。
        def quickSelect(leftBound, rightBound):
            if leftBound >= rightBound:
                return nums[leftBound]

            # 优化：随机选择 pivot 并交换到最右侧，避免最坏情况 O(N^2)
            random_idx = random.randint(leftBound, rightBound)
            nums[random_idx], nums[rightBound] = nums[rightBound], nums[random_idx]

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