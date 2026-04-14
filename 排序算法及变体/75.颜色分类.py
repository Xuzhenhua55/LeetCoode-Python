class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # 荷兰国旗问题：两个指针实现
        # 核心思路：先变 2 → ≤1 就变 1 → 是 0 托变 0
        # 后面的设置会覆盖前面的，保证顺序正确
        nextRedIndex, nextWhiteIndex = 0, 0

        for i, num in enumerate(nums):
            nums[i] = 2  # 先设为 2
            if num <= 1:
                nums[nextWhiteIndex] = 1  # ≤1 就变 1
                nextWhiteIndex += 1
            if num == 0:
                nums[nextRedIndex] = 0  # 是 0 才变 0
                nextRedIndex += 1


# 感悟：
# 假设已有有序序列 001122，插入新元素时：
# - 插入 0：需要把 1122 都往后移，0 放在最后一个 0 的下一位
# - 插入 1：需要把 22 往后移，1 放在最后一个 1 的下一位
# - 插入 2：当前位置不变即可
#
# 关键洞察：插入 0 会影响 1 的位置，但插入 1 不会影响 0 的位置
# 因此最优顺序是：先变 2（不影响任何人）→ ≤1 就变 1 → 是 0 才变 0
# 这样后面的操作覆盖前面，两个指针就足够了