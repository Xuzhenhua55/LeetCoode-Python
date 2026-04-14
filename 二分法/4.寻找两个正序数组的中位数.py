# ============================================================
# 基础解法：合并两个有序数组，找中位数
# 时间复杂度：O(m+n)，空间复杂度：O(m+n)
# ============================================================
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m, n = len(nums1), len(nums2)
        p1, p2 = 0, 0
        fullList = list()

        # 合并两个有序数组
        while p1 < m and p2 < n:
            if nums1[p1] <= nums2[p2]:
                fullList.append(nums1[p1])
                p1 += 1
            else:
                fullList.append(nums2[p2])
                p2 += 1

        # 处理剩余元素
        while p1 < m:
            fullList.append(nums1[p1])
            p1 += 1
        while p2 < n:
            fullList.append(nums2[p2])
            p2 += 1

        # 找中位数
        if (m + n) % 2 != 0:  # 奇数：取中间那个
            return fullList[(m + n) // 2]
        else:  # 偶数：取中间两个的平均值
            return (fullList[(m + n) // 2 - 1] + fullList[(m + n) // 2]) / 2


# 注意：运算符优先级
# m + n % 2 是 m + (n % 2)，不是 (m + n) % 2
# % 的优先级高于 +，必须加括号！


# ============================================================
# 解法二：双指针，只遍历到中位数位置
# 时间复杂度：O((m+n)/2)，空间复杂度：O(1)
# ============================================================
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m, n = len(nums1), len(nums2)
        prev, curr = None, None  # 维护前一个值和当前值
        p1, p2 = 0, 0

        # 只需要遍历到中位数位置，不需要合并整个数组
        for k in range((m + n) // 2 + 1):
            prev = curr  # 记录前一个值（偶数情况需要）
            # 比较当前两个指针位置的值，选较小的前进
            if p1 < m and (p2 >= n or nums1[p1] <= nums2[p2]):
                curr = nums1[p1]
                p1 += 1
            else:
                curr = nums2[p2]
                p2 += 1

        # 奇数：返回 curr；偶数：返回 (curr + prev) / 2
        if (m + n) % 2 != 0:
            return curr
        else:
            return (curr + prev) / 2


# 核心思路：
# 用 p1 和 p2 分别指向 nums1 和 nums2 中等待比较的值
# 总共前进 (m+n)//2 + 1 次，这样奇数时 curr 就是中位数
# 偶数时需要 prev 和 curr 的平均值，所以维护这两个变量


# ============================================================
# 二分解法：在较短数组上二分划分位置
# 时间复杂度：O(log(min(m,n)))，空间复杂度：O(1)
# ============================================================
from math import inf

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # 保证 nums1 是较短的数组，在 nums1 上二分
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        m, n = len(nums1), len(nums2)
        left, right = 0, m + 1

        while left < right:
            # midIndex1 是 nums1 的划分位置（左边有 midIndex1 个元素）
            midIndex1 = (left + right) // 2
            # midIndex2 是 nums2 的划分位置，保证左边总共有 (m+n+1)//2 个元素
            midIndex2 = (m + n + 1) // 2 - midIndex1

            # 四个边界值：左边最大、右边最小
            leftNum1 = nums1[midIndex1 - 1] if midIndex1 > 0 else -inf
            rightNum1 = nums1[midIndex1] if midIndex1 < m else inf
            leftNum2 = nums2[midIndex2 - 1] if midIndex2 > 0 else -inf
            rightNum2 = nums2[midIndex2] if midIndex2 < n else inf

            # 检查划分是否满足条件：左边的所有元素 <= 右边的所有元素
            if leftNum1 <= rightNum2 and leftNum2 <= rightNum1:
                # 找到了正确的划分！计算中位数
                if (m + n) % 2 != 0:  # 奇数：取左边最大的
                    return max(leftNum1, leftNum2)
                else:  # 偶数：取左边最大和右边最小的平均值
                    return (max(leftNum1, leftNum2) + min(rightNum1, rightNum2)) / 2

            # 划分不满足条件，调整二分范围
            elif leftNum1 > rightNum2:  # nums1 左边太大，需要往左移
                right = midIndex1
            else:  # nums1 左边太小，需要往右移
                left = midIndex1 + 1


# ============================================================
# 二分解法核心思路解析
# ============================================================
#
# 目标：将两个有序数组分成两部分，使得：
#   - 左边部分的总长度 = 右边部分的总长度（或左边多一个）
#   - 左边所有元素 <= 右边所有元素
#
# 设 nums1 划分位置为 i（左边有 i 个元素），nums2 划分位置为 j
# 则需要满足：
#   1. i + j = (m + n + 1) // 2  （左边总长度）
#   2. nums1[i-1] <= nums2[j]    （nums1 左边最大 <= nums2 右边最小）
#   3. nums2[j-1] <= nums1[i]    （nums2 左边最大 <= nums1 右边最小）
#
# 为什么在较短数组上二分？
#   - 较短数组划分位置范围小（0~m），二分更快
#   - j = (m+n+1)//2 - i，确定了 i 就确定了 j
#
# 边界处理：
#   - i=0 时，nums1 左边没有元素，leftNum1 设为 -inf
#   - i=m 时，nums1 右边没有元素，rightNum1 设为 inf
#   - 同理处理 j=0 和 j=n 的情况
#
# 中位数计算：
#   - 奇数：左边最大的（max(leftNum1, leftNum2)）
#   - 偶数：(左边最大的 + 右边最小的) / 2
#
# 时间复杂度：O(log(min(m,n)))，只在较短数组上二分