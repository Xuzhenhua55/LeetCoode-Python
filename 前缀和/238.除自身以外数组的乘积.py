class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        preMulti = [0] * n
        postMulti = [0] * n

        # 前缀乘积：preMulti[i] = nums[0] * nums[1] * ... * nums[i]
        preMulti[0] = nums[0]
        for i in range(1, n):
            preMulti[i] = preMulti[i - 1] * nums[i]

        # 后缀乘积：postMulti[i] = nums[n-1] * nums[n-2] * ... * nums[n-1-i]
        postMulti[0] = nums[-1]
        for i in range(1, n):
            postMulti[i] = postMulti[i - 1] * nums[n - 1 - i]

        # 结果：result[i] = preMulti[i-1] * postMulti[n-1-(i+1)]
        result = []
        result.append(postMulti[-2])  # 第一个元素：只有后缀乘积（不含第一个）
        for i in range(1, n - 1):
            result.append(preMulti[i - 1] * postMulti[n - 1 - (i + 1)])
        result.append(preMulti[-2])  # 最后一个元素：只有前缀乘积（不含最后一个）

        return result


# ============================================================
# 核心思路：前缀乘积 + 后缀乘积
# ============================================================
#
# result[i] = nums[0] * nums[1] * ... * nums[i-1] * nums[i+1] * ... * nums[n-1]
#           = 前缀乘积（不含 i） * 后缀乘积（不含 i）
#
# 定义：
# - preMulti[i] = nums[0] ~ nums[i] 的乘积
# - postMulti[i] = nums[n-1-i] ~ nums[n-1] 的乘积（从后往前）
#
# result[i] = preMulti[i-1] * postMulti[n-1-(i+1)]
#           = nums[0]~nums[i-1] 乘积 * nums[i+1]~nums[n-1] 乘积
#
# 易错点：
# - 第一个元素：没有前缀，result[0] = postMulti[n-2] = nums[1]~nums[n-1] 乘积
# - 最后一个元素：没有后缀，result[n-1] = preMulti[n-2] = nums[0]~nums[n-2] 乘积
# - postMulti 索引计算复杂，用 n-1-(i+1) 避免
#
# 时间复杂度：O(n)
# 空间复杂度：O(n)


# 优化版：O(1) 空间（不含输出数组）
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [1] * n

        # 前缀乘积直接存到 result
        prefix = 1
        for i in range(n):
            result[i] = prefix
            prefix *= nums[i]

        # 后缀乘积直接乘到 result
        postfix = 1
        for i in range(n - 1, -1, -1):
            result[i] *= postfix
            postfix *= nums[i]

        return result