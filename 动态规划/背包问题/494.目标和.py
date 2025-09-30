class Solution(object):

    def findTargetSumWays(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        self.curSum = 0
        self.curCount = 0
        self.result = 0

        def DFS(start):
            if self.curCount == len(nums):
                if self.curSum == target:
                    self.result += 1

            for i in range(start, len(nums)):
                self.curSum += nums[i]
                self.curCount += 1
                DFS(i + 1)
                self.curSum -= nums[i]
                self.curCount -= 1

                self.curSum -= nums[i]
                self.curCount += 1
                DFS(i + 1)
                self.curSum += nums[i]
                self.curCount -= 1

        DFS(0)
        return self.result


print(Solution().findTargetSumWays([1, 1, 1, 1, 1], 3))


# 其实可以将dp[i][j]定义为 假设从0~i索引对应的物品中随意按照规则组合，那么能够组成target的种数
# 那么dp[i][j]的递推其实也比较自然了，当遍历一个新的物品时，如果选他作为+号，就相当于有dp[i-1][j-nums[i]]种方案 如果选它作为-号 那么就有dp[i-1][j+nums[i]]种方案
# 但是j-nums[i]这个东西 可能变成负数，因此需要将[-numsSum,numsSum]作为初始化的区间，由于数组索引非负，需要一个bias，但是在真实遍历的过程中可以忽略bias
# 有一个坑是 必须完整的序列才行，所以dp[i-1][j]其实是不能继承的
# 由于既有+又有-因此本质上没办法压缩成一维的
class Solution(object):

    def findTargetSumWays(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        numsSum = sum(nums)
        if numsSum < target or -numsSum > target: return 0
        dp = [[0] * (2 * (numsSum + 1)) for _ in range(len(nums))]
        bias = numsSum + 1
        dp[0][nums[0] + bias] += 1
        dp[0][-nums[0] + bias] += 1
        for i in range(1, len(nums)):
            for j in range(0, len(dp[0])):
                if j - nums[i] >= 0 and dp[i - 1][j - nums[i]] != 0:
                    dp[i][j] += dp[i - 1][j - nums[i]]
                if j + nums[i] <= len(dp[0]) - 1 and dp[i - 1][j +
                                                               nums[i]] != 0:
                    dp[i][j] += dp[i - 1][j + nums[i]]

            # print(dp)
        return dp[-1][target + bias]


print(Solution().findTargetSumWays([1, 1, 1, 1, 1], 3))


# 本质上的最终目标是 x - (sum - x) = target
# 所以需要求 x=(sum+target)/2 是否能够由nums中的元素求和得到 有多少种方案
# 需要注意的一个坑是0这个元素的特殊性，导致第一列需要单独初始化
class Solution(object):

    def findTargetSumWays(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        numsSum = sum(nums)
        if (numsSum < abs(target)): return 0
        if (numsSum + target) % 2 == 1: return 0
        x = int((numsSum + target) / 2)
        dp = [[0] * (x + 1) for _ in range(len(nums))]
        if nums[0] < len(dp[0]): dp[0][nums[0]] = 1
        zeroCount = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                zeroCount += 1
            dp[i][0] = int(pow(2.0, zeroCount))
        for i in range(1, len(nums)):
            for j in range(0, len(dp[0])):
                if j >= nums[i]:
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]]
                else:
                    dp[i][j] = dp[i - 1][j]
            print(dp[i])
        return dp[-1][x]


print(Solution().findTargetSumWays([1, 1, 1, 1, 1], 3))


class Solution(object):

    def findTargetSumWays(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        numsSum = sum(nums)
        if (numsSum < abs(target)): return 0
        if (numsSum + target) % 2 == 1: return 0
        x = int((numsSum + target) / 2)
        dp = [0] * (x + 1)
        if nums[0] < len(dp): dp[nums[0]] = 1
        zeroCount = 0
        if nums[0] == 0:
            dp[0] = 2
            zeroCount += 1
        else:
            dp[0] = 1
        for i in range(1, len(nums)):
            for j in range(len(dp) - 1, -1, -1):
                if j >= nums[i]:
                    dp[j] = dp[j] + dp[j - nums[i]]
                else:
                    dp[j] = dp[j]
            print(dp)
            if nums[i] == 0:
                zeroCount += 1
                dp[0] = int(pow(2.0, zeroCount))
        return dp[x]


print(Solution().findTargetSumWays([1, 1, 1, 1, 1], 3))
