# 1.两个石头相撞，结果要么为x-y，要么为y-x
# 2.无论你怎么两两相碰，永远有的数字前为正号，有的为负号,因此你总可以把最终式化为一堆和减去另外一堆数字和
# 3.因此我们要找的是这个集合的两个子集之和的最小差
# 4.要想子集之和差最小，则两者应该尽量接近或者相等
# 5.这个时候我们就可以把sum/2作为背包容量，使用01背包来解题了
class Solution(object):

    def lastStoneWeightII(self, stones):
        """
        :type stones: List[int]
        :rtype: int
        """
        stonesSum = sum(stones)
        dp = [0] * (stonesSum // 2 + 1)
        for i in range(len(stones)):
            for j in range(len(dp) - 1, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return stonesSum - dp[-1] - dp[-1]
