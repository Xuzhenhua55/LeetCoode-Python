# dp[i][j][k]表示在0~i中选，满足<=j个0 <=k个1的最大子集的长度
# 当j k小于对应的新子串的zeroCount oneCount时，只能选择继承之前的最大长度，因为无法新增
# 当j k大于对应的新子串的zeroCount oneCount时，如果不加入，那么就是继承，如果加入，那么就在dp[i][j-zeroCount][k-oneCount]的基础上+1
class Solution(object):

    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """

        def countZeroAndOne(s):
            zeroCount = oneCount = 0
            for ch in s:
                if ch == '0': zeroCount += 1
                else: oneCount += 1
            return zeroCount, oneCount

        fullStr = ''.join(strs)
        zeroCount, oneCount = countZeroAndOne(fullStr)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        zeroCount, oneCount = countZeroAndOne(strs[0])
        for i in range(zeroCount, len(dp)):
            for j in range(oneCount, len(dp[0])):
                dp[i][j] = 1
        # print(dp)
        for i in range(1, len(strs)):
            zeroCount, oneCount = countZeroAndOne(strs[i])
            for j in range(len(dp) - 1, -1, -1):
                for k in range(len(dp[0]) - 1, -1, -1):
                    if j - zeroCount >= 0 and k - oneCount >= 0:
                        # dp[j][k] = max(
                        #     max(dp[j - zeroCount][k - oneCount] + 1, dp[j][k]),
                        #     1)
                        # 由于如果dp[j - zeroCount][k - oneCount]为0的话+1必然已经不需要再和1进行对比了，所以一个max就能解决
                        dp[j][k] = max(dp[j - zeroCount][k - oneCount] + 1,
                                       dp[j][k])
                    else:
                        dp[j][k] = dp[j][k]
            # print(dp)
        return dp[m][n]


print(Solution().findMaxForm(["10", "0001", "111001", "1", "0"], 50, 50))
