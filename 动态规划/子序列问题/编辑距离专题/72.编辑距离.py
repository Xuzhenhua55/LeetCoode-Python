# 相比583.两个字符串的删除操作多了一个可以修改的选项，因此在不匹配的过程中dp的状态转移可以从三个旧状态进行转移
class Solution(object):

    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        for i in range(len(word1) + 1):
            dp[i][0] = i
        for j in range(len(word2) + 1):
            dp[0][j] = j
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]: dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]),
                                   dp[i - 1][j - 1]) + 1
        return dp[-1][-1]
