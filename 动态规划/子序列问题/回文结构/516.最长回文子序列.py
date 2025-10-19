class Solution(object):

    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        # dp[i][j]表示[i,j]最长的回文子序列
        dp = [[0] * len(s) for _ in range(len(s))]
        for i in range(len(s) - 1, -1, -1):
            for j in range(i, len(s)):
                if s[i] != s[j]: dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
                else:
                    if j - i <= 1: dp[i][j] = j - i + 1
                    else: dp[i][j] = dp[i + 1][j - 1] + 2
        # print(dp)
        return dp[0][-1]


Solution().longestPalindromeSubseq("abcdef")
