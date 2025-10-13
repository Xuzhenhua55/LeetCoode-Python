class Solution(object):

    def longestCommonSubsequence(self, text1, text2):
        """
        :type text1: str
        :type text2: str
        :rtype: int
        """
        dp = [[0] * len(text2) for _ in range(len(text1))]
        for i in range(len(text1)):
            if text1[i] == text2[0]: dp[i][0] = 1
            elif text1[i] != text2[0] and i > 0: dp[i][0] = dp[i - 1][0]
        for j in range(len(text2)):
            if text1[0] == text2[j]: dp[0][j] = 1
            elif text1[0] != text2[j] and j > 0: dp[0][j] = dp[0][j - 1]
        for i in range(1, len(text1)):
            for j in range(1, len(text2)):
                if text1[i] == text2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        print(dp)
        return dp[-1][-1]


Solution().longestCommonSubsequence("abcde", "ace")
