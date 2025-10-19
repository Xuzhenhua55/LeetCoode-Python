class Solution(object):

    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # dp[i][j]表示从[i,j]是否为回文子串
        result = 0
        dp = [[False] * len(s) for _ in range(len(s))]
        for i in range(len(s) - 1, -1, -1):
            for j in range(i, len(s)):
                if s[i] != s[j]: continue
                if j - i <= 1:
                    dp[i][j] = True
                    result += 1
                else:
                    if dp[i + 1][j - 1]:
                        dp[i][j] = True
                        result += 1
        return result
