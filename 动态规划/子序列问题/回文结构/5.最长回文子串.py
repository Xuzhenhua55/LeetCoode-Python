class Solution(object):

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        dp = [[False] * len(s) for _ in range(len(s))]
        result = ""
        for i in range(len(s) - 1, -1, -1):
            for j in range(i, len(s)):
                if s[i] != s[j]: continue
                if j - i <= 1:
                    dp[i][j] = True
                    if j - i + 1 > len(result):
                        result = s[i:j + 1]
                else:
                    if dp[i + 1][j - 1]:
                        dp[i][j] = True
                        if j - i + 1 > len(result):
                            result = s[i:j + 1]
        return result
