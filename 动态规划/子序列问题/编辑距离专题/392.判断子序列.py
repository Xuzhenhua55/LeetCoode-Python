class Solution(object):

    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) == 0: return True
        if len(t) == 0: return False
        dp = [[False] * len(t) for _ in range(len(s))]
        for j in range(len(t)):
            if s[0] == t[j]: dp[0][j] = True
            elif s[0] != t[j] and j > 0: dp[0][j] = dp[0][j - 1]
        for i in range(1, len(s)):
            for j in range(1, len(t)):
                if s[i] == t[j]: dp[i][j] = dp[i - 1][j - 1] and True
                else: dp[i][j] = dp[i][j - 1]
        # print(dp)
        return dp[-1][-1]


class Solution(object):

    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) == 0: return True
        if len(t) == 0: return False
        dp = [[0] * (len(t) + 1) for _ in range(len(s) + 1)]
        for i in range(1, len(s) + 1):
            for j in range(1, len(t) + 1):
                if s[i - 1] == t[j - 1]: dp[i][j] = dp[i - 1][j - 1] + 1
                else: dp[i][j] = dp[i][j - 1]
        # print(dp)
        return dp[-1][-1] == len(s)


Solution().isSubsequence("abc", "ahbgdc")
