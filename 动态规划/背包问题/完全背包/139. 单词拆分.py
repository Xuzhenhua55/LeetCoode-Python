class Solution(object):

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for j in range(1, len(dp)):
            for i in range(len(wordDict)):
                if j >= len(wordDict[i]):
                    if dp[j - len(wordDict[i])] and wordDict[i] == s[
                            j - len(wordDict[i]):j]:
                        dp[j] = True
                        break
        return dp[-1]


Solution().wordBreak("leetcode", ["leet", "code"])
