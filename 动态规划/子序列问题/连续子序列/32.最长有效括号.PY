# 如果是左括号 那直接加入栈中 没有问题
# 如果是有括号 那么如果栈中有左括号与之对应 那么必然意味着多了一个合法序列 那么 有可能是对(前的序列的扩展 也可能是对)前一位的进行扩展 所以综合相加
# 其中dp[i]表示前i个元素最长有效括号
class Solution(object):

    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack = []
        dp = [0] * (len(s) + 1)
        for i, ch in enumerate(s):
            if ch == "(":
                stack.append(i)
            else:
                if stack:
                    leftIndex = stack.pop()
                    dp[i + 1] = dp[leftIndex] + dp[i] + 2
        return max(dp)
