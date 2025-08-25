class Solution(object):
    def isValid(self, s):
        from collections import deque
        """
        :type s: str
        :rtype: bool
        """
        mapDict=dict()
        mapDict[')'] = '('
        mapDict['}'] = '{'
        mapDict[']'] = '['
        stack=deque()
        for i in range(len(s)):
            if stack and s[i] in mapDict and stack[-1]==mapDict[s[i]]:
                stack.pop()
            else:
                stack.append(s[i])
        return not stack
        