class Solution(object):
    def removeDuplicates(self, s):
        """
        :type s: str
        :rtype: str
        """
        from collections import deque
        stack=deque()
        for i in range(len(s)):
            if bool(stack) and stack[-1]==s[i]:
                stack.pop()
            else:
                stack.append(s[i])
        return ''.join(stack)