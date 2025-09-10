class Solution(object):

    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        g.sort(key=lambda x: -x)
        s.sort(key=lambda x: -x)
        sum = 0
        childIndex = 0
        for cake in s:
            for i in range(childIndex, len(g)):
                if cake >= g[i]:
                    sum += 1
                    childIndex = i + 1
                    break
        return sum


class Solution(object):

    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        g.sort(key=lambda x: -x)
        s.sort(key=lambda x: -x)
        i = j = sum = 0
        while i < len(g) and j < len(s):
            if g[i] <= s[j]:
                sum += 1
                i += 1
                j += 1
            else:
                i += 1
        return sum


Solution().findContentChildren([1, 2, 3], [1, 1])
