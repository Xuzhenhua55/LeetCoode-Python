class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        s=list(s)
        left=0
        while left<len(s):
            _left,right=left,min(left+k-1,len(s)-1)
            while _left < right:
                s[_left], s[right] = s[right], s[_left]
                _left += 1
                right-=1
            left=min(left+2*k,len(s))
        return ''.join(s)