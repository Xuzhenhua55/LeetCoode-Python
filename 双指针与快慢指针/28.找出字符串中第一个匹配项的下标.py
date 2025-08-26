class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        resultList=[]
        for left in range(len(haystack)):
            if haystack[left]!=needle[0]:
                continue
            if left+len(needle)>len(haystack):
                break
            matched=True
            for bias in range(len(needle)):
                if haystack[left+bias]!=needle[bias]:
                    matched=False
                    break
            if matched:
                return left
        return -1

# TODO：使用KMP算法完成该题