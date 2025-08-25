# class Solution(object):
#     def reverseWords(self, s):
#         """
#         :type s: str
#         :rtype: str
#         """
#         wordList = s.strip().split()
#         wordList.reverse()
#         return " ".join(wordList)
    

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s=' '+s
        right = len(s)-1
        resultList=[]
        while right>0:
            while right>=0 and s[right]==' ': right-=1
            if right<0:break
            left=right-1
            while s[left]!=' ' and left-1>=0: left-=1
            resultList.append(s[left+1:right+1])
            right=left
        return ' '.join(resultList)

s=Solution()
s.reverseWords("the sky is blue")
            
        