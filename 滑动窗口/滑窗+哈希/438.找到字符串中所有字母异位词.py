class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        if len(s)<len(p):
            return []
        resultList=[]
        pDict=dict()
        for ch in p:
            pDict[ch]=pDict.get(ch,0)+1
        subStrDict=dict()
        left=0
        for right in range(0,len(s)):
            subStrDict[s[right]] = subStrDict.get(s[right], 0)+1
            if right-left+1==len(p):
                if subStrDict == pDict:
                    resultList.append(left)
                subStrDict[s[left]]-=1
                if subStrDict[s[left]]==0:
                    del subStrDict[s[left]]
                left+=1

        return resultList