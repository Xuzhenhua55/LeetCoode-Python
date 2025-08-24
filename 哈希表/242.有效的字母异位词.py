class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        sCountDict=dict()
        tCountDict=dict()
        for ch in s:
            sCountDict[ch]=sCountDict.get(ch,0)+1
        for ch in t:
            tCountDict[ch]=tCountDict.get(ch,0)+1
        for ch in s:
            if ch not in tCountDict:
                return False
            tCountDict[ch]-=1
        for ch in tCountDict:
            if tCountDict[ch]!=0:
                return False
        return True
        

class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        sCountDict = dict()
        tCountDict = dict()
        for ch in s:
            sCountDict[ch] = sCountDict.get(ch, 0)+1
        for ch in t:
            tCountDict[ch] = tCountDict.get(ch, 0)+1
        return tCountDict==sCountDict