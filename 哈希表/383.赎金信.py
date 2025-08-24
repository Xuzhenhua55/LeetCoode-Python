class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        magazineCountList=[0]*26
        for ch in magazine:
            magazineCountList[ord(ch)-ord('a')]+=1
        for ch in ransomNote:
            magazineCountList[ord(ch)-ord('a')]-=1
        return all(x>=0 for x in magazineCountList)