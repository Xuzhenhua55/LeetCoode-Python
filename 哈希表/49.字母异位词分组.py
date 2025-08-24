# 暴力做法 能过 但是会超时
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        from collections import defaultdict
        usedList=[0]*len(strs)
        resultList=[]
        for i in range(len(strs)):
            if usedList[i]==1:
                continue
            usedList[i] = 1
            str=strs[i]
            iList=[str]
            strCountDict=defaultdict(int)
            for ch in str:
                strCountDict[ch]+=1
            for j in range(i+1,len(strs)):
                if usedList[j]==1:
                    continue
                jStr=strs[j]
                jStrCountDict=defaultdict(int)
                for ch in jStr:
                    jStrCountDict[ch]+=1
                if strCountDict==jStrCountDict:
                    iList.append(jStr)
                    usedList[j]=1
            resultList.append(iList)
        return resultList
            

class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        str_dict=dict()
        for str in strs:
            sortedStr=''.join(sorted(str))
            if sortedStr not in str_dict:
                str_dict[sortedStr] = [str]
            else:
                str_dict[sortedStr].append(str)
        return list(str_dict.values())

# https://www.yuque.com/u29134184/skgkmw/at07daybxeeeep6g?singleDoc# 《笔记：为什么 dict.get(...).append(...) 会导致 None？》
