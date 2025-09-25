# 如果当前字母出现在已有的区间中，那么必然需要加入区间
# 如果当前字母不出现在已有区间中，判断先前字母是否已经在全局中没有后续了，如果还有后续意味着还需要继续添加，如果没有了意味着此时分隔是最优的
class Solution(object):

    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        from collections import defaultdict
        countDict = defaultdict(int)
        resultList = []
        left = 0
        for ch in s:
            countDict[ch] += 1
        chSet = set()

        def isEnding():
            for wdCh in chSet:
                if countDict[wdCh] != 0:
                    return False
            return True

        for i in range(len(s)):
            ch = s[i]
            ending = isEnding()
            if chSet and ch not in chSet and ending:
                chSet.clear()
                resultList.append(i - left)
                left = i
            chSet.add(ch)
            countDict[ch] -= 1
        if isEnding():
            resultList.append(len(s) - left)
        return resultList


class Solution(object):

    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        intervalDict = dict()
        for i in range(len(s)):
            if s[i] not in intervalDict:
                intervalDict[s[i]] = [i, i]
            else:
                intervalDict[s[i]][1] = i
        left = 0
        maxRight = -1
        resultList = []
        for i in range(len(s)):
            interval = intervalDict[s[i]]
            if maxRight == -1 or interval[0] <= maxRight:
                maxRight = max(interval[1], maxRight)
            else:
                resultList.append(i - left)
                left = i
                maxRight = interval[1]
        resultList.append(len(s) - left)
        return resultList


class Solution(object):

    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        maxRightDct = dict()
        for i in range(len(s)):
            maxRightDct[s[i]] = i
        left = 0
        maxRight = 0
        resultList = []
        for i in range(len(s)):
            maxRight = max(maxRight, maxRightDct[s[i]])
            if i == maxRight:
                resultList.append(maxRight - left + 1)
                left = i + 1
        return resultList


Solution().partitionLabels("ababcbacadefegdehijhklij")
