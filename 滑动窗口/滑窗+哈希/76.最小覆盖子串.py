class Solution(object):
    # 2中的每一个key对应的value
    def needShrink(self, dict1, dict2, ch):
        contain = True
        for key in dict2:
            if dict1.get(key, 0) < dict2[key]:
                contain = False
                break
        if contain and (ch not in dict2 or dict2[ch] <= dict1[ch]-1):
            return True, True
        else:
            return contain, False

    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        windowDict = dict()
        left, right = 0, 0
        targetDict = dict()
        result = s+" "
        for ch in t:
            targetDict[ch] = targetDict.get(ch, 0)+1
        while right < len(s):
            windowDict[s[right]] = windowDict.get(s[right], 0)+1
            contain, needShrink = self.needShrink(
                windowDict, targetDict, s[left])
            while left < right and needShrink:
                windowDict[s[left]] -= 1
                if windowDict[s[left]] == 0:
                    del windowDict[s[left]]
                left += 1
                contain, needShrink = self.needShrink(
                    windowDict, targetDict, s[left])
            if right-left+1 < len(result) and contain:
                result = s[left:right+1]
            right += 1
        if result == s+" ":
            return ""
        else:
            return result
