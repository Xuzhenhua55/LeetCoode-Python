class Solution(object):

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        self.curList = []
        self.resultList = []

        def isValid(str):
            if str[0] == '0' and len(str) > 1: return False
            return int(str) >= 0 and int(str) <= 255

        def DFS(subStr):
            if len(self.curList) == 4 and not subStr:
                self.resultList.append('.'.join(self.curList))
            if 4 - len(self.curList) > len(subStr): return
            for i in range(0, len(subStr)):
                leftStr = subStr[0:i + 1]
                if not isValid(leftStr): continue
                self.curList.append(leftStr)
                DFS(subStr[i + 1:len(subStr)])
                self.curList.pop()

        DFS(s)
        return self.resultList


Solution().restoreIpAddresses("25525511135")
