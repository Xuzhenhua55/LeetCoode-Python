from time import sleep


class Solution(object):

    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        self.resultList = []
        self.curList = []

        def isHuiWen(str):
            left, right = 0, len(str) - 1
            while left < right:
                if str[left] != str[right]:
                    return False
                left += 1
                right -= 1
            return True

        def DFS(subStr):
            if subStr and isHuiWen(subStr):
                self.curList.append(subStr)
                self.resultList.append(list(self.curList))
                self.curList.pop()
            if subStr == None: return
            for i in range(0, len(subStr)):
                leftStr = subStr[0:i + 1]
                if isHuiWen(leftStr):
                    self.curList.append(leftStr)
                else:
                    continue
                rightStr = subStr[i + 1:len(subStr)]
                DFS(rightStr)
                self.curList.pop()
            return

        DFS(s)
        return self.resultList


# 本质上是 针对当前子串 分成两截 然后在下一层 判断左右子串是否都是回文串 如果是的话 就连同该子串之前的拆分一同加入到结果集中
# 其中左子串必须回文 因此 可以作为剪枝条件 右子串可以不回文 从而在下一层继续拆分
class Solution(object):

    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        self.resultList = []
        self.curList = []

        def isHuiWen(str):
            left, right = 0, len(str) - 1
            while left < right:
                if str[left] != str[right]:
                    return False
                left += 1
                right -= 1
            return True

        def DFS(subStr):
            if subStr and isHuiWen(subStr):
                self.resultList.append(list(self.curList))
            self.curList.pop()
            if subStr == None: return
            for i in range(0, len(subStr)):
                leftStr = subStr[0:i + 1]
                if isHuiWen(leftStr):
                    self.curList.append(leftStr)
                else:
                    continue
                rightStr = subStr[i + 1:len(subStr)]
                self.curList.append(rightStr)
                DFS(rightStr)
                self.curList.pop()
            return

        self.curList.append(s)
        DFS(s)
        return self.resultList


class Solution(object):

    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        self.resultList = []
        self.curList = []

        def isHuiWen(str):
            left, right = 0, len(str) - 1
            while left < right:
                if str[left] != str[right]:
                    return False
                left += 1
                right -= 1
            return True

        def DFS(subStr):
            if not subStr:
                self.resultList.append(list(self.curList))
            for i in range(0, len(subStr)):
                leftStr = subStr[0:i + 1]
                if isHuiWen(leftStr):
                    self.curList.append(leftStr)
                else:
                    continue
                DFS(subStr[i + 1:len(subStr)])
                self.curList.pop()
            return

        DFS(s)
        return self.resultList
