from time import sleep
from typing import List


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


# 解法四：用 startIndex 控制切割位置（统一回溯模板）
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def isPalindrome(targetStr):
            left, right = 0, len(targetStr) - 1
            while left < right:
                if targetStr[left] != targetStr[right]: return False
                left += 1
                right -= 1
            return True

        self.curPath = list()
        self.result = list()

        def backtracing(startIndex):
            # 递归开头：剪枝（最后一个分割必须是回文）
            if self.curPath and not isPalindrome(self.curPath[-1]): return
            # 收集结果
            if startIndex == len(s):
                self.result.append(list(self.curPath))
                return

            # 循环体：尝试所有切割位置
            for right in range(startIndex + 1, len(s) + 1):
                leftStr = s[startIndex:right]
                self.curPath.append(leftStr)
                backtracing(right)
                self.curPath.pop()

        backtracing(0)
        return self.result
