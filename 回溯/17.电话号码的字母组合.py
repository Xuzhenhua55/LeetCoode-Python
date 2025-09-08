class Solution(object):

    def initNumberDict(self):
        self.numberDict = {
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        self.initNumberDict()
        self.resultList = []
        self.curChList = []

        def DFS(start, end):
            if start > end and len(self.curChList) == len(digits):
                self.resultList.append(''.join(self.curChList))
                return
            if end - start + 1 < len(digits) - len(self.curChList): return
            for i in range(start, end + 1):
                curChoiceList = self.numberDict[digits[i]]
                for ch in curChoiceList:
                    self.curChList.append(ch)
                    DFS(i + 1, end)
                    self.curChList.pop()

        DFS(0, len(digits) - 1)
        return self.resultList


class Solution(object):

    def initNumberDict(self):
        self.numberDict = {
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        self.initNumberDict()
        self.resultList = []
        self.curChList = []

        def DFS(start, end):
            if start > end:
                self.resultList.append(''.join(self.curChList))
                return
            curChoiceList = self.numberDict[digits[start]]
            for ch in curChoiceList:
                self.curChList.append(ch)
                DFS(start + 1, end)
                self.curChList.pop()

        DFS(0, len(digits) - 1)
        return self.resultList


print(Solution().letterCombinations("23"))
