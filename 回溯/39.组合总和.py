class Solution(object):

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.curList = []
        self.curSum = 0
        self.resultList = []

        def DFS(start, end):
            if self.curSum == target:
                self.resultList.append(list(self.curList))
                return
            if self.curSum > target: return

            for i in range(start, end + 1):
                self.curList.append(candidates[i])
                self.curSum += candidates[i]
                DFS(i, end)
                self.curList.pop()
                self.curSum -= candidates[i]

        DFS(0, len(candidates) - 1)
        return self.resultList
