class Solution(object):

    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.curList = []
        self.curSum = 0
        self.resultList = []
        candidates.sort()

        def DFS(start, end):
            if self.curSum == target:
                self.resultList.append(list(self.curList))
                return
            if self.curSum > target or start > end: return
            for i in range(start, end + 1):
                if i > 0 and i != start and candidates[i] == candidates[i - 1]:
                    continue
                self.curList.append(candidates[i])
                self.curSum += candidates[i]
                DFS(i + 1, end)
                self.curList.pop()
                self.curSum -= candidates[i]

        DFS(0, len(candidates) - 1)
        return self.resultList


Solution().combinationSum2([10, 1, 2, 7, 6, 1, 5], 8)
