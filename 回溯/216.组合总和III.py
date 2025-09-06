class Solution(object):

    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        self.resultList = []
        self.curList = []
        self.curSum = 0

        def DFS(start, end):
            if self.curSum == n and len(self.curList) == k:
                self.resultList.append(list(self.curList))
                return
            if start > end: return
            if end - start + 1 < k - len(self.curList): return
            if self.curSum > n: return
            for i in range(start, end + 1):
                self.curList.append(i)
                self.curSum += i
                DFS(i + 1, end)
                self.curList.pop()
                self.curSum -= i

        DFS(1, 9)
        return self.resultList
