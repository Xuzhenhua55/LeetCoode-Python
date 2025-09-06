class Solution(object):

    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        self.resultList = []
        self.curList = []

        def DFS(start, end):
            if len(self.curList) == k:
                self.resultList.append(list(self.curList))
                return
            if start > end: return
            if end - start + 1 < k - len(self.curList): return
            for i in range(start, end + 1):
                self.curList.append(i)
                DFS(i + 1, end)
                self.curList.pop()

        DFS(1, n)
        return self.resultList


Solution().combine(4, 2)
