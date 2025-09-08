class Solution(object):

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.curList = []
        self.resultList = []

        def DFS(start):
            self.resultList.append(list(self.curList))
            if start > len(nums) - 1: return
            for i in range(start, len(nums)):
                self.curList.append(nums[i])
                DFS(i + 1)
                self.curList.pop()

        DFS(0)
        return self.resultList
