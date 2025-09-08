class Solution(object):

    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.curList = []
        self.resultList = []

        def DFS(start):
            if len(self.curList) >= 2:
                self.resultList.append(list(self.curList))
            if start > len(nums) - 1: return
            appearedSet = set()
            for i in range(start, len(nums)):
                if i > start and nums[i] in appearedSet:
                    continue
                appearedSet.add(nums[i])
                if not self.curList or (bool(self.curList)
                                        and nums[i] >= self.curList[-1]):
                    self.curList.append(nums[i])
                    DFS(i + 1)
                    self.curList.pop()

        DFS(0)
        return self.resultList
