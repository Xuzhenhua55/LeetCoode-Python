class Solution(object):

    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.curList = []
        self.resultList = []
        nums.sort()

        def DFS(start):
            self.resultList.append(list(self.curList))
            if start > len(nums) - 1: return
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]: continue
                self.curList.append(nums[i])
                DFS(i + 1)
                self.curList.pop()

        DFS(0)
        return self.resultList
