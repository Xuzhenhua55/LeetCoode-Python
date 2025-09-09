class Solution(object):

    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.curList = []
        self.resultList = []
        self.pathExisted = [False] * len(nums)

        def DFS():
            if len(self.curList) == len(nums):
                self.resultList.append(list(self.curList))
                return
            layerSet = set()
            for i in range(0, len(nums)):
                if nums[i] in layerSet: continue
                if self.pathExisted[i]: continue
                layerSet.add(nums[i])
                self.pathExisted[i] = True
                self.curList.append(nums[i])
                DFS()
                self.curList.pop()
                self.pathExisted[i] = False

        DFS()
        return self.resultList
