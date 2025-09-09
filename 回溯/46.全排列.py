class Solution(object):

    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.curList = []
        self.resultList = []
        self.pathSet = set()

        def DFS():
            if len(self.curList) == len(nums):
                self.resultList.append(list(self.curList))
                return
            for i in range(0, len(nums)):
                if nums[i] in self.pathSet:
                    continue
                self.pathSet.add(nums[i])
                self.curList.append(nums[i])
                DFS()
                self.curList.pop()
                self.pathSet.remove(nums[i])

        DFS()
        return self.resultList
