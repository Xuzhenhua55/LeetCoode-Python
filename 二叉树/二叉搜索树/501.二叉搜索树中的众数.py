# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def findMode(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """
        self.preVal=None
        self.curCount=0
        self.maxCount=-float('inf')
        self.resultList=[]
        def DFS(root):
            if not root:
                return
            DFS(root.left)
            if self.preVal != None and root.val == self.preVal:
                self.curCount+=1
            else:
                self.curCount=1
            if self.curCount > self.maxCount:
                self.resultList=[]
                self.resultList.append(root.val)
                self.maxCount=self.curCount
            elif self.curCount == self.maxCount:
                self.resultList.append(root.val)
            self.preVal=root.val
            DFS(root.right)
        DFS(root)
        return self.resultList
