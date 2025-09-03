# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def getMinimumDifference(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        self.preVal=None
        self.result=float('inf')
        def DFS(root):
            if not root: return
            DFS(root.left)
            if self.preVal!=None:
                self.result=min(abs(self.preVal-root.val),self.result)
            self.preVal = root.val
            DFS(root.right)
        DFS(root)
        return self.result
            
