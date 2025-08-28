# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        self.resultVal=0
        def DFS(root,direction):
            if not root.left and not root.right and direction=='left':
                self.resultVal += root.val
            if root.left: DFS(root.left,"left")
            if root.right: DFS(root.right,"right")
        if root: DFS(root,"right")
        return self.resultVal
            
            