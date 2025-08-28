# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        def getHeight(root):
            if not root: return 0
            return max(getHeight(root.left),getHeight(root.right))+1
        if not root or (not root.left and not root.right): return True
        return abs(getHeight(root.left)-getHeight(root.right))<=1 and self.isBalanced(root.left) and self.isBalanced(root.right)
    

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        self.isBalanced=True
        def DFS(root):
            if not root: return 0
            if not self.isBalanced:
                return 0
            leftDepth=DFS(root.left)
            rightDepth=DFS(root.right)
            if abs(leftDepth-rightDepth)>1: self.isBalanced=False
            return max(leftDepth,rightDepth)+1
        DFS(root)
        return self.isBalanced
            
