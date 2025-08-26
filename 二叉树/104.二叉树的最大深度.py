# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if root==None:
            return 0
        def getHeight(root):
            if root.left==None and root.right==None:
                return 1
            if root.left and root.right==None:
                return getHeight(root.left)+1
            if root.right and root.left==None:
                return getHeight(root.right)+1
            return max(getHeight(root.left),getHeight(root.right))+1
        return getHeight(root)
