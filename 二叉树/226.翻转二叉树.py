# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def invertTree(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: Optional[TreeNode]
        """
        if not root: return None
        leftNode,rightNode=None,None
        if root.left: 
            leftNode=self.invertTree(root.left)
        if root.right: 
            rightNode=self.invertTree(root.right)
        root.left,root.right=rightNode,leftNode
        return root
        