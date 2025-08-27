# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        if root==None:
            return True
        def isEqual(leftNode,rightNode):
            if (leftNode and not rightNode) or (rightNode and not leftNode):
                return False
            if not leftNode and not rightNode:
                return True
            return leftNode.val ==rightNode.val and isEqual(leftNode.left,rightNode.right) and isEqual(leftNode.right,rightNode.left)
        return isEqual(root.left,root.right)