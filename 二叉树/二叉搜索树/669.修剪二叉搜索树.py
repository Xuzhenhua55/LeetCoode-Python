# Definition for a binary tree node.
class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution(object):

    def trimBST(self, root, low, high):
        """
        :type root: Optional[TreeNode]
        :type low: int
        :type high: int
        :rtype: Optional[TreeNode]
        """
        if root == None: return None
        if root.val > high or root.val < low:
            if root.left == None and root.right == None:
                return None
            if (root.left == None and root.right) or root.val < low:
                return self.trimBST(root.right, low, high)
            if (root.left and root.right == None) or root.val > high:
                return self.trimBST(root.left, low, high)
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        return root
