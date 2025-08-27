# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def countNodes(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if root==None: return 0
        leftDepth,rightDepth=0,0
        left,right=root.left,root.right
        while left:
            leftDepth+=1
            left=left.left
        while right:
            rightDepth+=1
            right=right.right
        if leftDepth == rightDepth:
            return pow(2, leftDepth+1)-1
        return self.countNodes(root.left)+self.countNodes(root.right)+1
            