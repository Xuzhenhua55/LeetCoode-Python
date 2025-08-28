# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: Optional[TreeNode]
        :type targetSum: int
        :rtype: bool
        """
        self.hasPathSum=False
        def DFS(root,targetSum,curSum):
            if self.hasPathSum:
                return
            curSum+=root.val
            if not root.left and not root.right and targetSum==curSum:
                self.hasPathSum=True
                return
            if root.left:
                DFS(root.left,targetSum,curSum)
            if root.right:
                DFS(root.right,targetSum,curSum)
        if root: DFS(root,targetSum,0)
        return self.hasPathSum