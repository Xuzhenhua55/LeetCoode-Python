# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):

    def convertBST(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: Optional[TreeNode]
        """
        self.curSum = 0

        def DFS(root):
            if root == None: return 0
            DFS(root.right)
            self.curSum += root.val
            root.val = self.curSum
            DFS(root.left)

        DFS(root)
        return root
