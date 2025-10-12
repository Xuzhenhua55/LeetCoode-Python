# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):

    def rob(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """

        def DFS(root):
            if not root: return [0, 0]
            left, right = DFS(root.left), DFS(root.right)
            return [
                max(left[0], left[1]) + max(right[0], right[1]),
                root.val + left[0] + right[0]
            ]

        result = DFS(root)
        return max(result[0], result[1])
