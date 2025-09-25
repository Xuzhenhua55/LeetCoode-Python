# Definition for a binary tree node.
class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 每个节点有不同状态：2被覆盖、0未被覆盖、1是摄像头
class Solution(object):

    def minCameraCover(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        self.result = 0

        def dfs(root):
            if not root: return 2
            leftStatus = dfs(root.left)
            rightStatus = dfs(root.right)
            if leftStatus == 0 or rightStatus == 0:
                self.result += 1
                return 1
            if leftStatus == 1 or rightStatus == 1: return 2

            if leftStatus == 2 and rightStatus == 2: return 0

        if (dfs(root) == 0): self.result += 1
        return self.result
