# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.result = float('-inf')

        def maxGains(root):
            if not root: return 0
            leftGain = maxGains(root.left)
            rightGain = maxGains(root.right)

            # 当前节点作为"拐点"的路径和（左+根+右）
            curGain = root.val + max(leftGain, 0) + max(rightGain, 0)
            self.result = max(curGain, self.result)

            # 返回：只能选一条子树路径向上传递（左+根 或 右+根）
            return root.val + max(max(leftGain, rightGain), 0)

        maxGains(root)
        return self.result