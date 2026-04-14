class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.result = float('-inf')

        def maxHeight(root):
            if not root: return 0
            leftHeight = maxHeight(root.left)
            rightHeight = maxHeight(root.right)

            # 当前节点作为"拐点"时的直径（节点数 - 1 = 边数）
            curHeight = leftHeight + rightHeight + 1
            self.result = max(curHeight - 1, self.result)

            # 返回：只能选一条子树路径向上传递
            return max(leftHeight, rightHeight) + 1

        maxHeight(root)
        return self.result