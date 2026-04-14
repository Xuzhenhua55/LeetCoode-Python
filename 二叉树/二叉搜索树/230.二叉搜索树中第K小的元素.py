# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.curCnt = 0
        self.result = None

        def inOrder(root):
            if not root or self.result is not None:  # 找到结果后提前终止
                return

            inOrder(root.left)
            self.curCnt += 1
            if self.curCnt == k:
                self.result = root.val
                return  # 找到后立即返回，避免继续遍历
            inOrder(root.right)

        inOrder(root)
        return self.result


# 核心思路：BST 的中序遍历是升序序列
# 第 k 小的元素就是中序遍历的第 k 个元素
#
# 优化：找到结果后提前终止
# - if self.result is not None: return
# - 找到后不再继续遍历右子树，节省时间
#
# 其他解法：
# 1. 迭代版中序遍历：用栈模拟，找到第 k 个时停止
# 2. 记录子树节点数：预处理每个节点的左子树大小，O(h) 查找


# 迭代版解法
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        cur = root

        while cur or stack:
            while cur:  # 一直往左走到底
                stack.append(cur)
                cur = cur.left

            cur = stack.pop()  # 访问当前节点
            k -= 1
            if k == 0:
                return cur.val

            cur = cur.right  # 转向右子树