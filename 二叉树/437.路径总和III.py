# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import defaultdict

class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        # 前缀和 + 哈希表：类似数组的"和为 K 的子数组"
        self.sumToCnt = defaultdict(int)  # 记录每个前缀和出现的次数
        self.sumToCnt[0] = 1  # 空路径的前缀和为 0
        self.curSum = 0
        self.result = 0

        def preOrder(root):
            if not root:
                return

            # 前序位置：进入节点，更新前缀和
            self.curSum += root.val

            # 检查是否存在前缀和 = curSum - targetSum
            # 如果存在，说明从那个位置到当前位置的路径和 = targetSum
            if (self.curSum - targetSum) in self.sumToCnt:
                self.result += self.sumToCnt[self.curSum - targetSum]

            # 当前前缀和加入哈希表
            self.sumToCnt[self.curSum] += 1

            # 递归处理子树
            preOrder(root.left)
            preOrder(root.right)

            # 后序位置：离开节点，撤销前缀和（回溯）
            self.sumToCnt[self.curSum] -= 1
            self.curSum -= root.val

        preOrder(root)
        return self.result


# ============================================================
# 核心思路解析：前缀和 + 哈希表 + 回溯
# ============================================================
#
# 问题：找路径和等于 targetSum 的路径数量
#
# 前缀和定义：
# - 从根节点到当前节点的路径上所有节点值的和
# - 如果两个节点的前缀和差值 = targetSum，则这段路径和 = targetSum
#
# 思路：
# 1. 用哈希表记录每个前缀和出现的次数
# 2. 遍历到节点时，检查是否存在前缀和 = curSum - targetSum
# 3. 如果存在，说明从那个前缀和位置到当前位置的路径和 = targetSum
# 4. 回溯时撤销当前前缀和（离开节点时从哈希表移除）
#
# 类比：和数组题"560. 和为 K 的子数组"完全一样
# - 数组：一维前缀和
# - 二叉树：路径前缀和（需要在回溯时撤销）
#
# 易错点：
# - defaultdict(int) 必须指定 int，否则默认值是 None
# - self.result 不是 result（内部函数修改外部变量）
# - 回溯时 sumToCnt[curSum] -= 1，不是直接删除（可能多次出现）