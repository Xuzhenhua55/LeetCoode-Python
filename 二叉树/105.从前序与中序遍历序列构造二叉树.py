# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def create(preLeft, preRight, inLeft, inRight):
            if inLeft > inRight:
                return None

            # 前序遍历的第一个元素就是当前子树的根节点
            curVal = preorder[preLeft]

            # 在中序遍历中找到根节点的位置
            midIndex = 0
            for i, val in enumerate(inorder):
                if val == curVal:
                    midIndex = i  # 注意：是赋值 midIndex = i，不是比较 midIndex == i！
                    break

            curNode = TreeNode(curVal)

            # 左子树：前序 [preLeft+1, preLeft+leftSize]，中序 [inLeft, midIndex-1]
            # 右子树：前序 [preLeft+leftSize+1, preRight]，中序 [midIndex+1, inRight]
            leftSize = midIndex - inLeft
            curNode.left = create(preLeft + 1, preLeft + leftSize, inLeft, midIndex - 1)
            curNode.right = create(preLeft + leftSize + 1, preRight, midIndex + 1, inRight)

            return curNode

        return create(0, len(preorder) - 1, 0, len(inorder) - 1)


# 核心思路：
# 1. 前序遍历的第一个元素是根节点
# 2. 在中序遍历中找到根节点位置，左边是左子树，右边是右子树
# 3. 根据左子树大小，在前序遍历中划分左右子树范围
# 4. 递归构建左右子树
#
# 易错点：
# - midIndex = i 是赋值，不是 midIndex == i 比较（返回 True/False）
# - 用 leftSize = midIndex - inLeft 简化参数计算，避免复杂的边界表达式
#
# 优化：用哈希表预处理中序遍历的索引，O(1) 查找根节点位置
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        indexMap = {val: i for i, val in enumerate(inorder)}

        def create(preLeft, preRight, inLeft, inRight):
            if preLeft > preRight:
                return None

            curVal = preorder[preLeft]
            midIndex = indexMap[curVal]  # O(1) 查找
            leftSize = midIndex - inLeft

            curNode = TreeNode(curVal)
            curNode.left = create(preLeft + 1, preLeft + leftSize, inLeft, midIndex - 1)
            curNode.right = create(preLeft + leftSize + 1, preRight, midIndex + 1, inRight)

            return curNode

        return create(0, len(preorder) - 1, 0, len(inorder) - 1)