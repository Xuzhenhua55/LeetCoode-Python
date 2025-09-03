# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: Optional[TreeNode]
        :type key: int
        :rtype: Optional[TreeNode]
        """
        if not root: return None
        if root.val==key:
            if not root.right: return root.left
            if not root.left: return root.right
            preNode=root
            nextNode = root.right
            while nextNode.left:
                preNode=nextNode
                nextNode = nextNode.left
            root.val = nextNode.val
            # 这两个地方最为关键
            if preNode==root:
                # 如果下一个要删的是root的right那么应该将下一个right衔接而上
                root.right=preNode.right.right
            else:
                # 有可能这个被删的节点还有右子树 这个很容易忽略
                preNode.left = nextNode.right
            return root
        if root.val>key:
            root.left=self.deleteNode(root.left,key)
        elif root.val<key:
            root.right=self.deleteNode(root.right,key)
        return root
