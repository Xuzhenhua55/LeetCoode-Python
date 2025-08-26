# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """
        resultList=[]
        def DFS(root):
            if root==None:
                return root
            DFS(root.left)
            resultList.append(root.val)
            DFS(root.right)
        DFS(root)
        return resultList

# TODO：迭代法
