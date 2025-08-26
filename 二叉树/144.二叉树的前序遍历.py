# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def DFS(self,root,resultList):
        if root == None:
            return root
        resultList.append(root.val)
        self.DFS(root.left,resultList)
        self.DFS(root.right,resultList)
    
    def preorderTraversal(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """
        resultList=[]
        self.DFS(root,resultList)
        return resultList
        
# TODO：迭代法
