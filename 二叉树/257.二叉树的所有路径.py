# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[str]
        """
        self.resultList=[]
        def DFS(root,curPath):
            curPath.append(str(root.val))
            if not root.left and not root.right:
                self.resultList.append('->'.join(curPath))
                return
            if root.left:
                DFS(root.left,curPath)
                curPath.pop()
            if root.right:
                DFS(root.right, curPath)
                curPath.pop()
        curPath=[]
        if root: DFS(root,curPath)
        return self.resultList
            
            