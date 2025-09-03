# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        self.result=None
        def DFS(root,p,q):
            if self.result: return True,True
            if not root: return False,False
            pExistLeft,qExistLeft=DFS(root.left,p,q)
            pExistRight,qExistRight = DFS(root.right, p, q)
            # 需要注意的点 主要是可能root是p然后其中一个子树包含q 这种情况 而非pq在两侧
            pExist = pExistLeft or pExistRight or root == p
            qExist = qExistLeft or qExistRight or root == q

            if pExist and qExist and self.result is None:
                self.result=root
                return True, True
            return pExist, qExist
        DFS(root,p,q)
        return self.result
            

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root: return None
        if root==p or root==q:
            return root
        leftResult=self.lowestCommonAncestor(root.left,p,q)
        rightResult=self.lowestCommonAncestor(root.right,p,q)
        if leftResult and rightResult:
            return root
        return leftResult or rightResult
        
