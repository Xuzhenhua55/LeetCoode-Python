# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def isEqual(self,p,q):
        if (p and not q) or (not p and q): return False
        if not p and not q: return True
        return p.val==q.val and self.isEqual(p.left,q.left) and self.isEqual(p.right,q.right)
    def isSubtree(self, root, subRoot):
        """
        :type root: Optional[TreeNode]
        :type subRoot: Optional[TreeNode]
        :rtype: bool
        """
        if subRoot==None:return True
        if root==None: return False
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot) or self.isEqual(root, subRoot)
