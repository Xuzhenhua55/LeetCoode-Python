class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root: return None
        if root and root==p or root==q:
            return root
        if root.val > min(p.val,q.val) and root.val < max(p.val,q.val):
            return root
        elif root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left,p,q)
        else:
            return self.lowestCommonAncestor(root.right, p, q)
