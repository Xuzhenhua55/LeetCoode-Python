# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def isValidBST(self, root):
        def dfs(node):
            if not node:
                return True, float("inf"), float("-inf")

            leftValid, leftMin, leftMax = dfs(node.left)
            rightValid, rightMin, rightMax = dfs(node.right)

            if not leftValid or not rightValid:
                return False, 0, 0
            if not (leftMax < node.val < rightMin):
                return False, 0, 0

            return True, min(leftMin, node.val), max(rightMax, node.val)

        ok, _, _ = dfs(root)
        return ok


class Solution(object):
    def isValidBST(self, root):
        self.preNode=None
        def DFS(root):
            if not root: return True
            if not DFS(root.left):
                return False
            if self.preNode and root.val <= self.preNode.val:
                return False
            self.preNode = root
            return DFS(root.right)
        return DFS(root)

class Solution(object):
    def isValidBST(self, root):
        self.preNode = None
        self.isValid = True
        def DFS(root):
            if not root or not self.isValid:
                return
            DFS(root.left)
            if self.preNode and root.val <= self.preNode.val:
                self.isValid=False
                return
            self.preNode = root
            DFS(root.right)
        DFS(root)
        return self.isValid
