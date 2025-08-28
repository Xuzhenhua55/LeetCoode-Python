# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        from collections import deque
        deque=deque()
        lastNode=root
        deque.append(lastNode)
        while bool(deque):
            lastNode=deque.popleft()
            if lastNode.right: deque.append(lastNode.right)
            if lastNode.left: deque.append(lastNode.left)
        return lastNode.val