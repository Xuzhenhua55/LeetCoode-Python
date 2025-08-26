# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """
        from collections import deque
        deque=deque()
        resultList=[]
        if root!=None: deque.append(root)
        while bool(deque):
            layerNum=len(deque)
            for i in range(layerNum):
                curNode=deque.popleft()
                if i==0:
                    resultList.append(curNode.val)
                if curNode.right:
                    deque.append(curNode.right)
                if curNode.left:
                    deque.append(curNode.left)
        return resultList
