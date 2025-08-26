# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def largestValues(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """
        from collections import deque
        deque=deque()
        resultList=[]
        if root:deque.append(root)
        while bool(deque):
            layerNum=len(deque)
            layerMaxVal=float('-inf')
            for _ in range(layerNum):
                curNode=deque.popleft()
                if curNode.val>layerMaxVal:
                    layerMaxVal=curNode.val
                if curNode.left: deque.append(curNode.left)
                if curNode.right: deque.append(curNode.right)
            resultList.append(layerMaxVal)
        return resultList 
