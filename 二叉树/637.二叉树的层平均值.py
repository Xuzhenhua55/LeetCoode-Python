# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[float]
        """
        from collections import deque
        deque=deque()
        resultList=[]
        if root: deque.append(root)
        while bool(deque):
            layerNum=len(deque)
            layerSum=0.0
            for i in range(layerNum):
                curNode=deque.popleft()
                layerSum += curNode.val
                if curNode.left:deque.append(curNode.left)
                if curNode.right:deque.append(curNode.right)
            resultList.append(layerSum/layerNum)
        return resultList