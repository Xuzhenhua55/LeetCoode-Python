# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[List[int]]
        """
        from collections import deque
        deque=deque()
        resultList=[]
        if root!=None: deque.append(root)
        while bool(deque):
            layerNum=len(deque)
            layerList=[]
            for _ in range(layerNum):
                curNode=deque.popleft()
                layerList.append(curNode.val)
                if curNode.left!=None:
                    deque.append(curNode.left)
                if curNode.right!=None:
                    deque.append(curNode.right)
            resultList.append(layerList)
        return resultList
