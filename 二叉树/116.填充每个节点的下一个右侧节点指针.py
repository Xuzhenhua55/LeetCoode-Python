
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        from collections import deque
        deque=deque()
        if root:deque.append(root)
        while bool(deque):
            layerNum=len(deque)
            nextNode=None
            for i in range(layerNum):
                curNode=deque.popleft()
                curNode.next=nextNode
                nextNode=curNode
                if curNode.right:deque.append(curNode.right)
                if curNode.left:deque.append(curNode.left)
        return root