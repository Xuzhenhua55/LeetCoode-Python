
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

# 核心思想是遍历当前层（因为在上一层的遍历中已经连接好）的过程中将下一层进行连接
class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        curNode=root
        while curNode:
            preNode=Node()
            dummyNode=preNode #暂存下一层的起始点
            while curNode:
                if curNode.left:
                    preNode.next=curNode.left
                    preNode=curNode.left
                if curNode.right:
                    preNode.next=curNode.right
                    preNode=curNode.right
                curNode=curNode.next
            curNode=dummyNode.next #下一层的起点
        return root
            
        