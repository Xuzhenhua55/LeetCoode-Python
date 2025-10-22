"""
# Definition for a Node.
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random
"""


class Node:

    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution(object):

    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        oldToNew = dict()
        oldNode = head
        dummyNode = preNode = Node(0, None)
        while oldNode:
            newNode = Node(oldNode.val, None)
            preNode.next = newNode
            preNode = newNode
            oldToNew[oldNode] = newNode
            oldNode = oldNode.next

        oldNode = head
        newNode = dummyNode.next
        while oldNode:
            if oldNode.random:
                newNode.random = oldToNew[oldNode.random]
            oldNode = oldNode.next
            newNode = newNode.next
        return dummyNode.next
