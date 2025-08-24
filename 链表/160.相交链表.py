# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
# 掌握了No.19后这道题就会有一定的思路
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if headA==None or headB==None or headA==headB:
            return headA
        node1,node2=headA,headB
        while node1.next!=None and node2.next!=None:
            node1=node1.next
            node2=node2.next
        if node1.next==None:
            longHead,shortHead,slowNode=headB,headA,node2
        else:
            longHead,shortHead,slowNode=headA,headB,node1
        while slowNode.next != None:
            longHead=longHead.next
            slowNode=slowNode.next
        while shortHead!=None and longHead!=None:
            if shortHead==longHead:
                return shortHead
            shortHead=shortHead.next
            longHead=longHead.next
        return None
        