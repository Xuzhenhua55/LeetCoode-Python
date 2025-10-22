# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: Optional[ListNode]
        :type n: int
        :rtype: Optional[ListNode]
        """
        dummyHead=ListNode(0,head)
        slow,fast=dummyHead,dummyHead
        for i in range(n-1):
            fast=fast.next
        pre=dummyHead
        while fast.next!=None:
            pre=slow
            slow=slow.next
            fast=fast.next
        pre.next=slow.next
        return dummyHead.next
            