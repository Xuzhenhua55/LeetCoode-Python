# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: Optional[ListNode]
        :type val: int
        :rtype: Optional[ListNode]
        """
        if head == None:
            return None
        dummyHead=ListNode(0,head)
        slow,fast=dummyHead,dummyHead.next
        while fast!=None:
            if fast.val==val:
                slow.next=fast.next
                fast=slow.next
            else:
                slow=fast
                fast=slow.next

        return dummyHead.next
                