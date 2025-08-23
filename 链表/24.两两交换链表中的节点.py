# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        if head==None or head.next==None:
            return head
        dummyHead=ListNode(0,head)
        pre,cur,after=dummyHead,dummyHead.next,dummyHead.next.next
        while after!=None:
            pre.next=after
            cur.next=after.next
            after.next=cur
            pre=cur
            if pre.next!=None:
                cur=pre.next
                if cur.next!=None:
                    after=cur.next
                else:
                    break
            else:
                break
        return dummyHead.next
