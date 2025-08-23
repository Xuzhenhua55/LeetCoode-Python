# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        if head==None or head.next==None:
            return head
        dummyHead=ListNode(0,head)
        pre,cur=dummyHead,dummyHead.next
        while cur !=None:
            curNext=cur.next
            cur.next=pre
            pre=cur
            cur=curNext
        dummyHead.next.next=None # 需要要这一步的环打破，如果是dummyHead=None只是重置了指针，实际上内存区域上还是环装的
        return pre

s=Solution()
head=ListNode(1,next=ListNode(2,None))
s.reverseList(head)
