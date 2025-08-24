# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None or head.next==None:
            return None
        slow,fast=head,head
        # 如果是在同一个起点，那么需要移动之后再判断是否相等
        while fast!=None and fast.next!=None:
            fast = fast.next.next
            slow=slow.next
            if fast==slow:
                slow=head
                while slow!=fast:
                    slow=slow.next
                    fast=fast.next
                return slow

        return None

