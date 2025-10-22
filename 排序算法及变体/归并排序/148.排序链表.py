# Definition for singly-linked list.
class ListNode(object):

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):

    def sortList(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        if not head or not head.next: return head
        slow, fast = head, head
        while fast and fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        rightHead = slow.next
        slow.next = None
        left, right = self.sortList(head), self.sortList(rightHead)
        preNode = dummyHead = ListNode(0, None)
        while left and right:
            if left.val < right.val:
                curNode = ListNode(left.val, None)
                preNode.next = curNode
                left = left.next
            else:
                curNode = ListNode(right.val, None)
                preNode.next = curNode
                right = right.next
            preNode = curNode
        curNode.next = left if left else right
        return dummyHead.next
