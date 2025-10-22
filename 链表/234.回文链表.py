# Definition for singly-linked list.
class ListNode(object):

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):

    def isPalindrome(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: bool
        """
        valueList = []
        curNode = head
        while curNode:
            valueList.append(curNode.val)
            curNode = curNode.next
        left, right = 0, len(valueList) - 1
        while left < right:
            if valueList[left] != valueList[right]:
                return False
            left += 1
            right -= 1
        return True


class Solution(object):

    def reverseList(self, dummyHead):
        pre, cur = dummyHead, dummyHead.next
        while cur:
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
        dummyHead.next = pre

    def isPalindrome(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: bool
        """
        if not head or not head.next: return True
        slow, fast = head, head
        while fast and fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        self.reverseList(slow)
        left, right = head, slow.next
        while right != slow:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        return True


head = ListNode(1, ListNode(2, ListNode(2, ListNode(1, None))))
Solution().isPalindrome(head)
