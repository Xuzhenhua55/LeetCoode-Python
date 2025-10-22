# Definition for singly-linked list.
class ListNode(object):

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):

    def reverseKGroup(self, head, k):
        """
        :type head: Optional[ListNode]
        :type k: int
        :rtype: Optional[ListNode]
        """
        # 统计需要翻转的总次数，如果小于1次意味着不需要翻转直接返回
        totalCount = 0
        curNode = head
        while curNode:
            totalCount += 1
            curNode = curNode.next
        reverseCount, totalReverseCount = 0, totalCount // k
        if totalReverseCount == 0: return head
        newHead = None
        curLength = 0
        preEnd = ListNode(0, head)

        dummyNode = ListNode(0, head)
        preNode, curNode = dummyNode, dummyNode.next
        while curNode:
            nextNode = curNode.next
            curNode.next = preNode
            preNode = curNode
            curNode = nextNode
            curLength += 1
            if curLength == k:
                curLength = 0
                reverseCount += 1
                if reverseCount == 1:  # 在第一次翻转结束时将最后位置的Node作为新的头结点
                    newHead = preNode
                # 将上一个翻转完的子链表的结尾Node连上这个刚翻转完的子链表的开头
                preEnd.next = preNode
                # 将刚翻转的子链表的结尾作为新的子链表结尾点
                preEnd = dummyNode.next
                # 重置dummyNode，preNode和curNode进行下一轮循环
                dummyNode.next.next = curNode
                dummyNode = ListNode(0, curNode)
                preNode, curNode = dummyNode, dummyNode.next
            if reverseCount == totalReverseCount:
                break
        return newHead


class Solution(object):

    def reverseKGroup(self, head, k):
        """
        :type head: Optional[ListNode]
        :type k: int
        :rtype: Optional[ListNode]
        """
        totalCount = 0
        curNode = head
        while curNode:
            totalCount += 1
            curNode = curNode.next
        preGroupEnd = dummyHead = ListNode(0, head)
        preNode, curNode = None, head
        while totalCount >= k:
            totalCount -= k
            for _ in range(k):
                nextNode = curNode.next
                curNode.next = preNode
                preNode = curNode
                curNode = nextNode
            curGroupEnd = preGroupEnd.next
            curGroupEnd.next = curNode
            preGroupEnd.next = preNode
            preGroupEnd = curGroupEnd
        return dummyHead.next
