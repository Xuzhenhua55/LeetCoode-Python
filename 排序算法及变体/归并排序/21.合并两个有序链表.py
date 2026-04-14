# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # 归并排序的核心操作：合并两个有序链表
        dummyHead = ListNode(0, None)
        curNode = dummyHead

        # 比较两个链表头节点，选择较小的接入结果链表
        while list1 is not None and list2 is not None:
            if list1.val < list2.val:
                curNode.next = ListNode(list1.val, None)
                list1 = list1.next
            else:
                curNode.next = ListNode(list2.val, None)
                list2 = list2.next
            curNode = curNode.next

        # 处理剩余节点
        while list1 is not None:
            curNode.next = ListNode(list1.val, None)
            list1 = list1.next
            curNode = curNode.next
        while list2 is not None:
            curNode.next = ListNode(list2.val, None)
            list2 = list2.next
            curNode = curNode.next

        return dummyHead.next


# 优化版本：直接复用原节点，不创建新节点
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummyHead = ListNode(0, None)
        curNode = dummyHead

        while list1 and list2:
            if list1.val < list2.val:
                curNode.next = list1
                list1 = list1.next
            else:
                curNode.next = list2
                list2 = list2.next
            curNode = curNode.next

        # 直接连接剩余部分，不需要逐个创建
        curNode.next = list1 if list1 else list2

        return dummyHead.next