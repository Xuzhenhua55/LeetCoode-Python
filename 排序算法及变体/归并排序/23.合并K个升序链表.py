# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # 合并两个有序链表
        def mergeTwoList(leftHead, rightHead):
            leftNode, rightNode = leftHead, rightHead
            dummyNode = ListNode(0, None)
            curNode = dummyNode

            while leftNode and rightNode:
                if leftNode.val < rightNode.val:
                    curNode.next = leftNode
                    leftNode = leftNode.next
                else:
                    curNode.next = rightNode
                    rightNode = rightNode.next
                curNode = curNode.next

            curNode.next = leftNode if leftNode else rightNode
            return dummyNode.next

        # 分治递归合并 k 个链表
        def mergeKListsHelper(lists, left, right):
            if left > right: return None
            if left == right: return lists[left]

            # mid 必须偏向 left（下取整），归左半 [left, mid]
            # 如果 mid 归右半 [left, mid-1] 和 [mid, right]
            # 当区间只剩两个元素时，右半永远不缩小，导致死循环
            midIndex = (left + right) // 2
            leftList = mergeKListsHelper(lists, left, midIndex)
            rightList = mergeKListsHelper(lists, midIndex + 1, right)

            return mergeTwoList(leftList, rightList)

        return mergeKListsHelper(lists, 0, len(lists) - 1)


# 核心思路：分治 + 归并
# 1. 将 k 个链表分成两部分，分别合并
# 2. 再将两部分的合并结果合并
# 3. 时间复杂度：O(nklogk)，每个节点参与 logk 次合并
#
# 注意：mid 的划分方式
# - mid = (left + right) // 2，下取整，偏向 left
# - 左半 [left, mid]，右半 [mid+1, right]
# - 这样保证区间不断缩小，不会死循环
#
# 对比其他方法：
# - 顺序合并：O(nk^2)，每次合并一个链表，效率低
# - 堆合并：O(nklogk)，用堆每次选最小节点，但实现更复杂
# - 分治合并：O(nklogk)，递归分治，代码简洁