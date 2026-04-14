import heapq

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 最大堆解法：存负数，弹出 k-1 个后堆顶就是第 k 大
        heap = list()
        for num in nums:
            heapq.heappush(heap, -num)
        for i in range(k - 1):
            heapq.heappop(heap)
        return -heap[0]