import heapq

class MedianFinder:

    def __init__(self):
        # maxHeap 存较小的一半（堆顶是这部分的最大值）
        # minHeap 存较大的一半（堆顶是这部分的最小值）
        self.maxHeap = list()  # 用负数模拟最大堆
        self.minHeap = list()

    def addNum(self, num: int) -> None:
        # 步骤：
        # 1. 先加入 maxHeap（负数）
        # 2. 从 maxHeap 弹出最大值加入 minHeap（保证 maxHeap 所有值 <= minHeap 所有值）
        # 3. 如果 minHeap 元素多于 maxHeap，把 minHeap 最小值移回 maxHeap
        # 结果：maxHeap 元素数 >= minHeap 元素数，最多多 1 个

        heapq.heappush(self.maxHeap, -num)
        heapq.heappush(self.minHeap, -heapq.heappop(self.maxHeap))

        # 平衡：保持 maxHeap 元素数 >= minHeap 元素数
        if len(self.minHeap) - len(self.maxHeap) > 0:
            heapq.heappush(self.maxHeap, -heapq.heappop(self.minHeap))

    def findMedian(self) -> float:
        # 奇数：maxHeap 多一个，返回 maxHeap 堆顶
        # 偶数：两边相等，返回两个堆顶的平均值
        if len(self.maxHeap) > len(self.minHeap):
            return -self.maxHeap[0]
        else:
            return (-self.maxHeap[0] + self.minHeap[0]) / 2


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()


# ============================================================
# 核心思路：双堆维护中位数
# ============================================================
#
# 维护两个堆：
# - maxHeap（存负数模拟最大堆）：存较小的一半，堆顶是这部分的最大值
# - minHeap：存较大的一半，堆顶是这部分的最小值
#
# 平衡策略：
# - maxHeap 元素数 >= minHeap 元素数，最多多 1 个
# - 奇数时 maxHeap 多一个，中位数是 maxHeap 堆顶
# - 偶数时两边相等，中位数是两个堆顶的平均值
#
# addNum 流程：
# 1. 新元素先加入 maxHeap
# 2. 把 maxHeap 最大值移到 minHeap（保证 maxHeap <= minHeap）
# 3. 如果 minHeap 元素多了，把最小值移回 maxHeap
#
# 时间复杂度：
# - addNum: O(log n) 堆操作
# - findMedian: O(1) 直接取堆顶
#
# 注意：
# - 用 heapq.heappop()，不是 heap.pop()（后者弹出列表末尾）
# - maxHeap 存负数，取出时要取负还原