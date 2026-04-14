class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        import heapq
        countDict=dict()
        for num in nums:
            countDict[num]=countDict.get(num,0)+1
        topKElementHeap=[]
        for num in countDict:
            heapq.heappush(topKElementHeap, (countDict[num],num))
            if len(topKElementHeap) > k:
                heapq.heappop(topKElementHeap)
        resultList=[]
        while bool(topKElementHeap):
            resultList.append(heapq.heappop(topKElementHeap)[1])
        resultList.reverse()
        return resultList
        # 注意resultList.reverse()返回值为None
        # reversed(list) 返回一个 迭代器，需要用 list() 转换才能得到真正的列表。


# 解法二：使用 Counter + 小根堆（更简洁）
import heapq
from collections import Counter

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        result = list()
        heap = list()
        numToCnt = Counter(nums)

        for num, cnt in numToCnt.items():
            heapq.heappush(heap, (cnt, num))
            if len(heap) > k:
                heapq.heappop(heap)  # 弹出最小的，保留大的

        for i in range(k):
            curMaxNode = heap[0]
            result.append(curMaxNode[1])
            heapq.heappop(heap)

        return result
                
                
        