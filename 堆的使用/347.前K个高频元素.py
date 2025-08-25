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
                
                
        