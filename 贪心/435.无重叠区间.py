class Solution(object):

    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        intervals.sort(key=lambda x: (x[0], [x[1]]))
        resultList = []
        resultCount = 0
        for interval in intervals:
            if not resultList or interval[0] >= resultList[-1][1]:
                resultList.append(interval)
            else:
                if interval[1] < resultList[-1][1]:
                    resultList.pop()
                    resultList.append(interval)
                resultCount += 1
        return resultCount


Solution().eraseOverlapIntervals([[1, 2], [2, 3], [3, 4], [1, 3]])
