class Solution(object):

    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        intervals.sort(key=lambda x: x[0])
        resultList = []
        for interval in intervals:
            if not resultList or interval[0] > resultList[-1][1]:
                resultList.append(interval[:])
            else:
                resultList[-1][1] = max(resultList[-1][1], interval[1])
        return resultList
