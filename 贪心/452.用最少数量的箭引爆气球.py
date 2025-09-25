class Solution(object):

    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        points.sort(key=lambda x: x[0])
        resultList = []
        for point in points:
            if not resultList or resultList[-1][1] < point[0]:
                resultList.append(point)
            elif resultList[-1][1] >= point[0]:
                resultList[-1][1] = min(resultList[-1][1], point[1])
        return len(resultList)


Solution().findMinArrowShots([[3, 9], [7, 12], [3, 8], [6, 8], [9, 10], [2, 9],
                              [0, 9], [3, 9], [0, 6], [2, 8]])
