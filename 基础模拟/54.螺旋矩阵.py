class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        m, n = len(matrix), len(matrix[0])
        x, y = 0, 0
        resultList = []
        while m > 1 and n > 1:
            for j in range(n-1):
                resultList.append(matrix[x][y+j])
            for i in range(m-1):
                resultList.append(matrix[x+i][y+n-1])
            for j in range(n-1):
                resultList.append(matrix[x+m-1][y+n-1-j])
            for i in range(m-1):
                resultList.append(matrix[x+m-1-i][y])
            x += 1
            y += 1
            n -= 2
            m -= 2
        if m == 1:
            for j in range(n):
                resultList.append(matrix[x][y+j])
        elif n == 1:  # 这里比较重要，不能直接else，否则在m和n不均衡且n为偶数的的情况下 会得到m-n行已经加过一次的数据
            for i in range(m):
                resultList.append(matrix[x+i][y])
        return resultList


s = Solution()
s.spiralOrder([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
