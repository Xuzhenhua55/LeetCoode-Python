class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        resultMatrix = [[0]*n for _ in range(n)]
        x, y = 0, 0
        num = 1
        maxValue, length = n*n, n
        while num < maxValue:
            for j in range(n-1):
                resultMatrix[x][y+j] = num
                num += 1
            for i in range(n-1):
                resultMatrix[x+i][y+n-1] = num
                num += 1
            for j in range(n-1):
                resultMatrix[x+n-1][y+n-1-j] = num
                num += 1
            for i in range(n-1):
                resultMatrix[x+n-1-i][y] = num
                num += 1
            x += 1
            y += 1
            n -= 2
        if length % 2 != 0:
            resultMatrix[length/2][length/2] = num
        return resultMatrix


s = Solution()
s.generateMatrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
