# 原地修改，但是需要利用Python的特性 即[]中元素类型可以不一致
class Solution(object):

    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        MARK = float('inf')

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    for k in range(n):
                        if matrix[i][k] != 0:
                            matrix[i][k] = MARK
                    for k in range(m):
                        if matrix[k][j] != 0:
                            matrix[k][j] = MARK

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == MARK:
                    matrix[i][j] = 0

        return matrix


class Solution(object):

    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        rows, columns = [False] * m, [False] * n
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    rows[i] = True
                    columns[j] = True
        for i in range(m):
            for j in range(n):
                if rows[i] or columns[j]:
                    matrix[i][j] = 0
        return matrix


# 如果某个位置为0，那么这一行一列最终都为0 等价于 第一行和第一列对应的位置为0 因此可以把第一行第一列作为存储媒介
class Solution(object):

    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        zeroInFirstRow, zeroInFirstColumn = False, False
        for i in range(m):
            if matrix[i][0] == 0:
                zeroInFirstColumn = True
                break
        for j in range(n):
            if matrix[0][j] == 0:
                zeroInFirstRow = True
                break

        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0

        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        if zeroInFirstColumn:
            for i in range(m):
                matrix[i][0] = 0
        if zeroInFirstRow:
            for j in range(n):
                matrix[0][j] = 0

        return matrix
