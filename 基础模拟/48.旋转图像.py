# 顺时针90度翻转->矩阵转置+水平方向进行镜像翻转
# 逆时针90度翻转->矩阵转置+垂直方向进行水平翻转
# 180度翻转->水平方向进行镜像翻转+垂直方向进行镜像翻转
class Solution(object):

    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(1, n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        for i in range(n):
            for j in range(n // 2):
                matrix[i][j], matrix[i][n - 1 - j] = matrix[i][n - 1 -
                                                               j], matrix[i][j]
        return matrix
