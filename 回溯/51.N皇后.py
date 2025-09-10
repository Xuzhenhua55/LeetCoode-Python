class Solution(object):

    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        self.matrix = [['.' for _ in range(n)] for _ in range(n)]
        self.resultList = []

        def isValid(row, column):
            for i in range(1, n):
                if row - i >= 0 and self.matrix[row - i][column] == 'Q':
                    return False
                if row + i < n and self.matrix[row + i][column] == 'Q':
                    return False
                if row - i >= 0 and column - i >= 0 and self.matrix[row - i][
                        column - i] == 'Q':
                    return False
                if row - i >= 0 and column + i < n and self.matrix[row - i][
                        column + i] == 'Q':
                    return False
                if row + i < n and column - i >= 0 and self.matrix[row + i][
                        column - i] == 'Q':
                    return False
                if row + i < n and column + i < n and self.matrix[row +
                                                                  i][column +
                                                                     i] == 'Q':
                    return False
            return True

        def DFS(row):
            if row == n:

                self.resultList.append(
                    [''.join(layer) for layer in self.matrix])
                return
            for i in range(0, n):
                if not isValid(row, i): continue
                self.matrix[row][i] = 'Q'
                DFS(row + 1)
                self.matrix[row][i] = '.'

        DFS(0)
        return self.resultList
