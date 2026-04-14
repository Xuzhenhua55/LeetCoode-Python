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


# 解法二：哈希表去重（O(1) 判断合法性）
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        self.result = list()
        self.matrix = [['.' for _ in range(n)] for _ in range(n)]

        # 三个哈希表：列、↗️主对角线（row+col）、↖️副对角线（row-col）
        self.columnExisted = set()
        self.mainDiagonal = set()   # row + col
        self.antiDiagonal = set()   # row - col

        def backtracing(currRow):
            if currRow == n:
                self.result.append([''.join(row) for row in self.matrix])
                return

            for curColumn in range(n):
                # O(1) 判断是否合法
                if curColumn not in self.columnExisted \
                   and (currRow - curColumn) not in self.antiDiagonal \
                   and (currRow + curColumn) not in self.mainDiagonal:
                    # 前序：做选择
                    self.matrix[currRow][curColumn] = 'Q'
                    self.columnExisted.add(curColumn)
                    self.antiDiagonal.add(currRow - curColumn)
                    self.mainDiagonal.add(currRow + curColumn)

                    backtracing(currRow + 1)

                    # 后序：撤销选择
                    self.mainDiagonal.remove(currRow + curColumn)
                    self.antiDiagonal.remove(currRow - curColumn)
                    self.columnExisted.remove(curColumn)
                    self.matrix[currRow][curColumn] = '.'

        backtracing(0)
        return self.result
