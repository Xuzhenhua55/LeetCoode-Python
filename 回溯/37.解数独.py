class Solution(object):

    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """

        def calChunkIndex(row, column):
            return (row // 3) * 3 + (column // 3)

        from collections import defaultdict
        self.rowUnexistedDict = defaultdict(lambda: set(range(1, 10)))
        self.columnUnExistedDict = defaultdict(lambda: set(range(1, 10)))
        self.chunkUnexistedDict = defaultdict(lambda: set(range(1, 10)))
        for row in range(9):
            for column in range(9):
                if board[row][column] == '.': continue
                val = int(board[row][column])
                self.rowUnexistedDict[row].remove(val)
                self.columnUnExistedDict[column].remove(val)
                self.chunkUnexistedDict[calChunkIndex(row, column)].remove(val)

        def isValid(row, column, val):
            return val in self.rowUnexistedDict[
                row] and val in self.columnUnExistedDict[
                    column] and val in self.chunkUnexistedDict[calChunkIndex(
                        row, column)]

        def DFS(row, column):
            if row == 9 and column == 0:
                return True
            nextRow, nextColumn = row, column
            if column == 8: nextRow, nextColumn = row + 1, 0
            else:
                nextRow, nextColumn = row, column + 1
            if board[row][column] != '.':
                return DFS(nextRow, nextColumn)
            for val in range(1, 10):
                if not isValid(row, column, val): continue
                self.rowUnexistedDict[row].remove(val)
                self.columnUnExistedDict[column].remove(val)
                self.chunkUnexistedDict[calChunkIndex(row, column)].remove(val)
                board[row][column] = str(val)
                if DFS(nextRow, nextColumn): return True
                board[row][column] = '.'
                self.rowUnexistedDict[row].add(val)
                self.columnUnExistedDict[column].add(val)
                self.chunkUnexistedDict[calChunkIndex(row, column)].add(val)

        DFS(0, 0)
        return


class Solution(object):

    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """

        def calChunkIndex(row, column):
            return (row // 3) * 3 + (column // 3)

        from collections import defaultdict
        self.rowUnexistedDict = defaultdict(lambda: set(range(1, 10)))
        self.columnUnExistedDict = defaultdict(lambda: set(range(1, 10)))
        self.chunkUnexistedDict = defaultdict(lambda: set(range(1, 10)))
        self.emptyCells = []  # 记录所有待填位置

        for row in range(9):
            for column in range(9):
                if board[row][column] == '.':
                    self.emptyCells.append((row, column))
                    continue
                val = int(board[row][column])
                self.rowUnexistedDict[row].remove(val)
                self.columnUnExistedDict[column].remove(val)
                self.chunkUnexistedDict[calChunkIndex(row, column)].remove(val)

        def getCandidates(row, column):
            return self.rowUnexistedDict[row] & \
                   self.columnUnExistedDict[column] & \
                   self.chunkUnexistedDict[calChunkIndex(row, column)]

        def DFS():
            if not self.emptyCells:  # 没有空格了
                return True

            # 选候选数最少的位置
            minIndex, minCandidates = None, None
            for i, (r, c) in enumerate(self.emptyCells):
                cand = getCandidates(r, c)
                if minCandidates is None or len(cand) < len(minCandidates):
                    minIndex, minCandidates = i, cand
                if len(minCandidates) == 1:  # 提前剪枝
                    break

            if not minCandidates:  # 无解
                return False

            row, column = self.emptyCells[minIndex]
            self.emptyCells.pop(minIndex)  # 取出该位置

            for val in list(minCandidates):
                board[row][column] = str(val)
                self.rowUnexistedDict[row].remove(val)
                self.columnUnExistedDict[column].remove(val)
                self.chunkUnexistedDict[calChunkIndex(row, column)].remove(val)

                if DFS():
                    return True

                # 回溯
                board[row][column] = '.'
                self.rowUnexistedDict[row].add(val)
                self.columnUnExistedDict[column].add(val)
                self.chunkUnexistedDict[calChunkIndex(row, column)].add(val)

            # 回溯时把位置加回去
            self.emptyCells.insert(minIndex, (row, column))
            return False

        DFS()
        return
