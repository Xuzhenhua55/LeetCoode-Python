class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        self.curPath = list()
        self.visited = [[False for _ in range(n)] for _ in range(m)]
        self.resutl = False

        def dfs(x, y):
            if self.resutl: return
            if self.curPath[-1] != word[len(self.curPath) - 1]: return
            if len(self.curPath) == len(word):
                self.resutl = True
                return

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for xBias, yBias in directions:
                newX, newY = x + xBias, y + yBias
                if newX < 0 or newX >= m or newY < 0 or newY >= n or self.visited[newX][newY]:
                    continue
                self.curPath.append(board[newX][newY])
                self.visited[newX][newY] = True
                dfs(newX, newY)
                self.curPath.pop()
                self.visited[newX][newY] = False

        for i in range(m):
            for j in range(n):
                if not self.resutl:
                    self.curPath.append(board[i][j])
                    self.visited[i][j] = True
                    dfs(i, j)
                    self.curPath.pop()
                    self.visited[i][j] = False

        return self.resutl