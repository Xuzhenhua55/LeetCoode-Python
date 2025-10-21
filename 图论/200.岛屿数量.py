class Solution(object):

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        rowBias = [-1, 1, 0, 0]
        columnBias = [0, 0, 1, -1]
        gridTraversed = [[False] * len(grid[0]) for _ in range(len(grid))]

        def DFS(x, y):
            gridTraversed[x][y] = True
            for (xBias, yBias) in zip(rowBias, columnBias):
                newX, newY = x + xBias, y + yBias
                if 0 <= newX < len(grid) and 0 <= newY < len(
                        grid[0]) and grid[newX][newY] == "1" and gridTraversed[
                            newX][newY] == False:
                    DFS(x + xBias, y + yBias)

        result = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1" and gridTraversed[i][j] == False:
                    DFS(i, j)
                    result += 1
        return result
