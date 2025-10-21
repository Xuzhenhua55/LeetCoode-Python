class Solution(object):

    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        from collections import deque
        m, n = len(grid), len(grid[0])
        freshCount = 0
        queue = deque()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    freshCount += 1
                elif grid[i][j] == 2:
                    queue.append([i, j])
        if freshCount == 0: return 0  # all rotted when initialized
        rowBias = [1, -1, 0, 0]
        columnBias = [0, 0, 1, -1]
        result = -1
        while queue:
            result += 1
            layerNum = len(queue)
            for i in range(layerNum):
                x, y = queue.popleft()
                for xBias, yBias in zip(rowBias, columnBias):
                    newX, newY = x + xBias, y + yBias
                    if 0 <= newX < m and 0 <= newY < n and grid[newX][
                            newY] == 1:
                        freshCount -= 1
                        grid[newX][newY] = 2  # avoid repeated enter
                        queue.append([newX, newY])
        if freshCount == 0: return result
        return -1
