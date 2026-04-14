class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        upRow, downRow = 0, len(matrix)

        while upRow < downRow:
            midIndex = (upRow + downRow) // 2
            if matrix[midIndex][0] == target:
                return True
            elif matrix[midIndex][0] > target:
                downRow = midIndex
            else:
                # 内层二分：在当前行查找
                leftColumn, rightColumn = 0, len(matrix[0])
                while leftColumn < rightColumn:
                    midColumn = (leftColumn + rightColumn) // 2
                    if matrix[midIndex][midColumn] == target:
                        return True
                    elif matrix[midIndex][midColumn] > target:
                        rightColumn = midColumn
                    else:
                        leftColumn = midColumn + 1

                # 关键：如果 target 大于当前行所有元素，继续向上找
                if leftColumn == len(matrix[0]):
                    upRow = midIndex + 1
                    continue  # continue 很关键！否则会走到下面的 return False

                if leftColumn == rightColumn:
                    return False

        return False