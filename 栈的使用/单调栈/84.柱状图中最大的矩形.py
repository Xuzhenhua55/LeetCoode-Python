class Solution(object):

    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        heights = [0] + heights + [0]
        stack = []
        maxRectangle = 0
        stack.append(0)
        for i in range(1, len(heights)):
            if heights[i] == heights[stack[-1]]:
                stack.pop()
                stack.append(i)
            elif heights[i] > heights[stack[-1]]:
                stack.append(i)
            else:
                while heights[i] < heights[stack[-1]]:
                    mid = stack.pop()
                    maxRectangle = max(maxRectangle,
                                       (i - stack[-1] - 1) * heights[mid])
                stack.append(i)
        return maxRectangle
