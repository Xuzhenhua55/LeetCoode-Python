class Solution(object):

    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        stack = []
        sum = 0
        stack.append([height[0], 0])
        for i in range(1, len(height)):
            if height[i] == stack[-1][0]:
                stack.pop()
                stack.append([height[i], i])
            elif height[i] < stack[-1][0]:
                stack.append([height[i], i])
            else:
                while stack and height[i] > stack[-1][0]:
                    midElement = stack.pop()
                    if stack:
                        width = i - stack[-1][1] - 1
                        h = min(height[i], stack[-1][0]) - midElement[0]
                        sum += width * h
                stack.append([height[i], i])
        return sum
