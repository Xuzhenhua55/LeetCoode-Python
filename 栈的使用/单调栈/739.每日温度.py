class Solution(object):

    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        from collections import deque
        stack = deque()
        resultList = [0] * len(temperatures)
        for i in range(len(temperatures)):
            if i == 0 or temperatures[i] < stack[-1][0]:
                stack.append([temperatures[i], i])
            else:
                while stack and temperatures[i] > stack[-1][0]:
                    topElement = stack.pop()
                    resultList[topElement[1]] = i - topElement[1]
                stack.append([temperatures[i], i])
        return resultList
