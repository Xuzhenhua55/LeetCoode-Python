class Solution(object):

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        from collections import deque
        # 构建邻接表和入度列表
        nodeTable = [[] for _ in range(numCourses)]
        inDegreeList = [0] * numCourses
        for prerequisite in prerequisites:
            cur, pre = prerequisite
            nodeTable[pre].append(cur)
            inDegreeList[cur] += 1
        # 初始化队列元素
        queue = deque()
        for i in range(len(inDegreeList)):
            if inDegreeList[i] == 0:
                queue.append(i)
        # 元素出队并更新相关的入度表 伴随新元素入队
        count = 0
        while queue:
            nodeIndex = queue.popleft()
            count += 1
            for i in range(len(nodeTable[nodeIndex])):
                inDegreeList[nodeTable[nodeIndex][i]] -= 1
                if inDegreeList[nodeTable[nodeIndex][i]] == 0:
                    queue.append(nodeTable[nodeIndex][i])
        return count == numCourses
