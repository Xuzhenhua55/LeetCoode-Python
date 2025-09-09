class Solution(object):

    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        self.graghDict = dict()
        self.traversedPathDict = dict()
        for ticket in tickets:
            if ticket[0] not in self.graghDict:
                self.graghDict[ticket[0]] = [ticket[1]]
                self.traversedPathDict[ticket[0]] = {ticket[1]: False}
            else:
                self.graghDict[ticket[0]].append(ticket[1])
                self.traversedPathDict[ticket[0]][ticket[1]] = False
        self.curList = []
        self.resultList = []

        def DFS(curNode):
            if len(self.curList) == len(tickets) + 1:
                if not self.resultList: self.resultList = list(self.curList)
                isSmaller = False
                for i in range(0, len(self.curList)):
                    if self.curList[i] < self.resultList[i]:
                        isSmaller = True
                        break
                    elif self.curList[i] > self.resultList[i]:
                        break
                if isSmaller: self.resultList = list(self.curList)
                return
            for nextNode in self.graghDict[curNode]:
                if self.traversedPathDict[curNode][nextNode]:
                    continue
                self.curList.append(nextNode)
                self.traversedPathDict[curNode][nextNode] = True
                DFS(nextNode)
                self.curList.pop()
                self.traversedPathDict[curNode][nextNode] = False

        self.curList.append("JFK")
        DFS("JFK")
        return self.resultList


class Solution(object):

    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        from collections import defaultdict
        self.graghDict = defaultdict(list)
        self.traversedPathDict = defaultdict(int)
        for ticket in tickets:
            self.graghDict[ticket[0]].append(ticket[1])
            self.traversedPathDict[(ticket[0], ticket[1])] += 1

        for start in self.graghDict:
            self.graghDict[start].sort()
        self.curList = []
        self.resultList = []

        def DFS(curNode):
            if len(self.curList) == len(tickets) + 1:
                self.resultList = list(self.curList)
                return True
            for nextNode in self.graghDict[curNode]:
                if self.traversedPathDict[(curNode, nextNode)] == 0:
                    continue
                self.curList.append(nextNode)
                self.traversedPathDict[(curNode, nextNode)] -= 1
                if DFS(nextNode): return True
                self.curList.pop()
                self.traversedPathDict[(curNode, nextNode)] += 1

        self.curList.append("JFK")
        DFS("JFK")
        return self.resultList


Solution().findItinerary([["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"],
                          ["LHR", "SFO"]])
