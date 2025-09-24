class Solution(object):

    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        minGas = float("inf")
        sumGas = 0
        for i in range(0, len(gas)):
            biasGas = gas[i] - cost[i]
            sumGas += biasGas
            if sumGas < minGas: minGas = sumGas
        if sumGas < 0: return -1
        if minGas >= 0: return 0
        for i in range(len(gas) - 1, -1, -1):
            minGas += gas[i] - cost[i]
            if minGas >= 0:
                return i
        return -1
