class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        result = -float('inf')
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                if prices[j] - prices[i] > result:
                    result = prices[j] - prices[i]
        return max(result, 0)


class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        leftMin = float('inf')
        result = 0
        for i in range(len(prices)):
            leftMin = min(leftMin, prices[i])
            result = max(result, prices[i] - leftMin)
        return result
