class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        sum = 0
        for i in range(0, len(prices)):
            if i > 0 and prices[i] > prices[i - 1]:
                sum += prices[i] - prices[i - 1]
        return sum
