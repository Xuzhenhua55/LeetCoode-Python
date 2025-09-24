class Solution(object):

    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        candiesList = [1] * len(ratings)
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candiesList[i] = candiesList[i - 1] + 1
            elif ratings[i] < ratings[i - 1]:
                for j in range(i, -1, -1):
                    if ratings[j] < ratings[
                            j - 1] and candiesList[j] >= candiesList[j - 1]:
                        candiesList[j - 1] += 1
                    else:
                        break
        return sum(candiesList)


class Solution(object):

    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        candiesList = [1] * len(ratings)
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candiesList[i] = candiesList[i - 1] + 1
        for i in range(len(ratings) - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candiesList[i] = max(candiesList[i], candiesList[i + 1] + 1)
        return sum(candiesList)


print(Solution().candy([1, 3, 2, 2, 1]))
