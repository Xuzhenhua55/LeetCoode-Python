# 本质上是贪心，先排高的然后再继续排后面的即可
class Solution(object):

    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        queue = []
        people.sort(key=lambda x: (-x[0], x[1]))
        for x in people:
            queue.insert(x[1], x)
        return list(queue)
