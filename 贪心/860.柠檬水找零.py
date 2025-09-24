class Solution(object):

    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
        countDict = {5: 0, 10: 0}
        for bill in bills:
            if bill == 5:
                countDict[5] += 1
            elif bill == 10:
                if countDict[5] == 0:
                    return False
                countDict[5] -= 1
                countDict[10] += 1
            else:
                if countDict[10] != 0 and countDict[5] != 0:
                    countDict[10] -= 1
                    countDict[5] -= 1
                elif countDict[10] == 0 and countDict[5] >= 3:
                    countDict[5] -= 3
                else:
                    return False
        return True
