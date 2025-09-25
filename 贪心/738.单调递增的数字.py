class Solution(object):

    def monotoneIncreasingDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        digitList = [int(digit) for digit in str(n)]
        resultList = [9] * len(digitList)
        preNumber = -1
        reverseIndex = -1
        for i in range(len(digitList)):
            if digitList[i] >= preNumber:
                resultList[i] = digitList[i]
                preNumber = digitList[i]
            else:
                reverseIndex = i - 1
                break
        if reverseIndex != -1:
            for i in range(reverseIndex, -1, -1):
                if i == 0: resultList[i] = digitList[i] - 1
                else:
                    if digitList[i] - 1 >= digitList[i - 1]:
                        resultList[i] = digitList[i] - 1
                        break
                    else:
                        resultList[i] = 9

        if resultList[0] == 0 and len(resultList) > 1:
            resultList = resultList[1:]
        return int(''.join(map(str, resultList)))


# 根据上述代码可以发现本质上本题需要从后往前进行迭代更新
class Solution(object):

    def monotoneIncreasingDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        digitList = [int(digit) for digit in str(n)]
        lastUpdate = len(digitList)
        for i in range(len(digitList) - 1, 0, -1):
            if digitList[i] < digitList[i - 1]:
                if digitList[i - 1] == 0: digitList[i - 1] = 9
                else: digitList[i - 1] -= 1
                lastUpdate = i
        for i in range(lastUpdate, len(digitList)):
            digitList[i] = 9
        return int(''.join(map(str, digitList)))


Solution().monotoneIncreasingDigits(101)
