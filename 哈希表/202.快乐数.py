class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        usedSet=set()
        while True:
            sum=0
            while n>0:
                sum+=(n%10)*(n%10)
                n=n/10
            if sum == 1:
                return True
            elif sum in usedSet:
                return False
            else:
                usedSet.add(sum)
                n=sum

s=Solution()
s.isHappy(19)