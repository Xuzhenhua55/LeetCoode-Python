# 最后一个平方小于等于x的数据
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        left,right=0,x+1
        while left<right:
            mid=left+(right-left)/2
            if mid*mid<x or mid*mid ==x:
                left=mid+1
            else:
                right=mid
        return left-1