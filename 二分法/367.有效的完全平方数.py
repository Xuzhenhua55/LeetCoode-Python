# 找到第一个 平方大于num的数
class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        left,right=0,num+1
        while left<right:
            mid=left+(right-left)/2
            if mid*mid <= num:
                left=mid+1
            else:
                right=mid
        if (left-1)*(left-1)==num:
            return True
        else:
            return False
        