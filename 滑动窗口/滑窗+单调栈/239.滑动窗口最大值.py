# 本质上维护窗口内有可能成为最大值潜力的元素，使得最左侧的值被删除后，可以快速查找得到下一个最大值
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        from collections import deque
        deque=deque()
        left,right=0,0
        resultList=[]
        while right<len(nums):
            if right-left+1<=k:
                while bool(deque) and nums[right]>deque[-1]: deque.pop()
                deque.append(nums[right])
            else:
                if nums[left]==deque[0]:
                    deque.popleft()
                while bool(deque) and nums[right] > deque[-1]:
                    deque.pop()
                deque.append(nums[right])
                left += 1
            if right-left+1==k:
                resultList.append(deque[0])
            right+=1
        return resultList


class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        from collections import deque
        deque = deque()
        left, right = 0, 0
        resultList = []
        while right < len(nums):
            # 扩容
            while bool(deque) and nums[right] > deque[-1]:
                deque.pop()
            deque.append(nums[right])
            # 收缩
            if bool(deque) and nums[left] == deque[0] and right>=k:
                deque.popleft()
            if right>=k:
                left+=1
            # 判断
            if right-left+1 == k:
                resultList.append(deque[0])
            right += 1
        return resultList
                    
        