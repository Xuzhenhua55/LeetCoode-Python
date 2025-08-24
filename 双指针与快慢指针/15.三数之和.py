class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums=sorted(nums)
        resultList=[]
        for i in range(len(nums)):
            if nums[i]>0:
                return resultList
            if i>0 and nums[i]==nums[i-1]:
                continue
            left,right=i+1,len(nums)-1
            while left<right:
                if nums[i]+nums[left]+nums[right]<0:
                    left+=1
                elif nums[i]+nums[left]+nums[right]>0:
                    right-=1
                else:
                    resultList.append([nums[i],nums[left],nums[right]])
                    # 虽然此时找到了一个解，但是当left和right继续收缩的过程中仍然有可能找到新的解
                    while left<right and nums[left+1]==nums[left]:
                        left+=1
                    while right>left and nums[right-1]==nums[right]:
                        right-=1
                    left+=1
                    right-=1
        return resultList
                    
s=Solution()
s.threeSum([-1, 0,1,2,-1,-4])