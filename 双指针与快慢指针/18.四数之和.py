class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums=sorted(nums)
        resultList=[]
        for i in range(len(nums)):
            # 这里target是传入的，因此可能是更大的负数，所以不一定num[i]>target就意味着不能搜到了，只有>0的情况下且>target才是真的搜索不到了
            if nums[i]>target and nums[i]>0:
                return resultList
            if i>0 and nums[i]==nums[i-1]:
                continue
            for j in range(i+1,len(nums)):
                if j>i+1 and nums[j]==nums[j-1]:
                    continue
                left,right=j+1,len(nums)-1
                while left<right:
                    if nums[i]+nums[j]+nums[left]+nums[right]>target:
                        right-=1
                    elif nums[i]+nums[j]+nums[left]+nums[right]<target:
                        left+=1
                    else:
                        resultList.append([nums[i],nums[j],nums[left],nums[right]])
                        while left<right and nums[left]==nums[left+1]: left+=1
                        while right>left and nums[right]==nums[right-1]: right-=1
                        left+=1
                        right-=1
        return resultList

s=Solution()
s.fourSum([1, -2,-5,-4,-3,3,3,5],-11)