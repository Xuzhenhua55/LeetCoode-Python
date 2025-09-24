class Solution(object):

    def largestSumAfterKNegations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums.sort()
        sum = 0
        for i in range(0, len(nums)):
            if nums[i] < 0 and k > 0:
                nums[i] = -nums[i]
                k -= 1
            sum += nums[i]
        nums.sort()
        if k % 2 != 0: sum -= 2 * nums[0]
        return sum
