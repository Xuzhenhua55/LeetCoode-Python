class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        prefixSum = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefixSum[i + 1] = prefixSum[i] + nums[i]
        self.prefixSum = prefixSum

    def sumRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: int
        """
        return self.prefixSum[right + 1] - self.prefixSum[left]


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)
