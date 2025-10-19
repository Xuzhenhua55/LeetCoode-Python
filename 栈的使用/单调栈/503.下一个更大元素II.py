class Solution(object):

    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        stack = []
        i, length = 0, len(nums)
        result = [-1] * len(nums)
        while (i < 2 * length - 1):
            if not stack or nums[i % length] < stack[-1][0]:
                stack.append([nums[i % length], i % length])
            else:
                while stack and nums[i % length] > stack[-1][0]:
                    topElement = stack.pop()
                    result[topElement[1]] = nums[i % length]
                stack.append([nums[i % length], i % length])
            i += 1
        return result
