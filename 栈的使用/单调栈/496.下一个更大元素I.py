class Solution(object):

    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        indexDict = dict()
        result = [-1] * len(nums1)
        for i in range(len(nums1)):
            indexDict[nums1[i]] = i
        stack = []
        for i in range(len(nums2)):
            if not stack or nums2[i] < stack[-1]: stack.append(nums2[i])
            else:
                while stack and nums2[i] > stack[-1]:
                    num = stack.pop()
                    if num in indexDict: result[indexDict[num]] = nums2[i]
                stack.append(nums2[i])
        return result
