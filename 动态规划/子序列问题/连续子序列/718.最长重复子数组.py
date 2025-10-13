class Solution(object):

    def findLength(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        result = -float('inf')
        dp = [[0] * len(nums2) for _ in range(len(nums1))]
        for i in range(0, len(nums1)):
            for j in range(0, len(nums2)):
                if nums1[i] == nums2[j]:
                    if i - 1 >= 0 and j - 1 >= 0 and nums1[i - 1] == nums2[j -
                                                                           1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = 1
                result = max(result, dp[i][j])
        # print(dp)
        return result


Solution().findLength([1, 2, 3, 2, 1], [3, 2, 1, 4, 7])
