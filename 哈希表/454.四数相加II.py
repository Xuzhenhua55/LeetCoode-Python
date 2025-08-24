class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type nums3: List[int]
        :type nums4: List[int]
        :rtype: int
        """
        countDict=dict()
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                countDict[nums1[i]+nums2[j]] = countDict.get(nums1[i]+nums2[j],0)+1
        result=0
        for i in range(len(nums3)):
            for j in range(len(nums4)):
                if -(nums3[i]+nums4[j]) in countDict:
                    result += countDict[-(nums3[i]+nums4[j])]
        return result