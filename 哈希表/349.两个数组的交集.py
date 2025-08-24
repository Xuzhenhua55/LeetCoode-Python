class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        numsSet1,numsSet2=set(),set()
        for num in nums1:
            numsSet1.add(num)
        for num in nums2:
            numsSet2.add(num)
        return list(numsSet1&numsSet2)