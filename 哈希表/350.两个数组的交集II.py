class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        numsDict1,numsDict2=dict(),dict()
        for num in nums1:
            numsDict1[num]=numsDict1.get(num,0)+1
        for num in nums2:
            numsDict2[num]=numsDict2.get(num,0)+1
        resultList=[]
        for key in numsDict1:
            if key in numsDict2:
                resultList.extend([key]*min(numsDict1[key],numsDict2[key]))
        return resultList
