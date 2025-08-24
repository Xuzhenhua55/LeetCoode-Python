class Solution(object):
    def totalFruit(self, fruits):
        """
        :type fruits: List[int]
        :rtype: int
        """
        fruitMap=dict()
        left,right=0,0
        result=float('-inf')
        while right<len(fruits):
            if fruits[right] in fruitMap:
                fruitMap[fruits[right]]+=1

            else:
                fruitMap[fruits[right]]=1
            while left<right and len(fruitMap)>2:
                fruitMap[fruits[left]]-=1
                if fruitMap[fruits[left]]==0:
                    del fruitMap[fruits[left]]
                left+=1
            if right-left+1>result:
                result=right-left+1
            right+=1
        return result
                        