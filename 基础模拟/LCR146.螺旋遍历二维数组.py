class Solution(object):
    def spiralArray(self, array):
        """
        :type array: List[List[int]]
        :rtype: List[int]
        """
        if len(array)==0:
            return []
        
        m,n=len(array),len(array[0])
        x,y=0,0
        resultList=[]
        while m>1 and n>1:
            for j in range(n-1):
                resultList.append(array[x][y+j])
            for i in range(m-1):
                resultList.append(array[x+i][y+n-1])
            for j in range(n-1):
                resultList.append(array[x+m-1][y+n-1-j])
            for i in range(m-1):
                resultList.append(array[x+m-1-i][y])
            x+=1
            y+=1
            m-=2
            n-=2
        if m==1:
            for j in range(n):
                resultList.append(array[x][y+j])
        elif n==1:
            for i in range(m):
                resultList.append(array[x+i][y])
        return resultList