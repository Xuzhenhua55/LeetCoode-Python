# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: Optional[TreeNode]
        """
        def buildTree(nums,left,right):
            if left>right:
                return None
            maxVal,maxIndex=float("-inf"),-1
            for i in range(left,right+1):
                if nums[i]>maxVal:
                    maxVal=nums[i]
                    maxIndex=i
            root=TreeNode(maxVal)
            root.left=buildTree(nums,left,maxIndex-1)
            root.right = buildTree(nums,maxIndex+1, right)
            return root
        return buildTree(nums,0,len(nums)-1)
