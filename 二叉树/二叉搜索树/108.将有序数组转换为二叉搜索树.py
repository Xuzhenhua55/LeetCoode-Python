# Definition for a binary tree node.
class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution(object):

    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: Optional[TreeNode]
        """

        def createNode(nums, left, right):
            if left > right: return None
            midIndex = left + (right - left) / 2
            curNode = TreeNode(nums[midIndex])
            curNode.left = createNode(nums, left, midIndex - 1)
            curNode.right = createNode(nums, midIndex + 1, right)
            return curNode

        return createNode(nums, 0, len(nums) - 1)
