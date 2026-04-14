# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
  def flatten(self, root: Optional[TreeNode]) -> None:
    if not root: return None
    leftNode, rightNode = root.left, root.right
    if leftNode:
        root.right = leftNode
        root.left = None
        curNode = root.right
        while curNode and curNode.right:
            curNode = curNode.right
        curNode.right = rightNode
    self.flatten(root.right)
    
