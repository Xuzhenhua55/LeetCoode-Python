# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: Optional[TreeNode]
        """
        def buildTree(inorder,postorder,inLeft,inRight,postLeft,postRight):
            if inLeft>inRight or postLeft>postRight:
                return None
            rootVal = postorder[postRight]
            midIndex=-1
            for i in range(inLeft,inRight+1):
                if rootVal==inorder[i]:
                    midIndex=i
            root=TreeNode(rootVal)
            leftBias=midIndex-1-inLeft
            root.left=buildTree(inorder,postorder,inLeft,midIndex-1,postLeft,postLeft+leftBias)
            root.right=buildTree(inorder,postorder,midIndex+1,inRight,postLeft+leftBias+1,postRight-1)
            return root
        return buildTree(inorder,postorder,0,len(inorder)-1,0,len(postorder)-1)