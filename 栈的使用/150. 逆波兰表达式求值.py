class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        from collections import deque
        calculator=["+","-","*","/"]
        stack=deque()
        for token in tokens:
            if token not in calculator:
                stack.append(token)
            else:
                right=int(stack.pop())
                left=int(stack.pop())
                if token == '+': sum=left+right
                elif token == '-': sum=left-right
                elif token == '*': sum=left*right
                else: sum=int(float(left)/right)
                stack.append(str(sum))
        return int(stack.pop())

s=Solution()
s.evalRPN(["10", "6","9","3","+","-11","*","/","*","17","+","5","+"])
# 本题中 需要注意的主要是 x/y x//y遇到的一些问题，其中x/y是默认的真除法，如果两个是浮点数那么会产生小数，如果是整数那么不会产生小数，且向下取整，即负数向负无穷取整
# x//y则不管是小数还是整数都返回整数 只不过格式按照小数和整数返回一致的内容
# 在本题中可以观察案例3发现倾向于向上取整 因此需要使用浮点数做真除法随后int截断
# https://www.yuque.com/u29134184/skgkmw/gd1ixn1xzbzr1l0n?singleDoc# 《📘 Python 除法运算总结笔记》
