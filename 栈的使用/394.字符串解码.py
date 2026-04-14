class Solution:
    def decodeString(self, s: str) -> str:
        # 第一次写的解法：把所有字符压栈，遇到 ] 时再处理
        # 能过就行，但不够优雅
        result = ""
        stack = list()
        for ch in s:
            if ch == '[':
                stack.append(str('['))
            elif (ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'z'):
                stack.append(str(ch))
            else:
                targetStr = ""
                while stack and stack[-1][0] >= 'a' and stack[-1][0] <= 'z':
                    targetStr = stack.pop() + targetStr
                stack.pop()  # pop '['
                cntStr = ""
                while stack and stack[-1][0] >= '0' and stack[-1][0] <= '9':
                    cntStr = stack.pop() + cntStr
                stack.append(int(cntStr) * targetStr)
        return ''.join(stack)


# 更优雅的解法：两个栈分别维护数字和字符串
class Solution:
    def decodeString(self, s: str) -> str:
        numStack = []
        strStack = []
        curNum = 0
        curStr = ""

        for ch in s:
            if ch.isdigit():
                curNum = curNum * 10 + int(ch)  # 处理多位数字
            elif ch == '[':
                # 遇到 [，把当前数字和字符串压栈，重置
                numStack.append(curNum)
                strStack.append(curStr)
                curNum = 0
                curStr = ""
            elif ch == ']':
                # 遇到 ]，弹出栈顶，拼接结果
                repeatNum = numStack.pop()
                prevStr = strStack.pop()
                curStr = prevStr + repeatNum * curStr
            else:
                curStr += ch

        return curStr