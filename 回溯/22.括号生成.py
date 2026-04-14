class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        self.result = list()
        self.curPath = list()

        def isValid(targetPath):
            if not targetPath: return False
            stack = list()
            for ch in targetPath:
                if ch == '(':
                    stack.append(ch)
                else:
                    if stack and stack[-1] == '(':
                        stack.pop()
                    else:
                        stack.append(ch)
            return not stack

        def dfs():
            if len(self.curPath) == 2*n and isValid(self.curPath):
                self.result.append(''.join(self.curPath))
            if len(self.curPath) > 2*n: return

            self.curPath.append('(')
            dfs()
            self.curPath.pop()

            self.curPath.append(')')
            dfs()
            self.curPath.pop()

        dfs()
        return self.result