from collections import deque

class MinStack:

    def __init__(self):
        self.fullStack = deque()
        self.monoStack = deque()

    def push(self, val: int) -> None:
        self.fullStack.append(val)
        # 只有比当前最小值更小（或相等）才入栈
        if not self.monoStack or val <= self.monoStack[-1]:
            self.monoStack.append(val)

    def pop(self) -> None:
        val = self.fullStack.pop()
        # pop 的值恰好是当前最小值时，才从 monoStack 中移除
        if self.monoStack[-1] == val:
            self.monoStack.pop()

    def top(self) -> int:
        return self.fullStack[-1]

    def getMin(self) -> int:
        return self.monoStack[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()