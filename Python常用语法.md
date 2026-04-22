# Python 常用语法速查

## 1. 队列与双端队列

Python 中队列和双端队列都使用 `collections.deque`：

```python
from collections import deque

dq = deque([1, 2, 3])

# 右端（队尾）
dq.append(4)      # deque([1, 2, 3, 4])
dq.pop()          # deque([1, 2, 3])

# 左端（队首）
dq.appendleft(0)  # deque([0, 1, 2, 3])
dq.popleft()      # deque([1, 2, 3])
```

| 数据结构 | Python 实现 | 说明 |
|---------|-----------|------|
| 单端队列（FIFO） | `deque`（只用 `append` + `popleft`） | 一端入队，另一端出队 |
| 双端队列 | `deque` | 两端都能 O(1) 操作 |
| 优先队列 | `heapq`（小顶堆） | 基于 list |

单端队列没有独立实现，是双端队列的子集用法。`queue.Queue` 是线程安全队列，算法题一般用 `deque` 即可。

---

## 2. Counter 计数器

`collections.Counter` 用于快速统计元素出现次数，本质是 `dict` 的子类。

```python
from collections import Counter

# 创建
c = Counter("abracadabra")     # Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
c = Counter([1, 2, 2, 3, 3, 3])  # Counter({3: 3, 2: 2, 1: 1})

# 常用操作
c['a']            # 5（不存在的 key 返回 0，不会报 KeyError）
c['x']            # 0
list(c.elements())  # ['a','a','a','a','a','b','b','r','r','c','d']
c.most_common(2)  # [('a', 5), ('b', 2)]  出现次数最多的前 2 个

# 数学运算
c1 = Counter("aab")
c2 = Counter("abc")
c1 + c2           # Counter({'a': 3, 'b': 2, 'c': 1})
c1 - c2           # Counter({'a': 1})  负数和 0 会被忽略

# 更新
c.update("aaa")   # 在原有计数上累加
c.subtract("aa")  # 在原有计数上减少（允许负数）
```

### Counter vs 手动 dict 计数

```python
# 手动写法
d = {}
for ch in s:
    d[ch] = d.get(ch, 0) + 1

# Counter 写法
c = Counter(s)
```

### Counter 遍历

```python
c = Counter("abracadabra")

# 遍历 key（元素）
for key in c:
    print(key)           # a, b, r, c, d

# 遍历 key 和 count
for key, count in c.items():
    print(key, count)    # ('a', 5), ('b', 2), ...

# 遍历所有元素（展开，每个元素出现 count 次）
for elem in c.elements():
    print(elem)          # a, a, a, a, a, b, b, r, r, c, d

# 遍历按 count 排序（降序）
for key, count in c.most_common():
    print(key, count)    # ('a', 5), ('b', 2), ('r', 2), ('c', 1), ('d', 1)

# 遍历前 k 个高频元素
for key, count in c.most_common(3):
    print(key, count)    # ('a', 5), ('b', 2), ('r', 2)
```

### 注意：Counter 用 `==` 比较时会忽略 count 为 0 的 key

```python
Counter("aab") == Counter("aabc")  # True（'c' 的 count 为 0 被忽略）
```

如果需要精确比较 key 集合，可以用 `dict(Counter(...))` 转为普通 dict。

### Counter 子集判断

```python
c1 = Counter("aabbcc")
c2 = Counter("abc")

c1 >= c2   # True  c1 包含 c2
c2 <= c1   # True  c2 是 c1 的子集
```

`>=` / `<=` 判断每个 key 的 count 是否都大于等于对方，等价于 `c1 - c2` 不产生负数。

---

## 3. 二维数组排序

```python
arr = [[3, 1], [1, 4], [2, 2]]

# 按第一个元素排序（默认）
arr.sort(key=lambda x: x[0])

# 按第二个元素排序
arr.sort(key=lambda x: x[1])

# 先按第一个升序，再按第二个升序
arr.sort(key=lambda x: (x[0], x[1]))

# 第一个升序，第二个降序
arr.sort(key=lambda x: (x[0], -x[1]))
```

复杂排序规则可以抽成函数：

```python
def compare(item):
    w, h = item[0], item[1]
    return (-w * h, w)  # 面积降序，面积相同按宽升序

arr.sort(key=compare)
```

`key` 接受任何可调用对象，`def` 和 `lambda` 效果一样，`def` 适合多行逻辑。

---

## 4. dict 创建的两种方式

**方式一：直接用 `{}`（推荐）**

```python
numToStr = {
    2: "abc",
    3: "def",
    4: "ghi",
    5: "jkl",
    6: "mno",
    7: "pqrs",
    8: "tuv",
    9: "wxyz"
}
```

**方式二：用 `dict()` 配合键值对列表**

```python
numToStr = dict([
    (2, "abc"),
    (3, "def"),
    (4, "ghi"),
    (5, "jkl"),
    (6, "mno"),
    (7, "pqrs"),
    (8, "tuv"),
    (9, "wxyz")
])
```

**❌ 错误写法：`dict()` 关键字参数不支持数字作为键名**

```python
numToStr = dict(
    2: "abc",   # SyntaxError
    3: "def"
)
```

关键字参数的键必须是合法的 Python 标识符（字母或下划线开头），数字开头的标识符是非法的。因此数字键只能用 `{}` 或 `dict([...])` 创建。

---

## 5. heapq 堆（优先队列）

Python 使用 `heapq` 模块实现堆，默认是**最小堆**：

```python
import heapq

# 初始化
heap = []  # 直接用列表

# 入堆
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)

# 堆化已有列表（O(n)）
arr = [3, 1, 4, 2]
heapq.heapify(arr)  # arr 变成堆

# 弹出最小值（O(log n)）
min_val = heapq.heappop(heap)

# 查看堆顶（不弹出）
min_val = heap[0]

# 最大堆技巧：存负数
heapq.heappush(heap, -val)  # 存负数
max_val = -heapq.heappop(heap)  # 取出时再取负

# 元组比较：先比第一个元素
heapq.heappush(heap, (freq, val))  # 按 freq 排序
```

| 操作 | 语法 | 时间复杂度 |
|-----|------|-----------|
| 入堆 | `heapq.heappush(heap, val)` | O(log n) |
| 弹出最小值 | `heapq.heappop(heap)` | O(log n) |
| 堆化 | `heapq.heapify(list)` | O(n) |
| 查看堆顶 | `heap[0]` | O(1) |
| 前 k 大 | `heapq.nlargest(k, heap)` | O(n log k) |
| 前 k 小 | `heapq.nsmallest(k, heap)` | O(n log k) |

### 最大堆的实现方式

```python
# 方式一：存负数
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -1)
max_val = -heapq.heappop(max_heap)  # 取出 3

# 方式二：元组（用于需要额外信息的场景）
# 存 (-priority, item)，按 priority 降序
heapq.heappush(heap, (-priority, item))
```

### Top K 问题：维护大小为 k 的堆

```python
# 找最大的 k 个元素：用最小堆（堆顶是第 k 大）
heap = []
for num in nums:
    heapq.heappush(heap, num)
    if len(heap) > k:
        heapq.heappop(heap)  # 弹出最小的，保留大的

# 找最小的 k 个元素：用最大堆（堆顶是第 k 小）
heap = []
for num in nums:
    heapq.heappush(heap, -num)
    if len(heap) > k:
        heapq.heappop(heap)  # 弹出最大的（负数最小），保留小的
```

### 多属性元素的排序控制

heapq 默认按元素的**自然比较顺序**排序，元组会先比第一个元素，相等再比第二个：

```python
import heapq

heap = []
heapq.heappush(heap, (3, 'c'))
heapq.heappush(heap, (1, 'a'))
heapq.heappush(heap, (1, 'b'))  # (1,'a') 和 (1,'b') 按 'a' < 'b' 排序

heapq.heappop(heap)  # (1, 'a')
heapq.heappop(heap)  # (1, 'b')  第一个相等，按第二个排序
heapq.heappop(heap)  # (3, 'c')
```

**控制排序的关键：把排序依据放在元组第一位**

```python
# 按 freq 升序
heapq.heappush(heap, (freq, word))

# 按 freq 降序：存负数
heapq.heappush(heap, (-freq, word))

# 按 freq 升序，freq 相等按 word 降序
heapq.heappush(heap, (freq, -ord(word[0])))
```

**自定义类实现 `__lt__` 方法**

```python
class Node:
    def __init__(self, val, freq):
        self.val = val
        self.freq = freq

    def __lt__(self, other):
        return self.freq < other.freq  # 按 freq 升序

heap = []
heapq.heappush(heap, Node('a', 3))
heapq.heappush(heap, Node('b', 1))
heapq.heappop(heap)  # Node('b', 1)  freq 最小
```

| 方式 | 适用场景 | 示例 |
|-----|---------|------|
| 元组第一位放排序键 | 简单排序 | `(freq, val)` |
| 存负数 | 降序排序 | `(-freq, val)` |
| 自定义 `__lt__` | 复杂对象排序 | 类内实现比较逻辑 |

---

## 6. random.randint 语法细节

在 Python 中，`random.randint(a, b)` 的作用是返回一个在 `[a, b]` 范围内的随机整数。

关于它的语法细节，最需要注意的有以下几点：

### 1. 双闭区间（包含两端）
这是它最容易让人混淆的地方。与 Python 中常见的 `range(a, b)`（左闭右开，不包含 b）不同，`random.randint(a, b)` 是**双闭区间**，它**包含**起点 `a` 和终点 `b`。
* 也就是说，它返回的随机整数 $N$ 满足：$a \le N \le b$。

**示例：**
```python
import random

# 可能返回 1, 2, 3 中的任意一个，包含 3
num = random.randint(1, 3) 
```

### 2. 参数要求
* `a` 和 `b` 都必须是**整数**（Integer）。
* `a` 必须**小于或等于** `b`。如果 `a > b`，程序会抛出 `ValueError` 报错。
* 如果 `a == b`，它会固定返回 `a`。
