# Python 常见语法陷阱

## 1. sorted() 返回的是 list，不是 str

```python
s = "bac"
sorted(s)        # 返回 ['a', 'b', 'c']（list），不是 "abc"
type(sorted(s))  # <class 'list'>
```

**陷阱：list 不可哈希，不能作为 dict 的 key。**

```python
d = {}
key = sorted("bac")       # key 是 list
d[key] = 1                # TypeError: unhashable type: 'list'
```

**正确做法：用 `''.join()` 转回字符串。**

```python
key = ''.join(sorted("bac"))  # key = "abc"（str）
d[key] = 1                     # OK
```

> 实际踩坑：[49. 字母异位词分组](../哈希表/49.字母异位词分组.py)
> 当时写 `sorted_str = sorted(str)`，将 list 作为 dict key，导致运行时报错。

---

## 2. dict.values() 返回的是 dict_values，不是 list

```python
d = {'a': [1, 2], 'b': [3, 4]}
result = d.values()
type(result)  # <class 'dict_values'>
```

`dict_values` 不是 `list`，虽然可以迭代，但不能用下标访问，也不满足 `List[List[str]]` 类型签名。

**正确做法：用 `list()` 包一层。**

```python
result = list(d.values())  # [[1, 2], [3, 4]]
```

> 同样踩坑于 [49. 字母异位词分组](../哈希表/49.字母异位词分组.py)

---

## 3. sorted() vs .sort() 区别

| | `sorted()` | `.sort()` |
|---|---|---|
| 适用对象 | 任何可迭代对象 | 仅 list |
| 返回值 | 返回新 list | 返回 None（原地修改） |
| 原数据 | 不变 | 被修改 |

```python
# sorted
s = "cba"
new = sorted(s)      # new = ['c','b','a'], s 不变

# .sort()
arr = [3, 1, 2]
arr.sort()           # arr = [1, 2, 3]，原地修改
```

