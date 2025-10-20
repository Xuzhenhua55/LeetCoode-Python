## 一、题目描述：计算矩形的 IoU (Intersection over Union)

**题目名称：**
计算两个轴对齐矩形的交并比（Intersection over Union，IoU）

**题目说明：**
给定两个矩形，它们的边与坐标轴平行。每个矩形由左下角点和右上角点坐标表示，格式如下：

```
rect1 = [x1_min, y1_min, x1_max, y1_max]
rect2 = [x2_min, y2_min, x2_max, y2_max]
```

请你计算这两个矩形的 **IoU（交并比）**，即：$IoU = \frac{\text{交集面积}}{\text{并集面积}}$

其中：

* 交集面积是两个矩形重叠部分的面积；
* 并集面积 = 面积1 + 面积2 − 交集面积。

如果两矩形不相交，返回 `0.0`。

---

## 二、示例

**输入：**

```python
rect1 = [0, 0, 2, 2]
rect2 = [1, 1, 3, 3]
```

**输出：**

```python
0.14285714285714285
```

**解释：**

* 矩形1面积 = 2×2 = 4
* 矩形2面积 = 2×2 = 4
* 交集矩形坐标 = [1, 1, 2, 2] → 面积 = 1
* 并集面积 = 4 + 4 − 1 = 7
* IoU = 1 / 7 ≈ 0.14285714

---

## 💡 三、代码思路

1. **计算交集边界**：

   * 左边界 = `max(x1_min, x2_min)`
   * 下边界 = `max(y1_min, y2_min)`
   * 右边界 = `min(x1_max, x2_max)`
   * 上边界 = `min(y1_max, y2_max)`

2. **判断是否相交**：

   * 若 `right <= left` 或 `top <= bottom`，则不相交，IoU = 0

3. **计算面积**：

   * 面积1 = `(x1_max - x1_min) * (y1_max - y1_min)`
   * 面积2 = `(x2_max - x2_min) * (y2_max - y2_min)`
   * 交集面积 = `(right - left) * (top - bottom)`
   * 并集面积 = 面积1 + 面积2 − 交集面积

4. **计算IoU**：

   * 返回 `intersection / union`

---

## 四、LeetCode风格Python代码实现

```python
class Solution:
    def computeIoU(self, rect1: list[int], rect2: list[int]) -> float:
        # 计算交集边界
        x_left = max(rect1[0], rect2[0])
        y_bottom = max(rect1[1], rect2[1])
        x_right = min(rect1[2], rect2[2])
        y_top = min(rect1[3], rect2[3])
        
        # 若不相交
        if x_right <= x_left or y_top <= y_bottom:
            return 0.0
        
        # 面积计算
        inter_area = (x_right - x_left) * (y_top - y_bottom)
        area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        
        union_area = area1 + area2 - inter_area
        
        # IoU 计算
        iou = inter_area / union_area
        return iou


# 示例测试
if __name__ == "__main__":
    s = Solution()
    rect1 = [0, 0, 2, 2]
    rect2 = [1, 1, 3, 3]
    print(s.computeIoU(rect1, rect2))  # 输出: 0.14285714285714285
```
