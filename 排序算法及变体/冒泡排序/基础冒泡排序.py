def bubble_sort(arr):
    """
    基础冒泡排序算法实现
    :param arr: 待排序的列表
    :return: 排序后的列表
    """
    n = len(arr)
    # 遍历所有数组元素
    for i in range(n):
        # 提前退出冒泡循环的标志位
        swapped = False
        # 遍历数组从0到n-i-1
        for j in range(0, n - i - 1):
            # 如果当前元素大于下一个元素，则交换它们
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # 如果在这一轮中没有发生交换，说明数组已经有序，可以提前退出
        if not swapped:
            break
    return arr

if __name__ == "__main__":
    # 测试代码
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print("排序前:", test_arr)
    sorted_arr = bubble_sort(test_arr)
    print("排序后:", sorted_arr)
