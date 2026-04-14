from typing import List


class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        """
        二分查找 - 左闭右开写法
        核心思想：
        - 单独数出现之前：成对元素的第一个数下标是偶数(0,2,4...)
        - 单独数出现之后：成对元素的第一个数下标是奇数(1,3,5...)
        """
        left, right = 0, len(nums)  # 左闭右开
        
        while left < right:
            mid = left + (right - left) // 2
            
            if mid % 2 == 0:  # mid是偶数，应该和mid+1配对
                if mid + 1 < len(nums) and nums[mid] == nums[mid + 1]:
                    left = mid + 2  # 单独数在右边
                else:
                    right = mid     # 单独数在左边或就是mid
            else:  # mid是奇数，应该和mid-1配对
                if nums[mid] == nums[mid - 1]:
                    left = mid + 1  # 单独数在右边
                else:
                    right = mid     # 单独数在左边
        
        return nums[left]


class Solution2:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        """
        异或写法 - 更简洁
        mid^1: 偶数变成下一个奇数，奇数变成前一个偶数
        """
        left, right = 0, len(nums)
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] == nums[mid ^ 1]:
                left = mid + 1
            else:
                right = mid
        
        return nums[left]


if __name__ == "__main__":
    # 测试用例
    solution = Solution()
    
    # 测试1: 单独数在中间
    print(solution.singleNonDuplicate([2, 2, 3, 3, 4, 5, 5]))  # 输出: 4
    
    # 测试2: 单独数在开头
    print(solution.singleNonDuplicate([1, 2, 2, 3, 3]))  # 输出: 1
    
    # 测试3: 单独数在末尾
    print(solution.singleNonDuplicate([1, 1, 2, 2, 3]))  # 输出: 3
    
    # 测试4: 只有一个元素
    print(solution.singleNonDuplicate([1]))  # 输出: 1
