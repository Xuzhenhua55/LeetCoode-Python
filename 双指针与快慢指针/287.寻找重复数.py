class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # 本质上和链表找环是一样的
        # nums[i] 可以看作指向 nums[nums[i]] 的指针
        # 因为值范围是 1~n，索引范围是 0~n，所以形成了有环的"链表"
        slowIndex, fastIndex = 0, 0

        # 第一阶段：快慢指针找相遇点
        while True:
            fastIndex = nums[nums[fastIndex]]
            slowIndex = nums[slowIndex]
            if fastIndex == slowIndex:
                break

        # 第二阶段：从头开始，慢指针和相遇点指针同步前进找环入口
        slowIndex = 0
        while slowIndex != fastIndex:
            slowIndex = nums[slowIndex]
            fastIndex = nums[fastIndex]

        return slowIndex


# 感悟：
# 这道题本质是链表找环的入口
# nums[i] = j 可以看作节点 i 指向节点 j
# 因为值范围是 1~n，而索引范围是 0~n（长度为 n+1）
# 所以必然会形成环，重复的数就是环入口
# 快慢指针第一次相遇在环内，然后从头同步前进找环入口