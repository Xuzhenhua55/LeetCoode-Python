class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        result = 0
        wndSet = set()  # 用 set 记录窗口内的字符
        left = 0

        for i, ch in enumerate(s):
            # 如果 ch 已在窗口内，收缩左边界直到 ch 不在窗口内
            while ch in wndSet:
                wndSet.remove(s[left])
                left += 1

            # 加入当前字符，更新最大长度
            wndSet.add(ch)
            result = max(result, len(wndSet))

        return result


# ============================================================
# 核心思路：滑动窗口 + set
# ============================================================
#
# 维护一个窗口 [left, i]，窗口内无重复字符
# - 用 set 记录窗口内已有的字符
# - 遇到重复字符时，收缩左边界直到重复字符被移除
# - 每次扩展窗口后更新最大长度
#
# 时间复杂度：O(n)，每个字符最多进出窗口各一次
# 空间复杂度：O(min(n, 字符集大小))
#
# 变式：用 dict 记录字符最后出现的位置，可以直接跳过
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        charToIndex = {}  # 记录字符最后出现的索引
        left = 0
        result = 0

        for i, ch in enumerate(s):
            if ch in charToIndex and charToIndex[ch] >= left:
                # ch 在窗口内，直接跳到 ch 最后出现位置的下一位
                left = charToIndex[ch] + 1
            charToIndex[ch] = i
            result = max(result, i - left + 1)

        return result