class Solution(object):

    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        isUp = None
        result = 1
        for i in range(1, len(nums)):
            if isUp != False and nums[i] < nums[i - 1]:
                result += 1
                isUp = False
            elif isUp != True and nums[i] > nums[i - 1]:
                result += 1
                isUp = True

        return result


# 0 上升 1平台 2下降
class Solution(object):

    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        status = 1
        curLength = 1
        result = 1
        for i in range(1, len(nums), 1):
            newStatus = None
            if nums[i] > nums[i - 1]:
                newStatus = 0
            elif nums[i] == nums[i - 1]:
                newStatus = 1
            else:
                newStatus = 2
            if newStatus == 0 and (status == 1 or status == 2):
                curLength += 1
                status = newStatus
            # 主要是需要注意平台期，当遇到平台期的时候其实不应该更新状态 因为这会丢失之前的上升/下降的状态
            # 平台期会覆盖掉上一次的真实趋势，导致连续相等元素后紧跟的同方向元素仍然被当作一次“转折”来计数，从而虚增长度
            elif newStatus == 1:
                pass
            elif newStatus == 2 and (status == 0 or status == 1):
                curLength += 1
                status = newStatus
            result = max(result, curLength)
        return result


class Solution(object):

    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        status = 1
        curLength = 1
        result = 1
        for i in range(1, len(nums), 1):
            newStatus = None
            if nums[i] > nums[i - 1]:
                newStatus = 0
            elif nums[i] == nums[i - 1]:
                newStatus = 1
            else:
                newStatus = 2
            if newStatus != 1 and newStatus != status:
                curLength += 1
                status = newStatus
            result = max(result, curLength)
        return result
