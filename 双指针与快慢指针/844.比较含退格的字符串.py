class Solution(object):
    def afterBackspace(self, input_str):
        str_list = list(input_str)
        slow = 0
        for fast in range(len(str_list)):
            if str_list[fast] != '#':
                str_list[slow] = str_list[fast]
                slow += 1
            else:
                slow -= 1
                if slow < 0:
                    slow = 0
        return str_list, slow

    def backspaceCompare(self, s, t):
        s1, len_s1 = self.afterBackspace(s)
        t1, len_t1 = self.afterBackspace(t)
        return s1[:len_s1] ==t1[:len_t1]