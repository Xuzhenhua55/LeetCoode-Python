class TreeNode:
    def __init__(self):
        self.son = dict()
        self.end = False

class Trie:

    def __init__(self):
        self.root = TreeNode()

    def insert(self, word: str) -> None:
        curNode = self.root
        for ch in word:
            if ch in curNode.son:
                curNode = curNode.son[ch]
            else:
                curNode.son[ch] = TreeNode()
                curNode = curNode.son[ch]
        curNode.end = True

    def find(self, target):
        curNode = self.root
        for ch in target:
            if ch not in curNode.son:
                return 0
            curNode = curNode.son[ch]
        return 2 if curNode.end else 1

    def search(self, word: str) -> bool:
        return self.find(word) == 2

    def startsWith(self, prefix: str) -> bool:
        return self.find(prefix) != 0


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)