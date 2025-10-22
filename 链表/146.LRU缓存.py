class Node(object):

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.head, self.tail = Node(), Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.keyToNode = dict()

    def moveToTail(self, key):
        curNode = self.keyToNode[key]
        curNode.prev.next = curNode.next
        curNode.next.prev = curNode.prev
        curNode.prev = self.tail.prev
        curNode.next = self.tail
        self.tail.prev.next = curNode
        self.tail.prev = curNode

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.keyToNode:
            self.moveToTail(key)
            return self.keyToNode[key].value
        return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.keyToNode:
            self.keyToNode[key].value = value
            self.moveToTail(key)
        else:
            if len(self.keyToNode) == self.capacity:
                oldNode = self.head.next
                self.keyToNode.pop(oldNode.key)
                oldNode.key, oldNode.value = key, value
                self.keyToNode[key] = oldNode
                self.moveToTail(key)
            else:
                newNode = Node(key, value)
                newNode.next = self.tail
                newNode.prev = self.tail.prev
                self.tail.prev.next = newNode
                self.tail.prev = newNode
                self.keyToNode[key] = newNode


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
