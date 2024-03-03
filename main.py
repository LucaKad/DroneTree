class TreeEvent:
    def __init__(self, info, spread_up=False, spread_down=False):
        self.info = info
        self.spread_up = spread_up
        self.spread_down = spread_down


class Node:
    def __init__(self, key, parent, weight):
        self.key = key
        self.children = []
        self.parent = parent
        self.weight = weight

    # receive information from the commanding node or child node and spread if necessary
    def receive_event(self, event):
        print("foo received")
        self.send_event(event, event.spread_up, event.spread_down)

    # send information to the commanding node or child nodes
    def send_event(self, event, send_parent=False, send_child=True):
        if send_parent and self.parent:
            print("parent bar sent")
        if send_child:
            if self.children:
                for child in self.children:
                    print("Child event sent")


class Root(Node):
    # print tree in order from left to right
    def inorder(self, node):
        nodes = []
        if node is not None:
            nodes.append(node.key)
            for child in node.children:
                childNodes = self.inorder(child)
                nodes += childNodes
        return nodes

    def weightNode(self, node):
        weight = 0
        if node is not None:
            weight += node.weight
            for child in node.children:
                childWeight = self.weightNode(child)
                weight += childWeight
        return weight

    # insert a node into the tree
    def insert(self, key, parent_key, weight):
        if self is None:
            return Node(key, None, weight)

        parent = self.find(self, parent_key)
        parent.children.append(Node(key, parent, weight))
        return self

    # find a node in the tree by its key value
    def find(self, node, x):
        if node.key == x:
            return node
        for child in node.children:
            n = self.find(child, x)
            if n:
                return n
        return None

    # delete a node from a tree
    def delete_node(self, node):
        if node is None:
            return

        if node.parent:
            node.parent.children.remove(node)
            if node.parent.children:
                childWeight = []
                for child in node.parent.children:
                    childWeight.append(self.weightNode(child))
                newParent = node.parent.children[childWeight.index(min(childWeight))]
                for child in node.children:
                    child.parent = newParent
                    newParent.children.append(child)
            else:
                for child in node.children:
                    child.parent = node.parent
        else:
            node = None


if __name__ == '__main__':
    # tree rebuilding testing
    root = Root(5, None, 5)
    root.insert(2, 5, 1)
    root.insert(7, 2, 1)
    root.insert(6, 2, 1)
    root.insert(8, 7, 1)
    root.insert(9, 7, 2)
    root.insert(10, 8, 6)
    root.insert(11, 8, 3)
    root.insert(12, 10, 2)
    node = root.find(root, 8)
    root.delete_node(node)
    print(root.inorder(root))
