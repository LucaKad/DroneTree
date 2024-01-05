class TreeEvent:
    def __init__(self, info, spread_up = False, spread_down = False):
        self.info = info
        self.spread_up = spread_up
        self.spread_down = spread_down
class Node:
    def __init__(self, key, parent):
        self.key = key
        self.left = None
        self.right = None
        self.parent = parent

    #receive information from the commanding node or child node and spread if necessary
    def receive_event(self, event):
        print("foo received")
        self.send_event(event, event.spread_up, event.spread_down)


    #send information to the commanding node or child nodes
    def send_event(self, event, send_parent = False, send_child = True):
        if(send_parent and self.parent):
            print("parent bar sent")
        if(send_child):
            if (self.left):
                print("left bar sent")
            if (self.right):
                print("right bar sent")

class Root(Node):
    #print tree in order from left to right
    def inorder(self, node):
        if node is not None:
            self.inorder(node.left)
            print(node.key, end=' ')
            self.inorder(node.right)

    #insert a node into the tree
    def insert(self, key, parent_key):
        if self is None:
            return Node(key, None)

        parent = self.find(self, parent_key)
        if(parent.left == None):
            parent.left = Node(key, parent)
        elif(parent.right == None):
            parent.right = Node(key, parent)
        else:
            print("Parent already has two child nodes")
        return self

    #find a node in the tree by it's key value
    def find(self, node, x):
        if node.key == x:
            return node
        if(node.left != None):
            n = self.find(node.left, x)
            if n:
                return n
        if(node.right != None):
            n = self.find(node.right, x)
            if n:
                return n
        return None

    #delete a node from a tree
    def delete_node(self, node):
        if node is None:
            return

        if node.left:
            replacement = node.left
            replacement.parent = node.parent
            if node.parent:
                if node.parent.left == node:
                    node.parent.left = replacement
                else:
                    node.parent.right = replacement
            else:
                node = replacement

            if replacement.right:
                #if the replacing node has a right child, move that right child to the replacing node's left child
                self.move_left_children_up(replacement.left, replacement.right)
                replacement.right = None

            #move the replaced node's right child to the replacing node
            replacement.right = node.right
            if node.right:
                node.right.parent = replacement
        else:
            if node.parent:
                if node.parent.left == node:
                    node.parent.left = node.right
                else:
                    node.parent.right = node.right
            else:
                node = None

    def move_left_children_up(self, left_child, right_subtree):
        #if left child exist, attach the right child to it
        if left_child:
            right = left_child.right
            left_child.right = right_subtree
            right_subtree.parent = left_child
            if(right != None):
                self.move_left_children_up(left_child.left, right)


if __name__ == '__main__':
    #tree rebuilding testing
    root = Root(5, None)
    root.insert(2, 5)
    root.insert(7, 2)
    root.insert(6, 2)
    root.insert(8, 7)
    root.insert(9, 7)
    root.insert(10, 8)
    root.insert(11, 8)
    root.insert(12, 10)
    node = root.find(root, 2)
    root.delete_node(node)
    root.inorder(root)

