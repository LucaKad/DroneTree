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
    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.key) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

    # receive information from the commanding node or child node and spread if necessary
    def receive_event(self, event):
        print(str(self.key) + ": received " + event.info)
        self.send_event(event)

    # send information to the commanding node or child nodes
    def send_event(self, event):
        if event.spread_up and self.parent:
            print(str(self.key) + ": sent " + event.info)
            self.parent.receive_event(event)
        if event.spread_down:
            if self.children:
                for child in self.children:
                    print(str(self.key) + ": sent " + event.info)
                    child.receive_event(event)


class Root(Node):
    def __init__(self, key, parent, weight, model):
        super(Root, self).__init__(key, parent, weight)
        self.model = model
    # print tree in order from left to right
    def inorder(self, node):
        nodes = []
        if node is not None:
            nodes.append(node.key)
            for child in node.children:
                childNodes = self.inorder(child)
                nodes += childNodes
        return nodes

    def predict(self, features):
        print(self.model.predict(features))

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
            if node.children:

                childWeight = []
                for child in node.children:
                    childWeight.append(self.weightNode(child))
                node.children[childWeight.index(min(childWeight))] = Root(node.children[childWeight.index(min(childWeight))].key, None, node.children[childWeight.index(min(childWeight))].weight, node.model)
                for child in node.children:
                    if child != node.children[childWeight.index(min(childWeight))]:
                        child.parent = node.children[childWeight.index(min(childWeight))]
                        node.children[childWeight.index(min(childWeight))].children.append(child)
                return node.children[childWeight.index(min(childWeight))]
            else:
                return None
        return self


def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
    return float(num[:-1])



if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    features = pd.read_csv('testing.csv')
    print(features.head(5))
    print('The shape of our features is:', features.shape)
    print(features.describe())


    labels = np.array(features['p9'])
    features = features.drop('p9', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)
    from sklearn.model_selection import train_test_split

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf.fit(train_features, train_labels);
    predictions = rf.predict(test_features)
    for i in range(len(predictions)):
        predictions[i] = proper_round(predictions[i])
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2))
    accuracy = 100 - sum(x != 0 for x in errors) / len(errors) * 100
    print('Accuracy:', round(accuracy, 2), '%.')

    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    # tree rebuilding testing
    root = Root(5, None, 5, rf)
    root.insert(2, 5, 1)
    root.insert(22, 5, 1)
    root.insert(7, 2, 1)
    root.insert(6, 2, 1)
    root.insert(8, 7, 1)
    root.insert(9, 7, 2)
    root.insert(10, 8, 6)
    root.insert(11, 8, 3)
    root.insert(12, 10, 2)
    print(root)
    root.send_event(TreeEvent("foo", False, True))

    node = root.find(root, 8)
    root = root.delete_node(node)
    print(root.inorder(root))
    print("Структура дерева:")
    print(root)
    print("Видаляю елемент 5")
    node = root.find(root, 5)
    root = root.delete_node(node)
    print("Структура дерева:")
    print(root)
    features = np.array([[2,4,2,0,0,1,0,1]])
    print("Отримані параметри: ")
    print(features)
    print("За заданими параметрами було прийнято рішення:\n(1 - зробити фотографію\n0 - не робити фотографію)")
    root.predict(features)
