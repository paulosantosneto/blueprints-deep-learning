import numpy as np

class Node(object):

    def __init__(self, value=None, ID='accumulate', left=None, right=None):
        self.left = left
        self.right = right
        self.value = value
        self.ID = ID

class Huffman_tree(object):

    def __init__(self, frequency):
        self.ids = [id_ for (id_, value) in frequency][::-1]
        self.values = [value for (id_, value) in frequency][::-1]
        self.root = None 
        print(self.values)
        print(self.ids)

    def build(self):
        
        accumulate = sum(self.values)
        root = Node(value=accumulate)

        def tree(i, node):
            if i >= len(self.values) - 1:
                node.ID = self.ids[i]
                return 
            
            if node.value - self.values[i] > 0:
                if node.value - self.values[i] > self.values[i]:
                    node.left = Node(value=self.values[i], ID=self.ids[i])
                    node.right = Node(value=node.value - self.values[i])
                    tree(i+1, node.right)
                else:
                    node.right = Node(value=self.values[i], ID=self.ids[i])
                    node.left = Node(value=node.value - self.values[i])
                    tree(i+1, node.left)

        tree(0, root)
            
        self.root = root

    def plot_tree(self):

        p = self.root 

        def dfs(node):
            
            if node.left != None:
                print(node.value, node.ID)
                dfs(node.left)
            if node.right != None:
                dfs(node.right)
            if node.left == None and node.right == None:
                print(node.value, node.ID)

        dfs(self.root)







