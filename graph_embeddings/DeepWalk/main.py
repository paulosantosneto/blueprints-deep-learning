import random
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import os
import collections
from huffman import Huffman_tree

def dict2list(G: dict):
    
    nodes = G.keys()
    edges = []
    
    for node in nodes:
        for neighbor in G[node]:
            edges.append((node, neighbor))

    return nodes, edges

def draw_graphs(G: dict):
    
    nodes, edges = dict2list(G)
    g = nx.Graph()
    g.add_edges_from(edges)
    g.add_nodes_from(nodes)
    nx.draw_shell(g, with_labels=True, font_weight='bold')
    plt.savefig(os.getcwd() + '/graph.png')


G = {1: [2, 3, 4], 2: [1, 3], 3: [1, 2, 5], 4: [1, 5], 5: [3, 4]}
v = 'b c a a d d d c c a c a c a c'.split()

def huffman_tree(v):

    frequency = sorted([(key, value) for (key, value) in collections.Counter(v).items()], key=lambda x: x[1])

    ht = Huffman_tree(frequency)
    ht.build()
    ht.plot_tree()

class DeepWalk(object):

    def __init__(self):
        pass

    def deepwalk(self, G, y, t, w, d):
        
        for i in range(y):
            nodes = list(G.keys())
            random.shuffle(nodes)

            for node in nodes:
                Wvi = self.random_walk(G, node, t)
                #print(Wvi)
                self.skipGram(phi, Wvi, w) 

    def random_walk(self, G, v, t):
        
        w = [v]
        for i in range(t):
           node = random.choice(G[w[-1]])
           w.append(node)
        
        return w


    """ 
    def skipGram(self, phi, Wvi, w):

        for j, v in enumerate(Wvi):
            for k, u in enumerate(Wvi[j - w : j + w]
    """
"""
y = 2
t = 4 
w = 2 
d = 2 ** 4
dw = DeepWalk()
dw.deepwalk(G, y, t) 
"""
huffman_tree(v)
#draw_graphs(G)
