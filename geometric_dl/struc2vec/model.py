import numpy as np
from dtw import dynamic_time_warping
from itertools import chain
import math
import torch.nn as nn
import torch

class Struc2Vec(object):

    def __init__(self, G: list, k: int, walk_length: int, number_walks: int, q: int, nodes: list, edges: list):
        """ 
        Args:
            - G (int): adjacency list of the selected graph;
            - k (int): number of layers;
            - walk_length (int): hyperparameter relative to the size of the walk;
            - q (float): hyperparameter that controls the probability of staying in the same layer;
            - number_walks (int): hyperparameter relative to the number of walks per node;
            - nodes (list): list with unique identifiers for each node;
            - edges (list): list with connections between nodes in the format (u, v).

        """
        
        # --- graph properties  ---
        self.G = G
        self.k = k
        self.nodes = nodes
        self.edges = edges
        self.num_nodes = (len(self.nodes) * (len(self.nodes) - 1)) // 2

        # --- hyperparameters ---

        self.walk_length = walk_length
        self.number_walks = number_walks
        self.q = q 

        # --- operational matrices ---

        self.F = np.zeros((len(nodes), len(nodes), k))
        self.R = {}
        self.upper_connect_layers = np.zeros((k, len(nodes)))
        self.down_connect_layers = np.zeros((k, len(nodes)))
        self.down_connect_layers[1:] = 1
        self.avg_edge_weight_graphs = np.zeros(k)

        # --- function call ---

        self.generate_set_nodes_distance()
        
    def run(self):
        
        # --- STEP 1: measuring structural similarity ---

        # generation of the distance matrix between the sets of degrees of the nodes given K-hop neighbors
        for ki in range(self.k):
            self.measuring_structural_similarity(ki)
        
        # simulates the recursive process of accumulation of layer-dependent structural similarities
        self.F = np.cumsum(self.F, axis=2)     

        # --- STEP 2: Constructing the context graph ---

        self.W = np.exp(-self.F).transpose(2, 0, 1)
         
        self.calculate_avg_edges_weights()

        self.constructing_context_graph()
        
        # --- STEP 3: Generating context for nodes ---

        context_data = self.generate_context_for_nodes()

        return context_data
    
    def generate_context_for_nodes(self):
        
        context_data = []

        for node in self.nodes:
            
            for nwalks in range(self.number_walks):
                walk = [node]
                layer = [0]
                
                for length in range(self.walk_length):
                    selected_node, selected_layer = self.calculate_probs(walk[-1], layer[-1])
                    walk.append(selected_node)
                    layer.append(selected_layer)
                
                context_data.append(walk)

        return context_data

    def calculate_probs(self, node: int, layer: int):
        
        if np.random.random() <= self.q:
            # stays same layer

            # calculates the normalization factor (Zk)
            Zk = np.sum(self.W[layer][:, node], axis=0) - self.W[layer][node, node]

            # generates the probabilities excluding the target node
            new_column = self.W[layer][:, node]
            new_column[node] = 0
            probs = new_column / Zk

            # selects the node according to the probabilities
            selected_node = np.random.choice(len(probs), p=probs)
            
            return selected_node, layer
        else:
            # change layer
            Pk_up = (self.upper_connect_layers[layer][node]) / (self.upper_connect_layers[layer][node] + self.down_connect_layers[layer][node])
            if np.random.random() <= 1 - Pk_up:
                # go to bottom layer
                return node, layer - 1
            else:
                # go to top layer
                return node, layer + 1

    def calculate_avg_edges_weights(self):
        
        for k, adj_graph in enumerate(self.W):
            # filters the upper triangular matrix in W and averages the values
            self.avg_edge_weight_graphs[k] = np.triu(self.W[k], k=1).sum() / self.num_nodes

    def generate_set_nodes_distance(self):

        for node in self.nodes:
            self.R[node] = self.BFS(node)
            
    def measuring_structural_similarity(self, k: int):

        for i, nodei in enumerate(self.nodes):
            for j, nodej in enumerate(self.nodes):
                if len(self.R[nodei]) >= k and len(self.R[nodej]) >= k:
                    self.F[i, j, k] = dynamic_time_warping(self.R[nodei][k], self.R[nodej][k])

    def constructing_context_graph(self):
        
        for k in range(self.k - 1):
            for i, nodes in enumerate(self.nodes):
                gammak = len(self.W[k][:, i][self.W[k][:, i] > self.avg_edge_weight_graphs[k]])
                self.upper_connect_layers[k][i] = np.log(gammak + np.exp(1))

    def BFS(self, node: int):
        """returns a list of nodes for k-hop neighbors"""
        
        queue_neighbors = [node]
        history = set([]) 
        kneighbors = []

        for k in range(self.k):
            history.update(queue_neighbors)
            aux_queue_neighbors = queue_neighbors[:]
            queue_neighbors = [self.G[neighbor] for neighbor in queue_neighbors]
            degree_neighbors = [len(neighbor) for neighbor in queue_neighbors]
            queue_neighbors = list(chain(*queue_neighbors))
            queue_neighbors = list(set([node for node in queue_neighbors if node not in history]))
            kneighbors.append(sorted(degree_neighbors, reverse=True))
        
        return kneighbors


            


