import torch
import torch.nn as nn
import random

class MyEmbedding(nn.Module):

    def __init__(self, vocab_size: int, embedding_size: int):
        super(MyEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.w_target = nn.Parameter(torch.randn(vocab_size, embedding_size), requires_grad=True)
        self.w_context = nn.Parameter(torch.randn(vocab_size, embedding_size), requires_grad=True)
        self.Softmax = nn.LogSoftmax(dim=0)
     
    def forward(self, X):

        X = torch.matmul(self.w_context, self.w_target[X].T)
        X = self.Softmax(X)
        X = X.view(1, -1)

        return X

class Node2Vec(object):

    def __init__(self, G, dimension, n_walks, walk_length, context_size, out_param, inout_param):
        self.G = G 
        self.d = dimension
        self.r = n_walks
        self.l = walk_length
        self.k = context_size
        self.p = out_param
        self.q = inout_param
        self.walks = self.generate_walks()

    def generate_walks(self):
        '''
        Generate (N . r) random paths through the graph.

        :return: a list of paths.
        '''

        walks = []

        for i in range(self.r):
            nodes = list(self.G.keys())
            random.shuffle(nodes)

            for node in nodes:
                walk = self.bias_walk(self.G, node)
                walks.append(walk)

        return walks

    def bias_walk(self, G, node):
        '''
        traverses L nodes of the graph and adds them to a list of elements, 
        composing the trajectory of the biased path.

        :param G (dict):  
        :param node (int):

        :return: 
        '''

        walk = [node]

        walk.append(random.choice(G[node]))

        for walk_iter in range(1, self.l):
            curr = walk[-1]
            previous = walk[-2]

            new_node = self.getNeighbors(curr, previous, G)
            walk.append(new_node)
        
        return walk

    def getNeighbors(self, curr, previous, G):

        transitionProbabilities = []

        for node in G[curr]:
            if node == previous:
                transitionProbabilities.append(1 / self.p)
            elif previous in G[node]:
                transitionProbabilities.append(1)
            else:
                transitionProbabilities.append(1 / self.q)
    
        selected_node = random.choices(G[curr], transitionProbabilities)[0]
            
        return selected_node
    
    def run(self):
        ''' turns random walks into central nodes along with their contexts. '''

        context_data = []
        
        for walk in self.walks:
            if len(walk) == len(set(walk)):

                center_pos = self.l // 2
                center = walk[center_pos]
                context = [(center, node) for i, node in enumerate(walk) if i != center_pos]

                context_data.extend(context)
        
        return context_data