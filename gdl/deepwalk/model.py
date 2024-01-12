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

class DeepWalk(object):

    def __init__(self, G: dict, T: int, gamma: int, w: int, d: int, args: dict):
        self.G = G
        self.T = T 
        self.gamma = gamma
        self.w = w
        self.walks = self.generating_random_walks()
        
    def generating_random_walks(self):
        '''
        Generate (N . y) random paths through the graph.

        :return: a list os paths.
        '''
        walks = []
        
        for _ in range(self.gamma):
            nodes = list(self.G.keys())
            random.shuffle(nodes)

            for node in nodes:
                walks.append(self.random_walk(node))

        return walks

    def random_walk(self, node: int):
        '''
        Transverses paths through the graph from a uniform distribuition.

        :param node: start node.

        :return: a list containing the path taken.
        '''

        path = [node]
    
        for _ in range(self.T):
            node = random.choice(self.G[path[-1]])
            path.append(node)

        return path

    def run(self):
        ''' turns random walks into central nodes along with their contexts. '''

        context_data = []

        for walk in self.walks:
            if len(walk) == len(set(walk)):
            
                center_pos = self.T // 2
                center = walk[center_pos]
                context = [(center, node) for i, node in enumerate(walk) if i != center_pos]

                context_data.extend(context)
        
        return context_data
