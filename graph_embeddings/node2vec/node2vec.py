import random
import torch
import torch.nn as nn
import networkx as nx
import argparse

from networkx.generators.random_graphs import erdos_renyi_graph
from utils import *

class Node2vec(object):

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
            nodes = list(G.keys())
            random.shuffle(nodes)

            for node in nodes:
                walk = self.bias_walk(G, node)
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
                context = [node for i, node in enumerate(walk) if i != center_pos]
                context_data.append((center, context))
        
        return context_data

def train(context_data: any, epochs: int, model: any, loss_fn: any, optimizer: any):
    '''Go through each epoch and adjust the parameters for each context. '''

    loss_history = []

    for epoch in range(epochs):
        local_loss = 0

        for cnode, context in context_data:
            context = torch.tensor(context, dtype=torch.long)
            cnode = torch.tensor([cnode], dtype=torch.long)

            probs = model(cnode)

            local_loss += loss_fn(probs, context)

        print(f'Epochs: {epoch} | Loss: {local_loss.item():.2f}')
        loss_history.append(local_loss.item())
        
        optimizer.zero_grad()
        local_loss.backward()
        optimizer.step()

    return loss_history

class SkipGram(nn.Module):

    def __init__(self, vocab_size: int, embedding_size: int, context_size: int):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.first_layer = nn.Linear(in_features=embedding_size, out_features=128)
        self.hidden_activation = nn.ReLU()
        self.second_layer = nn.Linear(in_features=128, out_features=context_size * vocab_size)
        self.out_activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.first_layer(x)
        x = self.hidden_activation(x)
        x = self.second_layer(x)
        x = x.view(self.context_size, self.vocab_size)

        return x

def generate_random_graph(number_nodes: int, porc_edges: float):
    
    g = erdos_renyi_graph(number_nodes, porc_edges)

    return g.nodes, g.edges

if __name__ == '__main__':


    args = get_args()

    if args.mode == 'train':

        nodes, edges = generate_random_graph(args.number_nodes, args.percent_edges)
        plot_graph(nodes, edges)

        G = list2dict(nodes, edges)

        context_data = Node2vec(G, args.embedding_size, args.r, args.l, args.k, args.p, args.q).run()

        model = SkipGram(vocab_size=len(G.keys()), embedding_size=args.embedding_size, context_size=args.l)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        loss_history = train(context_data, args.epochs, model, loss_fn, optimizer) 

        plot_loss(args.epochs, loss_history)
        plot_pca(G.keys(), model.embeddings.weight.detach())

        save_model(model, str(args.epochs))
        save_parameters(len(G.keys()), args.embedding_size, args.l)

    elif args.mode == 'inference':

        params = load_parameters(args.model_config_path) 

        model = SkipGram(vocab_size=params['vocab_size'], embedding_size=params['embedding_size'], context_size=params['context_size'])
        
        model.load_state_dict(torch.load(f'weights/{args.model_load_path}'))

        test_node = args.node
        X = torch.tensor([test_node], dtype=torch.long)
        print([node for node in torch.argmax(model(X), dim=1).numpy()])
