import random
import torch
import torch.nn as nn
import networkx as nx
import argparse

from networkx.generators.random_graphs import erdos_renyi_graph
from utils import *
from model import MyEmbedding, Node2Vec

def train(context_data: any, epochs: int, model: any, loss_fn: any, optimizer: any):
    ''' go through each epoch and adjust the parameters for each context. '''
        
    loss_history = []

    for epoch in range(epochs):
        local_loss = 0

        for cnode, context in context_data:

            context = torch.tensor([context], dtype=torch.long)
            cnode = torch.tensor([cnode], dtype=torch.long)

            probs = model(cnode)
            
            local_loss += loss_fn(probs, context)

        print(f'Epochs: {epoch} | Loss: {local_loss.item():.2f}')
        loss_history.append(local_loss.item())
            
        optimizer.zero_grad()
        local_loss.backward()
        optimizer.step()

    return loss_history

if __name__ == '__main__':

    args = get_args()

    if args.mode == 'train':

        nodes, edges = generate_random_graph(args.number_nodes, args.percent_edges)
        plot_graph(nodes, edges)

        G = list2dict(nodes, edges)

        context_data = Node2Vec(G, args.embedding_size, args.r, args.l, args.k, args.p, args.q).run()

        model = MyEmbedding(vocab_size=len(G.keys()), embedding_size=args.embedding_size).to(args.device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        loss_history = train(context_data, args.epochs, model, loss_fn, optimizer) 

        if args.plot_loss:
            plot_loss(args.epochs, loss_history)
        
        if args.plot_pca:
            plot_pca(G.keys(), model.w_target.detach())

        if args.save_model:
            save_model(model, str(args.epochs))
            save_parameters(len(G.keys()), args.embedding_size, args.l)

    elif args.mode == 'inference':

        params = load_parameters(args.model_config_path) 

        model = MyEmbedding(vocab_size=params['vocab_size'], embedding_size=params['embedding_size'])
        
        model.load_state_dict(torch.load(f'weights/{args.model_load_path}'))

        test_node = args.node
        X = torch.tensor([test_node], dtype=torch.long)
        
        nodes = torch.exp(torch.sort(model(X), dim=1, descending=True).values.detach()).numpy()[0][:args.most_similar]
        indices = torch.sort(model(X), dim=1, descending=True).indices.numpy()[0][:args.most_similar]
        
        for node, indice in zip(nodes, indices):
            
            print(f'Node {indice}: {node*100:.2f}%')
