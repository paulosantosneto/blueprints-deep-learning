import os
import torch
import json
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
import numpy as np

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs.')
    parser.add_argument('--embedding_size', default=100, type=int, help='the latent dimension or latent space to which the nodes will be reduced.')
    parser.add_argument('--mode', default='train', type=str, help='the execution mode: training or inference.')
    parser.add_argument('--T', default=4, type=int)
    parser.add_argument('--gamma', default=3, type=int)
    parser.add_argument('--w', default=2, type=int)
    parser.add_argument('--number_nodes', default=50, type=int)
    parser.add_argument('--edges', default=0.5, type=float)
    parser.add_argument('--model_load_path', type=str)
    parser.add_argument('--model_config_path', type=str)
    parser.add_argument('--node', type=str)
    
    args = parser.parse_args()

    return args

def save_parameters(vocab_size: int, embedding_size: int, context_size: int):
    
    params = {
            'vocab_size': vocab_size,
            'embedding_size': embedding_size,
            'context_size': context_size
    }

    with open('weights/parameters.json', 'w') as json_config:
        json.dump(params, json_config)

def load_parameters(config_path):
    
    params = {}

    with open(f'weights/{config_path}', 'r') as json_config:
        params = json.load(json_config)

    return params

def load_graph(path: str):
    ''' Load the json graph file '''
    
    graph = {}

    with open(f'{path}') as f:
        graph = json.load(f)

    return graph

def save_model(model: any, epochs: str):

    try:
        if not os.path.exists('weights'):
            os.makedirs('weights')

        torch.save(model.state_dict(), f'weights/weights_{epochs}.pth')

        return 'Model was successfully saved!'
    except:
        raise 'There was a problem saving the model!'

def plot_loss(epochs: int, loss_history: list):

    plt.clf()
    plt.plot(np.arange(epochs), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.clf()

def plot_pca(vocab: list, data: list):

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    plt.scatter(data_pca[:, 0], data_pca[:, 1])

    for i, word in enumerate(vocab):
        plt.annotate(word, (data_pca[i, 0], data_pca[i, 1]), fontsize=8)

    plt.savefig('PCA.png')

def edges_nodes(graph: list):

    nodes = []
    
    for nds in graph:
        nodes.extend(nds)
        
    nodes = set(nodes)
    
    return nodes, graph

def list2dict(nodes: list, edges: list):

    graph = {i: [] for i in nodes}

    for node in nodes:
        for edge in edges:
            if edge[1] not in graph[edge[0]]:
                graph[edge[0]].append(edge[1])
            if edge[0] not in graph[edge[1]]:
                graph[edge[1]].append(edge[0])
    
    return graph

def plot_graph(nodes: list, edges: list):

    g = nx.Graph()
    g.add_edges_from(edges)
    g.add_nodes_from(nodes)
    layout = nx.random_layout(g)
    nx.draw(g, pos=layout, with_labels=True)
    plt.savefig(os.getcwd() + '/graph.png')


