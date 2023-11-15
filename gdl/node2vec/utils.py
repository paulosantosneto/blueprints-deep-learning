import os
import torch
import json
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs.')
    parser.add_argument('--embedding_size', default=100, type=int, help='the latent dimension or latent space to which the nodes will be reduced.')
    parser.add_argument('--mode', default='train', type=str, help='the execution mode: training or inference.')
    parser.add_argument('--l', default=4, type=int, help='walk legth.')
    parser.add_argument('--r', default=3, type=int, help='walks per node.')
    parser.add_argument('--k', default=2, type=int, help='context size.')
    parser.add_argument('--number_nodes', default=50, type=int, help='number of nodes in graph.')
    parser.add_argument('--percent_edges', default=0.5, type=float, help='percentage of connections between nodes.')
    parser.add_argument('--model_load_path', type=str)
    parser.add_argument('--model_config_path', type=str)
    parser.add_argument('--node', default=0, type=int)
    parser.add_argument('--plot_graph', type=bool, default=False)
    parser.add_argument('--plot_pca', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--p', default=0.5, type=float, help='out parameter.')
    parser.add_argument('--q', default=0.5, type=float, help='In-out parameter.')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--most_similar', default=1, type=int)
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
    plt.plot(np.arange(epochs), loss_history, c='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.clf()

def plot_pca(vocab: list, data: list):

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    plt.scatter(data_pca[:, 0], data_pca[:, 1], c='black')

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

def generate_random_graph(number_nodes: int, porc_edges: float):
    
    g = erdos_renyi_graph(number_nodes, porc_edges)

    return g.nodes, g.edges

def plot_graph(nodes: list, edges: list):

    g = nx.Graph()
    g.add_edges_from(edges)
    g.add_nodes_from(nodes)
    layout = nx.circular_layout(g)
    nx.draw(g, alpha=0.9, node_size=1000, node_color='black', pos=layout, with_labels=True, font_color='whitesmoke')
    plt.savefig(os.getcwd() + '/graph.png')



