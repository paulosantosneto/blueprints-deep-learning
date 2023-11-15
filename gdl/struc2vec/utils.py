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
    parser.add_argument('--embedding_size', default=300, type=int, help='the latent dimension or latent space to which the nodes will be reduced.')
    parser.add_argument('--mode', default='train', type=str, help='the execution mode: training or inference.')
    parser.add_argument('--graph', default='karate_club_graph', type=str)
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--plot_graph', type=bool, default=False)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--number_walks', default=3, type=int)
    parser.add_argument('--walk_length', default=6, type=int)
    parser.add_argument('--window', default=4, type=int)
    parser.add_argument('--q', default=0.5, type=float)
    parser.add_argument('--model_load_path', type=str)
    parser.add_argument('--model_config_path', type=str)
    parser.add_argument('--plot_pca', type=bool, default=False)
    parser.add_argument('--plot_loss', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--number_nodes', default=20, type=int, help='number of nodes in erdos graph.')
    parser.add_argument('--percent_edges', default=0.3, type=float, help='percentage of connections between nodes in erdos graph.')
    parser.add_argument('--target', default=0, type=int)
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

def list2dict(nodes: list, edges: list):

    graph = {i: [] for i in nodes}

    for node in nodes:
        for edge in edges:
            if edge[1] not in graph[edge[0]]:
                graph[edge[0]].append(edge[1])
            if edge[0] not in graph[edge[1]]:
                graph[edge[1]].append(edge[0])
    
    return graph

# --- Plots ---

def plot_pca(word_vectors: list, args: dict, labels: list):

    pca = PCA(n_components=2)
    word_vectors_2d = pca.fit_transform(word_vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], alpha=0.5)

    for i, label in enumerate(labels):
        x, y = word_vectors_2d[i]
        plt.text(x, y, label, fontsize=10)

    plt.title('Word Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
            
    plt.savefig(f'{args.graph}_pca.png')

def plot_graph(nodes: list, edges: list, seed: int):
    
    g = nx.Graph()
    g.add_edges_from(edges)
    g.add_nodes_from(nodes)
    layout = nx.spring_layout(g, seed=seed)
    nx.draw(g, alpha=0.9, node_size=300, node_color='Black', with_labels=True, font_color='whitesmoke')
    plt.savefig(os.getcwd() + '/graph.png')

# --- Graphs ---

def karate_club_graph():

    G = nx.karate_club_graph()
    
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    num_nodes = len(nodes)

    nodes2 = list(np.array(G.nodes) + num_nodes)
    edges2 = [(n1+num_nodes, n2+num_nodes) for n1, n2 in edges]
    
    nodes.extend(nodes2)
    edges.extend(edges2)
    edges.extend([(0, 34)])

    return nodes, edges

def barbell_graph():

    G = nx.barbell_graph(10, 10)

    return G.nodes, G.edges

def generate_random_graph(number_nodes: int, porc_edges: float, seed: int):
    
    g = erdos_renyi_graph(number_nodes, porc_edges, seed=seed)

    return g.nodes, g.edges


