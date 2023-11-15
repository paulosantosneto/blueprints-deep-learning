import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
import dgl

import matplotlib.pyplot as plt

from collections import defaultdict
from scipy import sparse

from .utilities import runtime

def load_dataset(args: dict):

    if args.dataset.lower() == 'cora':
        
        dataset = dgl.data.CoraGraphDataset()
        graph = dataset[0]
        num_class = dataset.num_classes

        feature_tensor = graph.ndata['feat'].T

        print(feature_tensor.shape)
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']

        sparse_matrix = graph.adjacency_matrix()
        adjacency_tensor = sparse_mx_to_torch_sparse_tensor(sparse_matrix)

        print(train_mask)

        labels = graph.ndata['label']

    elif args.dataset.lower() == 'zachary':

        dataset = dgl.data.KarateClubDataset()
        graph = dataset[0]
        num_class = dataset.num_classes

        feature_tensor = syntactic_feature_tensor_to_zachary_dataset(graph.num_nodes())

        train_mask = torch.ones(graph.num_nodes(), dtype=torch.bool)
        train_mask[20:] = False
        val_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        val_mask[20:27] = True
        test_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        test_mask[27:] = True

        sparse_matrix = graph.adjacency_matrix()
        adjacency_tensor = sparse_mx_to_torch_sparse_tensor(sparse_matrix)

        labels = graph.ndata['label']
    
    return feature_tensor, adjacency_tensor, num_class, labels, train_mask, val_mask, test_mask

@runtime
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    indices = sparse_mx.indices()
    values = sparse_mx.val
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

@runtime
def syntactic_feature_tensor_to_zachary_dataset(num_nodes: int):

    indices = torch.cat((torch.arange(num_nodes).unsqueeze(0), torch.arange(num_nodes).unsqueeze(0)), 0)
    values = torch.ones(num_nodes)
    shape = torch.Size([num_nodes, num_nodes])

    feature_tensor = torch.sparse_coo_tensor(indices, values, shape)

    return feature_tensor



