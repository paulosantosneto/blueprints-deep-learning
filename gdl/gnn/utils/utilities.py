import os
import torch
import functools
import argparse
import time 

import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', default='train', help='the execution mode: training or inference.')
    parser.add_argument('--plot_graph', action='store_true')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--dataset', default='zachary')
    parser.add_argument('--hidden_layers', nargs='+', type=int)
    parser.add_argument('--weight_decay', default=0.0005)
    args = parser.parse_args()

    return args

def accuracy(logits, labels):

    preds = torch.argmax(logits, dim=1)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def runtime(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        start = time.time()
        out = f(*args, **kwargs)
        end = time.time()

        print(f"Function' {f.__name__}' took {end - start:.6f} seconds to execute.")

        return out
    
    return wrapper
