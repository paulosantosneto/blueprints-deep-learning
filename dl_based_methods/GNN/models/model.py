import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphNeuralNetwork(nn.Module):

    def __init__(self, nfeat: int, adj: list, nclass: int, nhid: list=[64]):
        super(GraphNeuralNetwork, self).__init__()

        nhidden = [nfeat]
        nhidden.extend(nhid)
        nhidden.append(nclass)

        self.adj = adj

        self.layers = nn.ModuleList([
            GNNLayer(nhidden[i-1], nhidden[i]) 
            for i in range(1, len(nhidden))
        ])

    def forward(self, X):
        
        for layer in self.layers:
            X = F.relu(layer(X, self.adj))

        return F.log_softmax(X.T, dim=1)
    
    def get_parameters(self):
        return list(self.parameters()) + list(self.layers.parameters())

class GNNLayer(nn.Module):

    def __init__(self, in_d: int, out_d: int, bias=True):
        super(GNNLayer, self).__init__() 
        
        self.weight_u = nn.Parameter(torch.FloatTensor(in_d, out_d))
        self.weight_v = nn.Parameter(torch.FloatTensor(in_d, out_d))
        self.bias = nn.Parameter(torch.FloatTensor(out_d))

        # --- weight initialization ---

        std = 1. / (math.sqrt(self.weight_u.size(1)))
        self.weight_u.data.uniform_(-std, std)
        self.weight_v.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
        
    def forward(self, X, adj):

        # weight_u [d_{k} x d_{k-1}].T @ X [d_{k-1} x |V|] = Z_u [|V| x d_{k}].T
        zu = torch.spmm(self.weight_u.T, X).T
        # weight_v [d_{k} x d_{k-1}].T @ (X [d_{k-1} x |V|] @ adj [|V| x |V|]) = Z_v [|V| x d_{k}].T
        zv = torch.matmul(self.weight_v.T, torch.spmm(X, adj)).T

        hk = zu + zv + self.bias
        
        return hk.T


