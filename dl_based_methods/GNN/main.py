import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F

from utils import *
from models import *

def train(model, optimizer, epochs, ftensor, adjtensor, labels, idx_train, idx_val):
    
    for epoch in range(epochs):
        start = time.time()
        model.train()
        optimizer.zero_grad()
        
        logits = model(ftensor)

        loss_train = F.nll_loss(logits[idx_train], labels[idx_train])
        acc_train = accuracy(logits[idx_train], labels[idx_train])
        
        loss_train.backward()
        optimizer.step()

        model.eval()
        logits = model(ftensor)

        loss_val = F.nll_loss(logits[idx_val], labels[idx_val])
        acc_val = accuracy(logits[idx_val], labels[idx_val])
        
        print(f'Epochs: {epoch} | Loss Train: {loss_train.item():.4f} | Acc Train: {acc_train:.2f} | Loss val: {loss_val:.4f} | Acc Val: {acc_val:.2f} | Time: {time.time() - start:.2f} seconds.')

def test(model, ftensor, labels, idx_test):

    model.eval()

    logits = model(ftensor)
    
    loss_test = F.nll_loss(logits[idx_test], labels[idx_test])
    acc_test = accuracy(logits[idx_test], labels[idx_test])

    print(f'Loss Test: {loss_test:.4f} | Acc: {acc_test:.2f}')

def main(args):

    if args.mode == 'train':

        feature_tensor, adjacency_tensor, num_class, labels, idx_train, idx_val, idx_test = load_dataset(args)

        model = GraphNeuralNetwork(nfeat=feature_tensor.shape[0],
                                   adj=adjacency_tensor,
                                   nhid=args.hidden_layers,
                                   nclass=num_class
        )

        optimizer = torch.optim.Adam(model.get_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        train(model, optimizer, args.epochs, feature_tensor, adjacency_tensor, labels, idx_train, idx_val)
        
        test(model, feature_tensor, labels, idx_test)

if __name__ == '__main__':

    args = get_args()

    main(args)
    
