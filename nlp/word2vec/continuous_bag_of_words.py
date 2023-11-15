import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA

from utils import *

class CBOW(nn.Module):

    def __init__(self, vocab_size: int, embedding_size: int):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.first_layer = nn.Linear(in_features=embedding_size, out_features=128)
        self.hidden_activation = nn.ReLU()
        self.second_layer = nn.Linear(in_features=128, out_features=vocab_size)
        self.out_activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.embeddings(x).sum(axis=0)[None]
        x = self.first_layer(x)
        x = self.hidden_activation(x)
        x = self.second_layer(x)
        x = self.out_activation(x)
        
        return x

def train_model(context_data: list, model: torch.nn.Sequential, loss_fn: any, optimizer: any, epochs:int):
    '''go through each epoch and adjust the parameters for each context.

    @context_data: tuple list. Each tuple contains a list with the indices of the context words and another element with the index of the center word.'''

    loss_history = []

    for epoch in range(epochs):
        local_loss = 0

        for context, cword in context_data:
            
            # in this step the purely string words are converted to their indices and then wrapped in a tensor.
            context = torch.tensor([word2idx[word] for word in context], dtype=torch.long)
            cword = torch.tensor([word2idx[cword]], dtype=torch.long)
            
            # passes the context to the model and returns a vector of the vocabulary size with the probabilites.
            probs = model(context)
            
            local_loss += loss_fn(probs, cword)
            
        print(f'Epochs: {epoch} | Loss: {local_loss.item():.2f}')
        loss_history.append(local_loss.item())

        optimizer.zero_grad()
        local_loss.backward()
        optimizer.step()
        
    return loss_history

if __name__ == '__main__':
    
    args = get_args()
    
    # loads the texts and defines the vocabulary (unique word).
    texts = load_doc(args.path_file)
    vocab = list(set(texts))
    
    # convert between words and their indices.
    # is this necessary because the input of model is the indices of words and not the words themselves.
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for i, word in enumerate(vocab)}

    # created the context of occurrence of each word (context, center_word).
    context_representation = create_context_representation(texts=texts, sliding_window=args.sliding_window)

    VOCAB_SIZE = len(vocab)
    EMBEDDING_SIZE = args.embedding_size
    
    model = CBOW(VOCAB_SIZE, EMBEDDING_SIZE)

    # defines the optimizer and loss function.
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss_history = train_model(context_representation, model, loss_fn, optimizer, args.epochs) 

    if args.plot:

        plot_loss(args.epochs, loss_history)
        plot_pca(vocab, model.embeddings.weight.detach()) 
    
    # example of inference. 
    context_test = [word.lower() for word in ['Artificial','Intelligence', 'has', 'emerged']]
    X = torch.tensor([word2idx[word] for word in context_test], dtype=torch.long)
    print(idx2word[torch.argmax(model(X)).item()])
        


