import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs.')
    parser.add_argument('--path_file', type=str, required=True, help='path of file containing the texts.')
    parser.add_argument('--sliding_window', default=2, type=int, help='span length from center word.')
    parser.add_argument('--embedding_size', default=100, type=int, help='the latent dimension or latent space to which the vocabulary will be reduced.')
    parser.add_argument('--plot', default=True, type=bool, help='in case you want to plot the graph of the loss function and PCA.')
    parser.add_argument('--mode', default='train', type=str, help='the execution mode: training or inference.')
    parser.add_argument('--model_load_path', type=str, required=True, help='path of weights.')

    args = parser.parse_args()
    return args

def load_doc(path: str):
    ''' Load the text file '''
    
    text = []

    with open(path, 'r') as f:
        text = [s.lower() for s in f.read().split()]

    return text

def save_model(model: any, model_type: str, epochs: str):

    try:
        if not os.path.exists('weights'):
            os.makedirs('weights')

        torch.save(model.state_dict(), f'weights/weights_{model_type}_{epochs}.pth')

        return 'Model was successfully saved!'
    except:
        raise 'There was a problem saving the model!'

def create_context_representation(texts: list, sliding_window: int):
    ''' creates the representation of contexts (context, center word).
    The context size is defined according to the sliding window parameter.'''

    all_contexts = []

    for center_word in range(sliding_window, len(texts) - sliding_window):
        context = np.concatenate((texts[center_word-sliding_window: center_word], \
                texts[center_word+1: center_word+sliding_window+1]), axis=0).tolist()
        word = texts[center_word]

        all_contexts.append((context, word))

    return all_contexts

def plot_loss(epochs: int, loss_history: list):

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
