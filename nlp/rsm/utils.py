import os
import re
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

def get_args():

	parser = argparse.ArgumentParser()

	# --- general ---

	parser.add_argument('--device', default='cuda', type=str)
	parser.add_argument('--seed', default='123', type=str)

	# --- train ---
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--path_file', default='./data/input.txt', type=str)
	parser.add_argument('--mode', default='train', type=str)
	parser.add_argument('--sequence_length', default=10, type=int)
	parser.add_argument('--dimension', default=128, type=int)
	parser.add_argument('--save_model', default=True, type=bool)
	parser.add_argument('--sequence_model', nargs='+', default='rnn', choices=['rnn', 'lstm', 'gru'], type=str)
	parser.add_argument('--plot_loss', action='store_true')
	parser.add_argument('--learning_rate', default=0.01, type=float)
	parser.add_argument('--verbose', action='store_true')

	# --- inference ---

	parser.add_argument('--model_load_path', nargs='+', default=None, type=str)
	parser.add_argument('--model_config_path', nargs='+', default=None, type=str)
	parser.add_argument('--max_length', default=20, type=int)
	parser.add_argument('--start_sequence', nargs='+', default='artificial', type=str)
	parser.add_argument('--temperature', default=0.1, type=float)
	args = parser.parse_args()

	if args.mode == 'inference' and (args.model_load_path is None or args.model_config_path is None):
		parser.error('--model_load_path is mandatory when --mode is "inference"')

	return args

# --- leading with loading ---

def load_doc(path: str):
	''' Load the text file '''
	
	content = None

	with open(path, 'r') as f:
		content = f.readlines()

	return content

def preprocess_text(raw_text_lines: list):
	
	cleaned_text_lines = [re.sub('\n+', '', string.lower()).split() for string in raw_text_lines]
	
	single_tokens = []
	
	for sentence in cleaned_text_lines:
		single_tokens.extend(sentence)

	return single_tokens

def split_text_in_chunks(input_sequence: torch.tensor, length: int):
	
	chunks = torch.chunk(input_sequence, chunks=input_sequence.size(1) // length, dim=1)
	
	return chunks

# --- plots ---

def plot_loss(epochs: int, loss_histories: list, labels: list):
    plt.clf()
    
    line_styles = ['-', '--', ':']  # Adicione mais estilos se necessário
    
    for i in range(len(loss_histories)):
        plt.plot(np.arange(epochs), loss_histories[i], label=labels[i], linestyle=line_styles[i])

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.clf()

# --- Load Model ---

def save_model(model: any, epochs: str, architecture: str):

    try:
        if not os.path.exists('weights'):
            os.makedirs('weights')

        torch.save(model.state_dict(), f'weights/best_model_{architecture}_{epochs}.pth')

        return 'Model was successfully saved!'
    except:
        raise 'There was a problem saving the model!'

def save_parameters(dimension: int, word2idx: dict, idx2word: dict, vocab_size: int, model: str):
    
    params = {
            'word2idx': word2idx,
            'idx2word': idx2word,
            'dimension': dimension,
			'vocab_size': vocab_size,
			'model': model
    }

    with open(f'weights/parameters_{model}.json', 'w') as json_config:
        json.dump(params, json_config)

def load_parameters(config_path):
    
    params = {}

    with open(f'weights/{config_path}', 'r') as json_config:
        params = json.load(json_config)

    return params
