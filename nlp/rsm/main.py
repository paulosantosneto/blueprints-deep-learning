import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import random

from utils import *
from models import *

def generate_text(model: any, start_sequence: list, length: int, word2idx: dict, idx2word: dict, temperature: float):

	with torch.no_grad():

		current_sequence = start_sequence[:]
		generated_text = ' '.join(start_sequence)

		for i in range(length):
			
			current_sequence_idx = []

			for word in current_sequence:
				if word not in word2idx.keys():
					word = 'unknow'
				current_sequence_idx.append(word2idx[word])

			input_tensor = torch.tensor(current_sequence_idx).unsqueeze(0)

			logits, _ = model(input_tensor)

			scaled_logits = logits[-1, :] / min(0.1, temperature)
			probs = torch.softmax(scaled_logits, dim=-1)

			sampled_idx = torch.multinomial(probs, 1).item()
	
			next_word = idx2word[str(sampled_idx)]
			
			generated_text += ' ' + next_word
			
			current_sequence.append(next_word)
			
		print(f'{generated_text}\n')

def train_model(args: dict, model: torch.nn.Sequential, loss_fn: any, optimizer: any, input_seq: list, target_seq: list):
	
	loss_history = []
		
	for epoch in range(args.epochs):
		
		local_loss = 0

		combined_data = list(zip(input_seq, target_seq))
		random.shuffle(combined_data)
		input_sample, target_sample = zip(*combined_data)
		
		for batch, sequence in enumerate(input_sample):

			logits, _ = model(sequence)
			
			local_loss += loss_fn(logits, target_sample[batch].squeeze(0))
		
		if args.verbose:
			print(f'Epochs: {epoch} | Loss: {local_loss.item():.5f}')

		loss_history.append(local_loss.item())

		optimizer.zero_grad()
		local_loss.backward()
		optimizer.step()
	
	return loss_history

			
if __name__ == '__main__':
    
	args = get_args()

	torch.manual_seed(args.seed)
	
	if args.mode == 'train':
		
		# --- Preprocessing Text ---

		texts = load_doc(args.path_file)
		preprocessed_text = preprocess_text(texts)
		
		preprocessed_text.append('unknow')

		vocab = list(set(preprocessed_text))
		
		word2idx = {word:idx for idx, word in enumerate(vocab)}
		idx2word = {idx:word for idx, word in enumerate(vocab)}

		print(word2idx['unknow'])
			
		if args.verbose:

			print('\n-------- Informations --------')
			print(f'\nLen Vocabulary (unique words): {len(vocab)}')
			print(f'Number of words: {len(preprocessed_text)}')
			print('\n-------- Epochs --------\n')

		indexed_data = [word2idx[word] for word in preprocessed_text]

		# --- Training Model ---
		
		for i in range(len(args.sequence_model)):
			# choose the model
			if args.sequence_model[i] == 'rnn':

				model = RNN(in_d=args.dimension, out_d=args.dimension, vocab_size=len(vocab))
			elif args.sequence_model[i] == 'lstm':
				# TODO	
				model = LSTM(in_d=args.dimension, out_d=args.dimension, vocab_size=len(vocab))
			else:
				# TODO
				pass

			loss_fn = nn.CrossEntropyLoss()
			optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
			
			input_sequence = torch.tensor(indexed_data[:-1], dtype=torch.long).unsqueeze(0)
			target_sequence = torch.tensor(indexed_data[1:], dtype=torch.long).unsqueeze(0)
			
			input_sequence_chunks = split_text_in_chunks(input_sequence, args.sequence_length)
			target_sequence_chunks = split_text_in_chunks(target_sequence, args.sequence_length)
			
			loss = train_model(args, model, loss_fn, optimizer, input_sequence_chunks, target_sequence_chunks)
			
			if args.plot_loss:
				
				plot_loss(args.epochs, loss)

			if args.save_model:

				save_model(model, str(args.epochs), args.sequence_model[i])
				save_parameters(args.dimension, word2idx, idx2word, len(vocab), args.sequence_model[i])

	elif args.mode == 'inference':

		for i in range(len(args.model_config_path)):

			print(f'\n-------- {args.sequence_model[i].upper()} Text --------\n')

			params = load_parameters(args.model_config_path[i])

			if params['model'] == 'rnn':

				model = RNN(in_d=params['dimension'], out_d=params['dimension'], vocab_size=params['vocab_size'])
			elif params['model']== 'lstm':
					
				model = LSTM(in_d=params['dimension'], out_d=params['dimension'], vocab_size=params['vocab_size'])
				
			model.load_state_dict(torch.load(f'./weights/{args.model_load_path[i]}'))
			
			generate_text(model, args.start_sequence, args.max_length, params['word2idx'], params['idx2word'], args.temperature)
