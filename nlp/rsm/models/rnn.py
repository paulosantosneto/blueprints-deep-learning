import torch
import torch.nn as nn
import numpy as np
import math

class RNN(nn.Module):
	def __init__(self, in_d: int, out_d: int, vocab_size: int, bias=True):
		super(RNN, self).__init__()
		
		self.start_length_hidden_state = in_d
		self.vocab_size = vocab_size
		self.weight_hh = nn.Parameter(torch.FloatTensor(in_d, out_d)) 
		self.weight_xh = nn.Parameter(torch.FloatTensor(vocab_size, in_d)) 
		self.weight_vh = nn.Parameter(torch.FloatTensor(vocab_size, out_d)) 
		
		# --- weight inicialization ---
		
		std = 1. / (math.sqrt(self.weight_vh.size(1)))
		self.weight_hh.data.uniform_(-std, std)
		self.weight_xh.data.uniform_(-std, std)
		self.weight_vh.data.uniform_(-std, std)
		
		if bias:
			self.b = nn.Parameter(torch.FloatTensor(out_d))
			self.b.data.uniform_(-std, std)
			self.c = nn.Parameter(torch.FloatTensor(vocab_size))
			self.c.data.uniform_(-std, std)
	
	def forward(self, X):
		
		start_hidden_state = torch.zeros(self.start_length_hidden_state, 1)
		
		hiddens = [start_hidden_state]
		outs = []

		for t in range(X.size(1)):
			x_t = torch.eye(self.vocab_size)[X[:, t]]

			h_t = torch.tanh(torch.matmul(self.weight_xh.T, x_t.T).T + torch.matmul(self.weight_hh, hiddens[-1]).T + self.b)

			z_t = torch.matmul(self.weight_vh, h_t.T).T + self.c

			outs.append(z_t)
			hiddens.append(h_t.T)
		
		hiddens = torch.stack(hiddens[1:], dim=1)
		outs = torch.stack(outs, dim=1)
		outs = torch.log_softmax(outs, dim=2).squeeze(0)

		return outs, hiddens