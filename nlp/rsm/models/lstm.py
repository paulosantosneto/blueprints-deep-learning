import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LSTM(nn.Module):

	def __init__(self, in_d: int, out_d: int, vocab_size: int, bias=True):

		super(LSTM, self).__init__()

		self.in_d = in_d
		self.vocab_size = vocab_size

		# forget gate

		self.weight_xf = nn.Parameter(torch.FloatTensor(vocab_size, out_d))
		self.weight_hf = nn.Parameter(torch.FloatTensor(in_d, out_d))
		
		# input gate

		self.weight_xi = nn.Parameter(torch.FloatTensor(vocab_size, out_d))
		self.weight_hi = nn.Parameter(torch.FloatTensor(in_d, out_d))

		# candidate gate

		self.weight_xc = nn.Parameter(torch.FloatTensor(vocab_size, out_d))
		self.weight_hc = nn.Parameter(torch.FloatTensor(in_d, out_d))

		# output gate

		self.weight_xo = nn.Parameter(torch.FloatTensor(vocab_size, out_d))
		self.weight_ho = nn.Parameter(torch.FloatTensor(in_d, out_d))

		self.linear_weight = nn.Parameter(torch.FloatTensor(vocab_size, out_d))

		# --- weight inicialization ---

		torch.nn.init.xavier_uniform_(self.weight_xf)
		torch.nn.init.xavier_uniform_(self.weight_hf)

		torch.nn.init.xavier_uniform_(self.weight_xi)
		torch.nn.init.xavier_uniform_(self.weight_hi)

		torch.nn.init.xavier_uniform_(self.weight_xc)
		torch.nn.init.xavier_uniform_(self.weight_hc)

		torch.nn.init.xavier_uniform_(self.weight_xo)
		torch.nn.init.xavier_uniform_(self.weight_ho)

		torch.nn.init.xavier_uniform_(self.linear_weight)

		if bias:
			self.b_f = nn.Parameter(torch.FloatTensor(out_d, 1))
			torch.nn.init.xavier_uniform_(self.b_f)

			self.b_i = nn.Parameter(torch.FloatTensor(out_d, 1))
			torch.nn.init.xavier_uniform_(self.b_i)

			self.b_c = nn.Parameter(torch.FloatTensor(out_d, 1))
			torch.nn.init.xavier_uniform_(self.b_c)

			self.b_o = nn.Parameter(torch.FloatTensor(out_d, 1))
			torch.nn.init.xavier_uniform_(self.b_o)

			self.linear_bias = nn.Parameter(torch.FloatTensor(1, vocab_size))
			torch.nn.init.xavier_uniform_(self.linear_bias)

	def forward(self, X):
		
		cell_states = [torch.zeros(1, self.in_d)]
		hidden_states = [torch.zeros(1, self.in_d)]
		outs = []

		for t in range(X.size(1)):
			x_t = torch.eye(self.vocab_size)[X[:, t]]

			forget_gate = torch.sigmoid(torch.matmul(x_t, self.weight_xf) + torch.matmul(hidden_states[-1], self.weight_hf) + self.b_f.T)
			input_gate = torch.sigmoid(torch.matmul(x_t, self.weight_xi) + torch.matmul(hidden_states[-1], self.weight_hi) + self.b_i.T)
			candidate_gate = torch.tanh(torch.matmul(x_t, self.weight_xc) + torch.matmul(hidden_states[-1], self.weight_hc) + self.b_c.T)
			output_gate = torch.sigmoid(torch.matmul(x_t, self.weight_xo) + torch.matmul(hidden_states[-1], self.weight_ho) + self.b_o.T)

			c_t = forget_gate * cell_states[-1] + input_gate * candidate_gate
			h_t = torch.tanh(c_t) * output_gate

			cell_states.append(c_t)
			hidden_states.append(h_t)

			y_t = torch.matmul(self.linear_weight, h_t.T).T + self.linear_bias

			outs.append(y_t)

		hidden_states = torch.stack(hidden_states[1:], dim=1)
		outs = torch.stack(outs, dim=1)
		outs = torch.log_softmax(outs, dim=2).squeeze(0)

		return outs, hidden_states