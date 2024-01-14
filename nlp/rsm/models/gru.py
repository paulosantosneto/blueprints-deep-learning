import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GRU(nn.Module):

	def __init__(self, in_d: int, out_d: int, vocab_size: int, bias=True):

		super(GRU, self).__init__()

		self.in_d = in_d
		self.vocab_size = vocab_size

		self.wz = nn.Parameter(torch.FloatTensor(vocab_size, in_d))
		self.uz = nn.Parameter(torch.FloatTensor(out_d, in_d))
		
		self.w = nn.Parameter(torch.FloatTensor(vocab_size, out_d))
		self.u = nn.Parameter(torch.FloatTensor(in_d, out_d))

		self.wr = nn.Parameter(torch.FloatTensor(vocab_size, in_d))
		self.ur = nn.Parameter(torch.FloatTensor(out_d, in_d))

		self.wy = nn.Parameter(torch.FloatTensor(vocab_size, out_d))

		# --- weight inicialization ---

		torch.nn.init.xavier_uniform_(self.w)
		torch.nn.init.xavier_uniform_(self.u)
		
		torch.nn.init.xavier_uniform_(self.wz)
		torch.nn.init.xavier_uniform_(self.uz)

		torch.nn.init.xavier_uniform_(self.wr)
		torch.nn.init.xavier_uniform_(self.ur)

		if bias:

			self.bz = nn.Parameter(torch.FloatTensor(1, out_d))
			torch.nn.init.xavier_uniform_(self.bz)
			
			self.br = nn.Parameter(torch.FloatTensor(1, out_d))
			torch.nn.init.xavier_uniform_(self.br)

			self.bh = nn.Parameter(torch.FloatTensor(1, out_d))
			torch.nn.init.xavier_uniform_(self.bh)

			self.by = nn.Parameter(torch.FloatTensor(1, vocab_size))
			torch.nn.init.xavier_uniform_(self.by)

	def forward(self, X):
		
		hiddens = [torch.zeros(1, self.in_d)]
		outs = []

		for t in range(X.size(1)):

			update_gate = torch.sigmoid(self.wz[X[:, t], :] + hiddens[-1].mm(self.uz.T) + self.bz)


			reset_gate = torch.sigmoid(self.wr[X[:, t], :] + hiddens[-1].mm(self.ur.T) + self.br)

			h_xh = torch.tanh(self.w[X[:, t], :] + (reset_gate * hiddens[-1]).mm(self.u.T) + self.bh)

			h = hiddens[-1] * update_gate + (1 - update_gate) * h_xh

			hiddens.append(h)

			y = h.mm(self.wy.T) + self.by

			outs.append(y)

		hiddens = torch.stack(hiddens[1:], dim=1)
		outs = torch.stack(outs, dim=1)
		outs = torch.log_softmax(outs, dim=2).squeeze(0)

		return outs, hiddens