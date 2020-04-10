import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class DNN(nn.Module):
	def __init__(self):
		super(DNN, self).__init__()
		self.line1 = nn.Sequential(
					 nn.Linear(398 * 400, 1024),
					 nn.ReLU(),
					 nn.BatchNorm1d(1024),
					 nn.Dropout(0.5),
					 )
		self.line2 = nn.Sequential(
					 nn.Linear(1024, 512),
					 nn.ReLU(),
					 nn.BatchNorm1d(512),
					 nn.Dropout(0.5),
					 )
		self.line3 = nn.Sequential(
					 nn.Linear(512, 256),
					 nn.ReLU(),
					 nn.BatchNorm1d(256),
					 nn.Dropout(0.5),
					 )
		self.line4 = nn.Sequential(
					 nn.Linear(256, 64),
					 nn.ReLU(),
					 nn.BatchNorm1d(64),
					 nn.Dropout(0.3),
					 )
		self.last  = nn.Linear(64, 48)

	def forward(self, trn):
		tmp = trn.reshape((trn.shape[0], -1))
		tmp = self.line1(tmp)
		tmp = self.line2(tmp)
		tmp = self.line3(tmp)
		tmp = self.line4(tmp)
		mod = self.last (tmp)
		return mod

class RNN(nn.Module):
	def __init__(self, mode1, mode2):
		super(RNN, self).__init__()
		self.mode1 = mode1
		self.mode2 = mode2

		# pad for the input sequence
		# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

		self.gru1  = nn.GRU (input_size=400, hidden_size=256, num_layers=1, batch_first=True)
		self.lstm1 = nn.LSTM(input_size=400, hidden_size=256, num_layers=1, batch_first=True)
		self.gru2  = nn.GRU (input_size=256, hidden_size=64, num_layers=1, batch_first=True)
		self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True)

		self.line1 = nn.Sequential(
					 nn.Linear(64 * 398, 4096),
					 nn.ReLU(),
					 nn.Dropout(0.5),
					 )

		self.line2 = nn.Sequential(
					 nn.Linear(4096, 1024),
					 nn.ReLU(),
					 nn.Dropout(0.3),
					 )

		self.line3 = nn.Sequential(
					 nn.Linear(1024, 256),
					 nn.ReLU(),
					 nn.Dropout(0.3),
					 )

		self.last  = nn.Sequential(
					 nn.Linear(256, 48),
					 nn.Softmax(dim=1),
					 )

	def forward(self, trn):
		tmp = None
		if self.mode1 == "gru" : tmp, _ = self.gru1 (trn)
		if self.mode1 == "lstm": tmp, _ = self.lstm1(trn)
		if self.mode2 == "gru" : tmp, _ = self.gru2 (tmp)
		if self.mode2 == "lstm": tmp, _ = self.lstm2(tmp)

		tmp = tmp.reshape((trn.shape[0], -1))
		tmp = self.line1(tmp)
		tmp = self.line2(tmp)
		tmp = self.line3(tmp)
		mod = self.last (tmp)
		return mod
