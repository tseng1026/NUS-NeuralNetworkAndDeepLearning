import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader

import Parsing
import Loading
import Model

if __name__ == "__main__":
	gpu = torch.cuda.is_available()

	# parse the arguments
	args = Parsing.Args()
	dataname = args.d
	lablname = args.l
	modlname = args.m
	epoch_num = int(args.e)
	nn_type = args.type

	# load the data and split into training and validation part
	train, valid = Loading.LoadTrn(dataname, lablname)
	train = Loading.DataSet(train, 0)
	valid = Loading.DataSet(valid, 0)

	train = DataLoader(train, batch_size=32, shuffle=True)
	valid = DataLoader(valid, batch_size=32, shuffle=False)
	print ("[Done] Loading all data!")

	# define loss function and optimizer
	model = None
	if nn_type == "lstm": model = Model.RNN("lstm", "lstm")
	if nn_type == "gru" : model = Model.RNN("gru",  "gru")
	if nn_type == "dnn" : model = Model.DNN()
	if os.path.isfile(modlname):
		state_dict = torch.load(modlname)
		model.load_state_dict(state_dict)
	if gpu: model = model.cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
	print ("[Done] Initializing model and all parameters!")

	train_best = -1.
	valid_best = -1.
	for epoch in range(epoch_num):
		print ("\n###### Epoch: {:d}".format(epoch + 1))

		# set to training mode
		model.train()

		train_loss = []
		train_scre = []
		for ind, (trn, lab) in enumerate(train):
			optimizer.zero_grad()

			# preprocess the sequence data
			if gpu: 
				trn = trn.cuda()
				lab = lab.cuda()
			out = model(trn)

			# compute the loss value
			loss = criterion(out, lab.squeeze())
			loss.backward()
			train_loss.append(loss.item())

			# compute the accuracy
			pred = torch.max(out, dim=1)[1]
			scre = np.mean((pred == lab.squeeze()).cpu().numpy())
			train_scre.append(scre)

			nn.utils.clip_grad_norm_(model.parameters(), 1)
			optimizer.step()

		print ("[Done] Computing train loss: {:.4f}".format(np.mean(train_loss)))
		print ("[Done] Computing train scre: {:.4f}".format(np.mean(train_scre)))

		# set to training mode
		model.eval()

		valid_loss = []
		valid_scre = []
		for ind, (val, lab) in enumerate(valid):

			# preprocess the sequence data
			if gpu: 
				val = val.cuda()
				lab = lab.cuda()
			out = model(val)

			# compute the loss value
			loss = criterion(out, lab.squeeze())
			valid_loss.append(loss.item())

			# compute the accuracy
			pred = torch.max(out, dim=1)[1]
			scre = np.mean((pred == lab.squeeze()).cpu().numpy())
			valid_scre.append(scre)

		print("[Done] Computing valid loss: {:.4f}".format(np.mean(valid_loss)))
		print("[Done] Computing valid scre: {:.4f}".format(np.mean(valid_scre)))

		# update the best model
		train_temp = np.mean(train_scre)
		valid_temp = np.mean(valid_scre)
		if valid_best <= valid_temp:
			train_best = train_temp
			valid_best = valid_temp
			torch.save(model.state_dict(), modlname)