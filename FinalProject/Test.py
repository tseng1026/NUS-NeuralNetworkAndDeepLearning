import numpy as np
import pandas as pd
import torch
from   torch.utils.data import DataLoader

import Parsing
import Loading
import Model

if __name__ == "__main__":
	gpu = torch.cuda.is_available()

	# parsing the arguments
	args = Parsing.Args()
	testname = args.t
	modlname = args.m
	outputfile = args.o
	nn = args.type

	# load the data
	test = Loading.LoadTst(testname)
	numb = len(test)

	test = Loading.DataSet(test, 1)
	test = DataLoader(test, batch_size=64, shuffle=False)
	print ("[Done] Segmenting and vectorizing all data!")

	# load done-trained model
	model = None
	if nn == "lstm": model = Model.RNN("lstm", "lstm")
	if nn == "gru" : model = Model.RNN("gru",  "gru")
	if nn == "dnn" : model = Model.DNN()
	check = torch.load(modlname)
	model.load_state_dict(check)
	if gpu: model.cuda()
	print ("[Done] Initializing all model!")

	# set to evaluation mode
	model.eval()

	predt = torch.LongTensor()
	if gpu:
		predt = predt.cuda()

	for ind, tst in enumerate(test):
		# preprocess the sequence data
		if gpu: 
			tst = tst.cuda()
		out = model(tst)
		
		# compute the accuracy
		pred = torch.max(out, dim=1)[1]
		predt = torch.cat((predt, pred))

	# write the results to file
	index = np.arange(numb)
	index = index.astype("int")

	predict = predt.type(torch.FloatTensor).cpu().numpy().squeeze()
	predict = predict.astype("int")

	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(outputfile, header = ["id", "label"], index = None)