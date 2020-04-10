import numpy as np
import torch
from   torch.utils.data import Dataset
from   tqdm import tqdm

def LoadTrn(trnDir, labDir):
	lab  = np.load(labDir, allow_pickle=True)
	trn  = np.load(trnDir, allow_pickle=True)
	
	# Vectorize the training data sequence
	maxlen = 398
	trnTmp = []
	labTmp = []

	fp = open("training_segment.txt", "r")
	content = fp.readlines()
	for k, data in tqdm(enumerate(content), "Vectorization"):
		# read the segment
		segment = [int(frame) for frame in data.split()]

		# crop the segment (train and label)
		for frame in range(len(segment) - 1):
			length = segment[frame+1] - segment[frame]
			trnTmp.append(trn[k][segment[frame]:segment[frame+1]].tolist())
			labTmp.append(lab[k][segment[frame]])

			if length < maxlen: trnTmp[-1] = trnTmp[-1] + [[0] * 400] * (maxlen - length)
			if length > maxlen: trnTmp[-1] = trnTmp[-1][:maxlen]

	trn = np.array(trnTmp)
	lab = np.array(labTmp)
	np.save("train_segment.npy", trn)
	np.save("label_segment.npy", lab)
	print ("[Done] Saving train.npy and label.npy after padding!")

	import random
	data = list(zip(trn, lab.reshape(-1, 1)))
	random.shuffle(data)

	length = len(data)
	train  = data[:length * 4 // 5]
	valid  = data[ length * 4 // 5:]
	return train, valid

def LoadTst(tstDir):
	tst  = np.load(tstDir, allow_pickle=True)

	maxlen = 398
	tstTmp = []

	tst = tst.tolist()
	fp = open("test_segment.txt", "r")
	content = fp.readlines()[:3]
	for k, data in tqdm(enumerate(content), "Vectorization"):
		# read the segment
		segment = [int(frame) for frame in data.split()]

		# crop the segment (train and label)
		for frame in range(len(segment) - 1):
			length = segment[frame+1] - segment[frame]
			tstTmp.append(tst[k][segment[frame]:segment[frame+1]].tolist())

			if length < maxlen: tstTmp[-1] = tstTmp[-1] + [[0] * 400] * (maxlen - length)
			if length > maxlen: tstTmp[-1] = tstTmp[-1][:maxlen]

	tst = np.array(tstTmp)
	np.save("test_segment.npy", tst)
	print ("[Done] Saving test.npy after padding!")

	return tst

class DataSet(Dataset):
	def __init__(self, data, train_or_test):
		self.data          = data
		self.train_or_test = train_or_test

	def __len__(self):
		return len(self.data)

	def __getitem__(self, ind):
		if self.train_or_test == 0:
			trn = torch.from_numpy(self.data[ind][0]).type(torch.FloatTensor)
			lab = torch.from_numpy(self.data[ind][1]).type(torch.LongTensor)
			return trn, lab

		if self.train_or_test == 1:
			tst = torch.from_numpy(self.data[ind]).type(torch.FloatTensor)
			return tst
