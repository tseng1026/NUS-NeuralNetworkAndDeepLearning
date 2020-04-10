import argparse

def Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", default="./train.npy")
	parser.add_argument("-l", default="./label.npy")
	parser.add_argument("-t", default="./test.npy")
	parser.add_argument("-m", default="./model_best.pth.tar")
	parser.add_argument("-o", default="./prediction.csv")
	parser.add_argument("-e", default="200")
	parser.add_argument("--type", default="lstm")

	args = parser.parse_args()
	return args
