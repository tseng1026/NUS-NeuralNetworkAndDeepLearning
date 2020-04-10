import numpy as np
import pandas as pd

import Parsing

if __name__ == "__main__":
	# parse the arguments
	args = Parsing.Args()
	outputfile = args.o

	# sum of the results
	filename = ["prediction.csv"]
	total = np.zeros((1284, ))
	for name in filename:
		pred = pd.read_csv(filename)
		pred = pred[["label"]].label.tolist()
		pred = np.array(pred)

		total += pred

	# write the results to file
	index = np.arange(1284)
	index = index.astype("int")

	predict = total // (len(filename) + 1 // 2)
	predict = predict.astype("int")

	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(outputfile, header = ["id", "label"], index = None)