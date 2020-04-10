# FinalProject - Video Action Classification

## Basic Execution
- **Platform:** Google Colab / Linux
- **Language:** Python3
- **Environment:** GPU
- **Usage:**
	- Implement the programs as follows in order.
	- ``python3 read_datasetBreakfast.py`` (need to modified the mode training or test)
	- ``CUDA_VISIBLE_DEVICES=<number> python3 Train.py -e <epoch_num> -d <train_data> -l <label_data> -m <model_name> --type <model_type>``
		- ``epoch_num``  - number of epoch
		- ``train_data`` - train.npy (produced by ``read_datasetBreakfast.py``)
		- ``label_data`` - label.npy (produced by ``read_datasetBreakfast.py``)
		- ``model_name`` - model\_<type>.pth.tar ex model\_dnn.pth.tar
		- ``type``       - dnn or lstm or gru
	- ``CUDA_VISIBLE_DEVICES=<number> python3 Test.py -t <test_data> -m <model_name> -o <output_name>``
		- ``test_data``  - test.npy (produced by ``read_datasetBreakfast.py``)
		- ``model_name`` - model\_<type>.pth.tar (produced by ``Train.py``)
		- ``type``       - dnn or lstm or gru
		- ``output_name``- prediction_<type>.csv ex prediction\_dnn.csv
	- ``python3 Emsemble.py``
		- need to modify the file (line 12 add the ensemble filename of csv)

- **Requirements:**
	- python 3.6
	- torch 1.2			``pip3 install torch``
	- torchvision 0.4	``pip3 install torchvision``
	- numpy 1.16			``pip3 install numpy``
	- pandas 0.24			``pip3 install pandas``
	- tqdm 4.43			``pip3 install tqdm``
- **Best Result:** Accuracy on kaggle - 0.44937

## Appendix
- **Files**
	- ``read_datasetBreakfast.py`` - provided by TA, modified a little.
	- ``Train.py`` - the main program for training; calculates loss as well as accuracy score on both training and validation datasets, and eventually saves the model with the best validation accuracy.
	- ``Test.py`` - the main program for testing; gets the result through the model saved by ``Train.py`` and eventually creates csv file for submission.
	- ``Parsing.py`` - parses the arguments for executing the code (have default values). Preprocessing.py - calculates the length of each segment in order to help us select a reasonable threshold for padding or cropping.
	- ``Loading.py`` - converts the numpy data into the wanted format.
	- ``Model.py`` - includes the model structures of deep neural network and recurrent neural network.
	- ``Ensemble`` - emsembles the results of different models.

- **Contribution**
	- Coding - *Yuting Tseng*
	- Executing and debugging - *Zhe Chen, Bihui Han, Xinyan, He*
	- Report - All teammates

## Reference
- *The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities*, H. Kuehne, A. B. Arslan & T. Serre. CVPR 2014
