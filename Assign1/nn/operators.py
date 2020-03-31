import numpy as np

from utils.tools import *
from nn.functional import sigmoid, img2col
# Attention:
# - Never change the value of input, which will change the result of backward

def My_get(x_shape, kernel_h, kernel_w, padding = 1, stride = 1):
	pad = padding
	batch, in_channel, in_height, in_width = x_shape
	out_height = 1 + (in_height + 2 * pad - kernel_h) // stride
	out_width  = 1 + (in_width  + 2 * pad - kernel_w) // stride

	i_prev = np.tile(np.arange(kernel_h), (kernel_w, in_channel)).T
	j_prev = np.tile(np.arange(kernel_w), (kernel_h, in_channel))
	i_prev = i_prev.reshape(in_channel * kernel_h * kernel_w)
	j_prev = j_prev.reshape(in_channel * kernel_h * kernel_w)

	i_next = stride * np.tile(np.arange(out_height), (out_width,  1)).T
	j_next = stride * np.tile(np.arange(out_width),  (out_height, 1))
	i_next = i_next.reshape(out_height * out_width)
	j_next = j_next.reshape(out_height * out_width)
	
	i = i_prev.reshape(-1, 1) + i_next.reshape(1, -1)
	j = j_prev.reshape(-1, 1) + j_next.reshape(1, -1)
	k = np.repeat(np.arange(in_channel), kernel_h * kernel_w).reshape(-1, 1)

	return (k.astype(int), i.astype(int), j.astype(int))

def My_img2col(x, x_shape, kernel_h, kernel_w, padding = 1, stride=1):
	batch, in_channel, in_height, in_width = x_shape

	pad = padding
	x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode = "constant")

	k, i, j = My_get(x_shape, kernel_h, kernel_w, padding, stride)

	col = x_pad[:, k, i, j]
	col = col.transpose(1, 2, 0)
	col = col.reshape(in_channel * kernel_h * kernel_w, -1)
	return col

def My_col2img(col, x_shape, kernel_h, kernel_w, padding = 1, stride = 1):
	batch, in_channel, in_height, in_width = x_shape

	pad = padding
	x_pad = np.zeros((batch, in_channel, in_height + 2 * pad, in_width  + 2 * pad), dtype=col.dtype)
	
	k, i, j = My_get(x_shape, kernel_h, kernel_w, pad, stride)

	col = col.reshape(in_channel * kernel_h * kernel_w, -1, batch)
	col = col.transpose(2, 0, 1)
	np.add.at(x_pad, (slice(None), k, i, j), col)

	if pad == 0: return x_pad
	return x_pad[:, :, pad:-pad, pad:-pad]

class operator(object):
	"""
	operator abstraction
	"""

	def forward(self, input):
		"""Forward operation, reture output"""
		raise NotImplementedError

	def backward(self, out_grad, input):
		"""Backward operation, return gradient to input"""
		raise NotImplementedError


class relu(operator):
	def __init__(self):
		super(relu, self).__init__()

	def forward(self, input):
		output = np.maximum(0, input)
		return output

	def backward(self, out_grad, input):
		in_grad = (input >= 0) * out_grad
		return in_grad


class flatten(operator):
	def __init__(self):
		super(flatten, self).__init__()

	def forward(self, input):
		batch  = input.shape[0]
		output = input.copy().reshape(batch, -1)
		return output

	def backward(self, out_grad, input):
		in_grad = out_grad.copy().reshape(input.shape)
		return in_grad


class matmul(operator):
	def __init__(self):
		super(matmul, self).__init__()

	def forward(self, input, weights):
		"""
		# Arguments
			input:   numpy array with shape (batch, in_features)
			weights: numpy array with shape (in_features, out_features)

		# Returns
			output: numpy array with shape(batch, out_features)
		"""
		return np.matmul(input, weights)

	def backward(self, out_grad, input, weights):
		"""
		# Arguments
			out_grad: gradient to the forward output of linear layer, with shape (batch, out_features)
			input:    numpy array with shape (batch, in_features)
			weights:  numpy array with shape (in_features, out_features)

		# Returns
			in_grad: gradient to the forward input with same shape as input
			w_grad:  gradient to weights, with same shape as weights            
		"""
		in_grad = np.matmul(out_grad, weights.T)
		w_grad  = np.matmul(input.T, out_grad)
		return in_grad, w_grad


class add_bias(operator):
	def __init__(self):
		super(add_bias, self).__init__()

	def forward(self, input, bias):
		"""
		# Arugments
		  input: numpy array with shape (batch, in_features)
		  bias:  numpy array with shape (in_features)

		# Returns
		  output: numpy array with shape(batch, in_features)
		"""
		return input + bias.reshape(1, -1)

	def backward(self, out_grad, input, bias):
		"""
		# Arguments
			out_grad: gradient to the forward output of linear layer, with shape (batch, out_features)
			input:    numpy array with shape (batch, in_features)
			bias:     numpy array with shape (out_features)
		
		# Returns
			in_grad: gradient to the forward input with same shape as input
			b_bias:  gradient to bias, with same shape as bias
		"""
		in_grad = out_grad
		b_grad  = np.sum(out_grad, axis=0)
		return in_grad, b_grad


class linear(operator):
	def __init__(self):
		super(linear, self).__init__()
		self.matmul   = matmul()
		self.add_bias = add_bias()

	def forward(self, input, weights, bias):
		"""
		# Arguments
			input:   numpy array with shape (batch, in_features)
			weights: numpy array with shape (in_features, out_features)
			bias:    numpy array with shape (out_features)

		# Returns
			output: numpy array with shape(batch, out_features)
		"""
		output = self.matmul.  forward(input, weights)
		output = self.add_bias.forward(output, bias)
		# output = np.matmul(input, weights) + bias.reshape(1, -1)
		# print ("linear", input.shape, output.shape)
		return output

	def backward(self, out_grad, input, weights, bias):
		"""
		# Arguments
			out_grad: gradient to the forward output of linear layer, with shape (batch, out_features)
			input:    numpy array with shape (batch, in_features)
			weights:  numpy array with shape (in_features, out_features)
			bias:     numpy array with shape (out_features)

		# Returns
			in_grad: gradient to the forward input of linear layer, with same shape as input
			w_grad:  gradient to weights, with same shape as weights
			b_bias:  gradient to bias, with same shape as bias
		"""
		# in_grad = np.matmul(out_grad, weights.T)
		# w_grad = np.matmul(input.T, out_grad)
		# b_grad = np.sum(out_grad, axis=0)
		out_grad, b_grad = self.add_bias.backward(out_grad, input, bias)
		in_grad,  w_grad = self.matmul.  backward(out_grad, input, weights)
		return in_grad, w_grad, b_grad


class conv(operator):
	def __init__(self, conv_params):
		"""
		# Arguments
			conv_params: dictionary, containing these parameters:
				"kernel_h":    The height of kernel.
				"kernel_w":    The width of kernel.
				"stride":      The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
				"pad":         The total number of 0s to be added along the height (or width) dimension; half of the 0s are added on the top (or left) and half at the bottom (or right). we will only test even numbers.
				"in_channel":  The number of input channels.
				"out_channel": The number of output channels.
		"""
		super(conv, self).__init__()
		self.conv_params = conv_params

	def forward(self, input, weights, bias):
		"""
		# Arguments
			input:   numpy array with shape (batch, in_channel, in_height, in_width)
			weights: numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
			bias:    numpy array with shape (out_channel)

		# Returns
			output:  numpy array with shape (batch, out_channel, out_height, out_width)
		"""
		kernel_h    = self.conv_params["kernel_h"]  # height of kernel
		kernel_w    = self.conv_params["kernel_w"]  # width of kernel
		pad         = self.conv_params["pad"]
		stride      = self.conv_params["stride"]
		in_channel  = self.conv_params["in_channel"]
		out_channel = self.conv_params["out_channel"]
		
		# code here	
		#####################################################################################
		# Compute basic input information with padding
		pad = pad // 2
		batch, in_channel, in_height, in_width = input.shape

		out_height = 1 + (in_height + 2 * pad - kernel_h) // stride
		out_width  = 1 + (in_width  + 2 * pad - kernel_w) // stride
		output = np.zeros((batch, out_channel, out_height, out_width))

		# Compute the matrix via My_img2col
		input_col   = My_img2col(input, input.shape, kernel_h, kernel_w, padding = pad, stride = stride)
		weights_col = weights.reshape(out_channel, -1)

		# Compute the output results
		output = np.matmul(weights_col, input_col) + bias.reshape(out_channel, 1)
		output = output.reshape(out_channel, out_height, out_width, batch)
		output = output.transpose(3, 0, 1, 2)
		#####################################################################################
		# print ("conv", input.shape, output.shape)
		return output

	def backward(self, out_grad, input, weights, bias):
		"""
		# Arguments
			out_grad: gradient to the forward output of conv layer, with shape (batch, out_channel, out_height, out_width)
			input:    numpy array with shape (batch, in_channel, in_height, in_width)
			weights:  numpy array with shape (out_channel, in_channel, kernel_h, kernel_w)
			bias:     numpy array with shape (out_channel)

		# Returns
			in_grad: gradient to the forward input of conv layer, with same shape as input
			w_grad:  gradient to weights, with same shape as weights
			b_bias:  gradient to bias, with same shape as bias
		"""
		kernel_h    = self.conv_params["kernel_h"]  # height of kernel
		kernel_w    = self.conv_params["kernel_w"]  # width of kernel
		pad         = self.conv_params["pad"]
		stride      = self.conv_params["stride"]
		in_channel  = self.conv_params["in_channel"]
		out_channel = self.conv_params["out_channel"]

		# code here
		#################################################################################
		# Compute basic information
		pad = pad // 2
		batch,  in_channel,  in_height,  in_width = input.shape
		batch, out_channel, out_height, out_width = out_grad.shape

		b_grad  = np.zeros_like(bias)
		in_grad = np.zeros_like(input)
		w_grad  = np.zeros_like(weights)

		# Compute the matrix via My_img2col
		input_col   = My_img2col(input, input.shape, kernel_h, kernel_w, padding = pad, stride = stride)
		output_col  = out_grad.transpose(1, 2, 3, 0).reshape(out_channel, -1)
		weights_col = weights.reshape(out_channel, -1)

		# Compute gradient of bias
		b_grad = np.sum(out_grad, axis = (0, 2, 3))

		# Compute gradient of weight
		w_grad = np.matmul(output_col, np.transpose(input_col))
		w_grad = w_grad.reshape(out_channel, in_channel, kernel_h, kernel_w)

		# Compute gradient of input
		in_grad = np.matmul(np.transpose(weights_col), output_col)
		in_grad = My_col2img(in_grad, input.shape, kernel_h, kernel_w, padding = pad, stride = stride)
		#################################################################################
		return in_grad, w_grad, b_grad

# pool_backward QQQ
class pool(operator):
	def __init__(self, pool_params):
		"""
		# Arguments
			pool_params: dictionary, containing these parameters:
				"pool_type":   The type of pooling, "max" or "avg"
				"pool_height": The height of pooling kernel.
				"pool_width":  The width of pooling kernel.
				"stride":      The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
				"pad":         The total number of 0s to be added along the height (or width) dimension; half of the 0s are added on the top (or left) and half at the bottom (or right). we will only test even numbers.
		"""
		super(pool, self).__init__()
		self.pool_params = pool_params

	def forward(self, input):
		"""
		# Arguments
			input: numpy array with shape (batch, in_channel, in_height, in_width)

		# Returns
			output: numpy array with shape (batch, in_channel, out_height, out_width)
		"""
		pool_type   = self.pool_params["pool_type"]
		pool_height = self.pool_params["pool_height"]
		pool_width  = self.pool_params["pool_width"]
		stride      = self.pool_params["stride"]
		pad         = self.pool_params["pad"]

		# code here
		#####################################################################################
		# Compute basic information
		pad = pad // 2
		batch, in_channel, in_height, in_width = input.shape

		out_height = 1 + (in_height + 2 * pad - pool_height) // stride
		out_width  = 1 + (in_width  + 2 * pad - pool_width ) // stride
		output = np.zeros((batch, in_channel, out_height, out_width))

		# Compute the matrix via My_img2col
		input_tmp = input.reshape(batch * in_channel, 1, in_height, in_width)
		input_col = My_img2col(input_tmp, input_tmp.shape, pool_height, pool_width, padding = pad, stride = stride)

		# Compute the output results for maxpooling
		if pool_type == "max":
			output = np.argmax(input_col, axis = 0)
			output = input_col[output, range(output.size)]
			output = output.reshape(out_height, out_width, batch, in_channel)
			output = output.transpose(2, 3, 0, 1)
		
		# Compute the output results for avgpooling
		elif pool_type == "avg":
			output = np.mean(input_col, axis = 0)
			output = output.reshape(out_height, out_width, batch, in_channel)
			output = output.transpose(2, 3, 0, 1)
		#####################################################################################
		# print ("pool", input.shape, output.shape)
		return output

	def backward(self, out_grad, input):
		"""
		# Arguments
			out_grad: gradient to the forward output of conv layer, with shape (batch, in_channel, out_height, out_width)
			input:    numpy array with shape (batch, in_channel, in_height, in_width)

		# Returns
			in_grad: gradient to the forward input of pool layer, with same shape as input
		"""
		pool_type   = self.pool_params["pool_type"]
		pool_height = self.pool_params["pool_height"]
		pool_width  = self.pool_params["pool_width"]
		stride      = self.pool_params["stride"]
		pad         = self.pool_params["pad"]

		batch, in_channel, in_height, in_width = input.shape
		out_height = 1 + (in_height - pool_height + pad) // stride
		out_width  = 1 + (in_width  - pool_width  + pad) // stride

		pad_scheme = (pad // 2, pad - pad // 2)
		input_pad = np.pad(input, pad_width=((0,0), (0,0), pad_scheme, pad_scheme),
						   mode="constant", constant_values=0)

		recep_fields_h = [stride*i for i in range(out_height)]
		recep_fields_w = [stride*i for i in range(out_width)]

		input_pool = img2col(input_pad, recep_fields_h,
							 recep_fields_w, pool_height, pool_width)
		input_pool = input_pool.reshape(
			batch, in_channel, -1, out_height, out_width)

		if pool_type == "max":
			input_pool_grad = (input_pool == np.max(input_pool, axis=2, keepdims=True)) * \
				out_grad[:, :, np.newaxis, :, :]

		elif pool_type == "avg":
			scale = 1 / (pool_height*pool_width)
			input_pool_grad = scale * \
				np.repeat(out_grad[:, :, np.newaxis, :, :],
						  pool_height*pool_width, axis=2)

		input_pool_grad = input_pool_grad.reshape(
			batch, in_channel, -1, out_height*out_width)

		input_pad_grad = np.zeros(input_pad.shape)
		idx = 0
		for i in recep_fields_h:
			for j in recep_fields_w:
				input_pad_grad[:, :, i:i+pool_height, j:j+pool_width] += \
					input_pool_grad[:, :, :, idx].reshape(
						batch, in_channel, pool_height, pool_width)
				idx += 1
		in_grad = input_pad_grad[:, :, pad:pad+in_height, pad:pad+in_width]
		return in_grad


class dropout(operator):
	def __init__(self, rate, training=True, seed=None):
		"""
		# Arguments
			rate:     float[0, 1], the probability of setting a neuron to zero
			training: boolean, apply this layer for training or not. If for training, randomly drop neurons, else DO NOT drop any neurons
			seed:     int, random seed to sample from input, so as to get mask, which is convenient to check gradients. But for real training, it should be None to make sure to randomly drop neurons
			mask:     the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input
		"""
		self.rate     = rate
		self.seed     = seed
		self.training = training
		self.mask     = None

	def forward(self, input):
		"""
		# Arguments
			input: numpy array with any shape

		# Returns
			output: same shape as input
		"""
		output = None
		if self.training:
			# Create the mask randomly
			np.random.seed(self.seed)
			rate = np.random.random_sample(input.shape)
			self.mask = (rate >= self.rate).astype("int")

			scle = 1 / (1 - self.rate)
			# code here
			#####################################################################################
			output = np.multiply(input, self.mask) * scle
			#####################################################################################
		else:
			output = input
		
		# print ("dropout", input.shape, output.shape)
		return output

	def backward(self, out_grad, input):
		"""
		# Arguments
			out_grad: gradient to forward output of dropout, same shape as input
			input:    numpy array with any shape
			mask:     the mask with value 0 or 1, corresponding to drop neurons (0) or not (1). same shape as input

		# Returns
			in_grad: gradient to forward input of dropout, same shape as input
		"""
		in_grad = None	
		if self.training:
			# code here
			#####################################################################################
			scle = 1 / (1 - self.rate)
			in_grad = np.multiply(out_grad * scle, self.mask)
			#####################################################################################
		else:
			in_grad = out_grad
		return in_grad


class vanilla_rnn(operator):
	def __init__(self):
		"""
		# Arguments
			in_features: int, the number of inputs features
			units:       int, the number of hidden units
			initializer: Initializer class, to initialize weights
		"""
		super(vanilla_rnn, self).__init__()

	def forward(self, input, kernel, recurrent_kernel, bias):
		"""
		# Arguments
			inputs: [input numpy array with shape (batch, in_features), 
					state numpy array with shape (batch, units)]

		# Returns
			outputs: numpy array with shape (batch, units)
		"""
		x, prev_h = input
		output = np.tanh(x.dot(kernel) + prev_h.dot(recurrent_kernel) + bias)
		return output

	def backward(self, out_grad, input, kernel, recurrent_kernel, bias):
		"""
		# Arguments
			in_grads: numpy array with shape (batch, units), gradients to outputs
			inputs:   [input numpy array with shape (batch, in_features), 
					   state numpy array with shape (batch, units)], same with forward inputs

		# Returns
			out_grads: [gradients to input numpy array with shape (batch, in_features), 
						gradients to state numpy array with shape (batch, units)]
		"""
		x, prev_h = input
		tanh_grad = np.nan_to_num(
			out_grad*(1-np.square(self.forward(input, kernel, recurrent_kernel, bias))))

		in_grad = [np.matmul(tanh_grad, kernel.T), np.matmul(
			tanh_grad, recurrent_kernel.T)]
		kernel_grad = np.matmul(np.nan_to_num(x.T), tanh_grad)
		r_kernel_grad = np.matmul(np.nan_to_num(prev_h.T), tanh_grad)
		b_grad = np.sum(tanh_grad, axis=0)

		return in_grad, kernel_grad, r_kernel_grad, b_grad


class gru(operator):
	def __init__(self):
		"""
		# Arguments
			in_features: int, the number of inputs features
			units:       int, the number of hidden units
			initializer: Initializer class, to initialize weights
		"""
		super(gru, self).__init__()

	def forward(self, input, kernel, recurrent_kernel):
		"""
		# Arguments
			inputs: [input numpy array with shape (batch, in_features), 
					 state numpy array with shape (batch, units)]

		# Returns
			outputs: numpy array with shape (batch, units)
		"""
		x, prev_h = input
		_, all_units = kernel.shape
		units = all_units // 3

		kernel_z = kernel[:, :units]
		kernel_r = kernel[:, units:2*units]
		kernel_h = kernel[:, 2*units:]

		recurrent_kernel_z = recurrent_kernel[:, :units]
		recurrent_kernel_r = recurrent_kernel[:, units:2*units]
		recurrent_kernel_h = recurrent_kernel[:, 2*units:]
		
		# code here
		#####################################################################################
		# Initialize
		x_z = None
		x_r = None
		x_h = None

		# Compute for reset, update and new gate (matrix1 + matrix2)
		x_z = sigmoid(np.matmul(x, kernel_z) + np.matmul(prev_h, recurrent_kernel_z))
		x_r = sigmoid(np.matmul(x, kernel_r) + np.matmul(prev_h, recurrent_kernel_r))
		x_h = np.tanh(np.matmul(x, kernel_h) + np.matmul(x_r * prev_h, recurrent_kernel_h))
		#####################################################################################
		
		output = (1 - x_z) * x_h + x_z * prev_h
		return output

	def backward(self, out_grad, input, kernel, recurrent_kernel):
		"""
		# Arguments
			in_grads: numpy array with shape (batch, units), gradients to outputs
			inputs:   [input numpy array with shape (batch, in_features), 
					   state numpy array with shape (batch, units)], same with forward inputs

		# Returns
			out_grads: [gradients to input numpy array with shape (batch, in_features), 
						gradients to state numpy array with shape (batch, units)]
		"""
		x, prev_h = input
		_, all_units = kernel.shape
		units = all_units // 3

		kernel_z = kernel[:, :units]
		kernel_r = kernel[:, units:2 * units]
		kernel_h = kernel[:, 2 * units:all_units]

		recurrent_kernel_z = recurrent_kernel[:, :units]
		recurrent_kernel_r = recurrent_kernel[:, units:2*units]
		recurrent_kernel_h = recurrent_kernel[:, 2*units:all_units]

		# code here
		#####################################################################################
		# Initialize
		x_grad, prev_h_grad = np.zeros_like(x), np.zeros_like(prev_h)
		kernel_z_grad, recurrent_kernel_z_grad = np.zeros_like(kernel_z), np.zeros_like(recurrent_kernel_z)
		kernel_r_grad, recurrent_kernel_r_grad = np.zeros_like(kernel_r), np.zeros_like(recurrent_kernel_r)
		kernel_h_grad, recurrent_kernel_h_grad = np.zeros_like(kernel_h), np.zeros_like(recurrent_kernel_h)

		# Compute basic information
		x_z = sigmoid(np.matmul(x, kernel_z) + np.matmul(prev_h, recurrent_kernel_z))
		x_r = sigmoid(np.matmul(x, kernel_r) + np.matmul(prev_h, recurrent_kernel_r))
		x_h = np.tanh(np.matmul(x, kernel_h) + np.matmul(prev_h * x_r, recurrent_kernel_h))

		# Compute for new gate
		tmp_h = out_grad * (1 - x_h**2) * (1 - x_z)
		matrix1_h = np.matmul(tmp_h, np.transpose(kernel_h))
		matrix2_h = np.matmul(tmp_h, np.transpose(recurrent_kernel_h))

		# Compute for update gate
		tmp_z = out_grad * (prev_h - x_h) * x_z * (1 - x_z)
		matrix1_z = np.matmul(tmp_z, np.transpose(kernel_z))
		matrix2_z = np.matmul(tmp_z, np.transpose(recurrent_kernel_z))

		# Compute for reset gate
		tmp_r = matrix2_h * prev_h * (x_r * (1 - x_r))
		matrix1_r = np.matmul(tmp_r, np.transpose(kernel_r))
		matrix2_r = np.matmul(tmp_r, np.transpose(recurrent_kernel_r))
		
		# Compute the gradient of input
		x_grad      = matrix1_z + matrix1_r + matrix1_h
		prev_h_grad = matrix2_z + matrix2_r + matrix2_h * x_r + out_grad * x_z
		
		# Compute the gradient of kernel
		kernel_r_grad = np.matmul(np.transpose(x), tmp_r)
		kernel_z_grad = np.matmul(np.transpose(x), tmp_z)
		kernel_h_grad = np.matmul(np.transpose(x), tmp_h)

		# Compute the gradient of recurrent kernel
		recurrent_kernel_r_grad = np.matmul(np.transpose(prev_h), tmp_r)
		recurrent_kernel_z_grad = np.matmul(np.transpose(prev_h), tmp_z)
		recurrent_kernel_h_grad = np.matmul(np.transpose(prev_h * x_r), tmp_h)
		#####################################################################################

		in_grad = [x_grad, prev_h_grad]
		kernel_grad           = np.concatenate([kernel_z_grad, kernel_r_grad, kernel_h_grad], axis=-1)
		recurrent_kernel_grad = np.concatenate([recurrent_kernel_z_grad, recurrent_kernel_r_grad, recurrent_kernel_h_grad], axis=-1)

		return in_grad, kernel_grad, recurrent_kernel_grad


class softmax_cross_entropy(operator):
	def __init__(self):
		super(softmax_cross_entropy, self).__init__()

	def forward(self, input, labels):
		"""
		# Arguments
			input:  numpy array with shape (batch, num_class)
			labels: numpy array with shape (batch,)
			eps:    float, precision to avoid overflow

		# Returns
			output: scalar, average loss
			probs:  the probability of each category
		"""
		# precision to avoid overflow
		eps = 1e-12

		batch = len(labels)
		input_shift = input - np.max(input, axis=1, keepdims=True)
		Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)

		log_probs = input_shift - np.log(Z+eps)
		probs = np.exp(log_probs)
		output = -1 * np.sum(log_probs[np.arange(batch), labels]) / batch
		return output, probs

	def backward(self, input, labels):
		"""
		# Arguments
			input:  numpy array with shape (batch, num_class)
			labels: numpy array with shape (batch,)
			eps:    float, precision to avoid overflow

		# Returns
			in_grad: gradient to forward input of softmax cross entropy, with shape (batch, num_class)
		"""
		# precision to avoid overflow
		eps = 1e-12

		batch = len(labels)
		input_shift = input - np.max(input, axis=1, keepdims=True)
		Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)
		log_probs = input_shift - np.log(Z+eps)
		probs = np.exp(log_probs)

		in_grad = probs.copy()
		in_grad[np.arange(batch), labels] -= 1
		in_grad /= batch
		return in_grad

		