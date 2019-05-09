import tensorflow as tf 
import numpy as np

def generate_weights(weight_shape, weight_init, layer_name, bias_shape=None):
	'''
	Function for generating weights and biases.
	:param: weight_init: Weight initializer. 
			Default is 'tf.glorot_uniform_initializer' glorot_uniform_initializer 
			(See https://www.tensorflow.org/api_docs/python/tf/get_variable)
	:param: List: weight_shape: Shape of layer weight matrix 
	:param: str: Layer name: Name for weight and bias variables. 
	:param: List: bias_shape: Shape of layer bias vector. Default is None. 
	:return: weights and biases if bias_shape is not None and just weights
			otherwise
	'''
	# Default weight init if None: 
	weights = tf.get_variable(
		shape=weight_shape, 
		initializer=weight_init, 
		name=layer_name+"_weights")
	if bias_shape != None:
		biases = tf.get_variable(
			shape=bias_shape, 
			initializer=tf.zeros_initializer(), 
			name=layer_name+"_biases")
		return weights, biases
	return weights


def conv1d(x, kernel_size, num_filters,  causal=False, dilation_rate=1, 
		padding='SAME', stride=1, weight_init=None, name=''):
	'''
	Wrapper for 1D convolutional layer (with both dilation and causal options)
	:param: tensor: x: Input to convolutional layer. 
	:param: int: kernel_size: Size of kernel for convolution. 
	:param: int: num_filters: Number of filters (a.k.a. kernels/channels) in output.
	:param: bool: causal: True for causal convolutions. Uses causal padding
			with zeros like in Keras: 
			[1] https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
	
	:param: dilation_rate: Rate for dilated convolution. If dilation=1,
			we just have normal convolution. 
	:param: padding: 'VALID' or 'SAME'. (Note default is 'VALID' and for causal 
			convolution, padding will be overwritten as 'VALID')
	:param: weight_init: Weight initializer. 
			Default is 'tf.glorot_uniform_initializer' glorot_uniform_initializer 
			(See https://www.tensorflow.org/api_docs/python/tf/get_variable)
	:param: Stride of convolution kernel: Default is 1. 
	:param: name: name of operation in tensorflow graph. 
	'''
	# Infer number of channels
	input_channels = int(x.get_shape()[-1])
	# Generate weights
	weights, biases=  generate_weights(
		[kernel_size, input_channels, num_filters], 
		 weight_init,
		 name,
		 bias_shape=[num_filters])	
	if causal:
		# Left padding for causal convolutions (See ref [1])
		causal_padding = dilation_rate*(kernel_size - 1)
		x = tf.pad(x, [[0,0],[causal_padding, 0], [0,0]])
		padding = 'VALID'

	if dilation_rate > 1:
		# Dialated convolution.
		# TODO: Need to put in a check to see if input % dilation_rate == 0
		# Docs under: 'tf.manip.space_to_batch_nd' for r1.11
		stb = tf.space_to_batch_nd(
			x, 
			paddings=[[0,0], [0,0]], 
			block_shape=[dilation_rate,1],
			name='stb_{}'.format(name)
			)
		conv_1d = tf.nn.conv1d(
			stb, 
			weights, 
			stride=stride, 
			padding=padding,
			name='op_{}'.format(name)
			)
		# conv_1d = tf.nn.bias_add(conv_1d, biases)
		conv_1d = tf.batch_to_space_nd(
			conv_1d, 
			crops=[[0,0], [0,0]],
			block_shape=[dilation_rate,1],
			name='bts_{}'.format(name)
			)
		conv_1d = tf.nn.bias_add(conv_1d, biases, name='bias_add_{}'.format(name))
	else: 
		conv_1d = tf.nn.conv1d(
			x, 
			weights, 
			stride=stride, 
			padding=padding, 
			name='op_{}'.format(name)
			)
		conv_1d = tf.nn.bias_add(conv_1d, biases, name='bias_add_{}'.format(name))
	return conv_1d 


def conv1d_transpose(x, kernel_size,  num_filters, stride=1, weight_init=None, name=''):

	'''
	Wrapper for 1D transposed convolutional layer, a.k.a. (incorrectly) 'deconvolution'. Used for upsampling
	autoencoders etc. 
	:param: tensor: x: Input to convolutional layer. 
	:param: int: kernel_size: Size of kernel for convolution. 
	:param: int: num_filters: Number of filters (a.k.a. kernels/channels) in output.
	:param: Stride of convolution kernel: Default is 1. 
	:param: weight_init: Weight initializer. 
			Default is 'tf.glorot_uniform_initializer' glorot_uniform_initializer 
			(See https://www.tensorflow.org/api_docs/python/tf/get_variable)
	:param: name: name of operation in tensorflow graph. 
	'''
	# Shape inference.
	input_channels = int(x.get_shape()[-1])
	batch_size= x.get_shape()[0]
	output_shape = [tf.shape(x)[0], tf.shape(x)[1]*stride, num_filters]
	weights, biases=  generate_weights(
		[kernel_size, num_filters,  input_channels], 
		weight_init,
		name,
		bias_shape=[num_filters]
		)	
	conv_1d_transpose = tf.contrib.nn.conv1d_transpose(
		x, 
		weights,
		output_shape=output_shape, 
		stride=stride,
		name=name
		)
	conv_1d_transpose = tf.nn.bias_add(conv_1d_transpose , biases, name='bias_add_{}'.format(name))
	return conv_1d_transpose