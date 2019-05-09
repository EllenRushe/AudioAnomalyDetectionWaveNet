import tensorflow as tf
from models.NetworkUtils import conv1d, conv1d_transpose

class CAE:

	def __init__(self, inputs, params):
		self.name = 'CAE'
		self.inputs = inputs
		for k, v in params.items():
			setattr(self, k, v)


	def model(self, is_training):
		# In this case the outputs are the inputs for calculating the MSE.  
		# Overload mu-law targets for standard autoencoders targets should 
		# just be the input. 
		is_training_pl = tf.placeholder_with_default(is_training, shape=[], name='is_training_pl')

		# He et al.,2015 http://arxiv.org/abs/1502.01852
		weight_init = tf.keras.initializers.he_uniform(seed=None)
		# Just to build up string representation of architecture. 
		model_str = ""
		self.inputs = tf.expand_dims(self.inputs, 2, name='data_entry_point')

		# ENCODER
		encoder = self.inputs
		model_str+='Input shape : {}\n'.format(encoder.get_shape())
		# Initialise number of hidden layer nodes then divide by two
		# each layer. Number of filters increases while feature maps
		# decrease in size. 
		hidden = self.num_filters
		for i in range(self.num_layers):
			# Stride alternates between 1 and 2 so we don't downsample too much.
			# Starts at 1.
			# Layer @ 10: [2,1,2,1,2,1,2,1,2,1]
			stride = 1 if (i+1)%2==0 else 2
			encoder =  conv1d(
				encoder, 
				kernel_size = self.kernel_size, 
				num_filters = hidden,
				causal = False,
				dilation_rate = 1,
				padding = 'SAME',
				stride = stride, 
				weight_init=weight_init,
				name='encoder_{}_conv1d'.format(i)
				)
			encoder = tf.layers.batch_normalization(
				inputs=encoder, 
				training=is_training_pl, 
				name='encoder_{}_bn'.format(i)
				)
			encoder = tf.nn.relu(encoder, name='encoder_relu_{}'.format(i))
			if i%2 !=0:
				hidden = hidden*2
			model_str+='Encoder {} : {}\n'.format(i+1, encoder.get_shape())
	
		encoding =  conv1d(
				encoder, 
				kernel_size = self.kernel_size, 
				num_filters = hidden,
				causal = False,
				dilation_rate = 1,
				padding = 'SAME',
				stride=2, 
				weight_init = weight_init,
				name='encoding_conv1d'
				)
		encoding = tf.layers.batch_normalization(
				inputs=encoding, 
				training=is_training_pl, 
				name='encoding_bn'
				)
		encoding = tf.nn.relu(encoding, name='encoding_relu')
		model_str+='Encoding: {}\n'.format(encoding.get_shape())

		# DECODER
		decoder = encoding
		hidden = hidden//2
		for i in range(self.num_layers):
			stride = 1 if (i+1)%2==0 else 2
			decoder =  conv1d_transpose(
				decoder,  
				kernel_size = self.kernel_size, 
				num_filters = hidden, 
				stride=stride, 
				weight_init = weight_init,
				name='decoder_{}_conv1d_trans'.format(i)
				)
			decoder = tf.layers.batch_normalization(
				inputs=decoder, 
				training=is_training_pl, 
				name='decoder_{}_bn'.format(i)
				)
			decoder = tf.nn.relu(decoder, name='decoder_relu_{}'.format(i))
			if i%2 !=0:
				hidden = hidden//2
			if i == self.num_layers:
				assert hidden == int(self.inputs.get_shape()[-1])
			model_str+='Decoder {} : {}\n'.format(i+1, decoder.get_shape())

		decoder =  conv1d_transpose(
			decoder,  
			kernel_size = self.kernel_size, 
			# Last layer has to have the same number of nodes as the input
			num_filters = int(self.inputs.get_shape()[-1]), 
			stride=2, 
			weight_init = weight_init,
			name='output_op'
			)
		model_str+='Ouput shape: {}\n'.format(decoder.get_shape())
		# Printing quick representation of architecture. 
		print(model_str)
		return decoder, self.inputs



