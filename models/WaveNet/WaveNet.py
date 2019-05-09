# Based on https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth/wavenet/h512_bo16.py
# which is Copyright 2019 The Magenta Authors which is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




import tensorflow as tf
from models.NetworkUtils import conv1d
from utils.PreProcess import shift_offset

''' '''

class WaveNet():

	def __init__(self, inputs, params):
		self.name = 'WaveNet'
		self.inputs = inputs
		for k, v in params.items():
			setattr(self, k, v)
	
	def model(self):
		# So we specifiy the channel in a single channel case. 
		self.inputs = tf.expand_dims(self.inputs , 2, name='data_entry_point')
		self.inputs = shift_offset(self.inputs)
		# Intialise the residual and skip layers. The first res layer is obviously just the first layer before it enters the loop. 
		res = conv1d(
			self.inputs,  
			kernel_size = self.kernel_size, 
			num_filters = self.num_filters,
			causal=True,
			dilation_rate=1, 
			name='first_layer'
			)
		skips = conv1d(
			self.inputs,  
			kernel_size = 1,
			num_filters = self.num_skip_filters, 
			causal=True,
			dilation_rate=1,  
			name='skips_init'
			)
		num_layers = self.stack_size*self.num_stacks
		for i in range(num_layers):
			dilation_rate = 2**(i % self.num_stacks)
			dil = conv1d(
				res, 
				kernel_size = self.kernel_size, 
				# Double up filters so they can be split. 
				num_filters = self.num_filters*2, 
				causal=True,
				dilation_rate=dilation_rate,  
				name='layer_{}'.format(i)
				)

			# Need to split channels because half the channels go to the gating, half go to the activation. 
			split_channels = dil.get_shape()[2]//2
			# Gating
			dil= tf.sigmoid(dil[:,:,:split_channels]) * tf.tanh(dil[:, :, split_channels:])
			# 1X1 conv to be passed to next layer. 
			res+=conv1d(
				dil, 
				kernel_size = 1, 
				num_filters = self.num_filters, 
				causal=False,
				dilation_rate=1,
				name='conv_before_res{}'.format(i)
				)
			skips+= conv1d(
				dil, 
				kernel_size = 1, 
				num_filters = self.num_skip_filters, 
				causal=False,
				dilation_rate=1,
				name='conv_before_skip{}'.format(i)
				)
		skips = tf.nn.relu(skips)
		skips = conv1d(
			skips, 
			kernel_size = 1, 
			num_filters = self.num_skip_filters, 
			causal=True,
			dilation_rate=1,
			name='conv_over_skips'
			)
		skips = tf.nn.relu(skips)
		logits = conv1d(
			skips, 
			kernel_size = 1, 
			num_filters = self.num_filters, 
			causal=True,
			dilation_rate=1,
			name='logits'
			)
	
		return logits 
		

