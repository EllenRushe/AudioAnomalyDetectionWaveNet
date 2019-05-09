import tensorflow as tf
from .CAE import CAE


def _parse_function(file):
	features = {
		'label': tf.FixedLenFeature([], tf.int64),
		'audio_inputs': tf.FixedLenFeature([], tf.string),
		'scene': tf.FixedLenFeature([], tf.string),
		'source_file': tf.FixedLenFeature([], tf.string)
	}
	parsed_features = tf.parse_single_example(file, features=features)
	label = [] # No labels being passed in training, just a placeholder.
	# Quanitised value
	audio_quant = tf.decode_raw(parsed_features['audio_inputs'], tf.int64)
	# Range between -1 and 1
	audio_inputs = tf.cast(audio_quant, dtype=tf.float32)/128 - 1
	scene = tf.decode_raw(parsed_features['scene'], tf.uint8)
	source_file = tf.decode_raw(parsed_features['source_file'], tf.uint8)
	return audio_inputs, label

def _test_parse_function(file):
	features = {
		'label': tf.FixedLenFeature([], tf.int64),
		'audio_inputs': tf.FixedLenFeature([], tf.string),
		'scene': tf.FixedLenFeature([], tf.string),
		'source_file': tf.FixedLenFeature([], tf.string)
	}
	parsed_features = tf.parse_single_example(file, features=features)
	label = tf.cast(parsed_features['label'], tf.int64)
	# Quanitised value
	audio_quant = tf.decode_raw(parsed_features['audio_inputs'], tf.int64)
	# Range between -1 and 1
	audio_inputs = tf.cast(audio_quant, dtype=tf.float32)/128 - 1
	scene = tf.decode_raw(parsed_features['scene'], tf.uint8)
	source_file = tf.decode_raw(parsed_features['source_file'], tf.uint8)
	return audio_inputs, label

def get_train_input_fn(filenames, batch_size, shuffle_size, num_epochs):
	def train_input_fn():
		dataset = tf.data.TFRecordDataset(filenames)
		dataset =  dataset.shuffle(shuffle_size)
		dataset = dataset.repeat(num_epochs)
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(batch_size)
		return dataset
	return train_input_fn

def get_eval_input_fn(filenames, batch_size):
	def eval_input_fn():
		dataset = tf.data.TFRecordDataset(filenames)
		dataset = dataset.map(_test_parse_function)
		dataset = dataset.batch(batch_size)
		iterator = dataset.make_one_shot_iterator()
		next_element = iterator.get_next()
		return next_element
	return eval_input_fn

def model_fn(features, labels, mode, params):
	'''
	We do not have targets here because the targets are the input. 
	:param: Tensor: inputs: Next batch of input data. 
	:param: tf.Variable: global_step: Variable that to keep track of training steps. 
	:param: tf.estimators.ModeKeys: mode: 'TRAIN', PREDICT', 'EVAL'
	'''
	is_training_pl = tf.placeholder(tf.bool, shape=[], name='is_training_pl')

	cae_model = CAE(features, params) 

	if mode == tf.estimator.ModeKeys.TRAIN:
		# Model function returns the decoder outputs and the inputs as the targets. 
		logits, targets= cae_model.model(is_training=True)
	else:
		logits, targets = cae_model.model(is_training=False)
	
	mse = tf.metrics.mean_squared_error(
			tf.cast(logits, tf.float32), 
			tf.cast(targets, tf.float32), 
			name = 'mse_metric'
			)

	if mode == tf.estimator.ModeKeys.PREDICT:
		# Add targets and set predict flag to get the whole batch. 
		predictions = {
			'predictions': logits,
			'targets': targets
			}
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions)

	loss = tf.losses.mean_squared_error(
			labels=targets, 
			predictions=logits,
			)
		


	metrics_eval = { 
		'mse': mse
		}

	if mode == tf.estimator.ModeKeys.EVAL: 	
		return tf.estimator.EstimatorSpec(
			mode=mode, 
			loss=loss, 
			eval_metrics_ops=metrics_eval
			)

	assert mode == tf.estimator.ModeKeys.TRAIN

	logging_hook = tf.train.LoggingTensorHook(
		{"loss" : loss, "step": tf.train.get_global_step()}, 
		every_n_iter=1
		)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	# Used for switching batch normalisation 'is training' variable. 
	
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(params['learning_rate'])
		update_op = optimizer.minimize(
			loss, 
			tf.train.get_global_step(), 
			name='update_op'
			)

	return tf.estimator.EstimatorSpec(
		mode=mode,
		loss=loss,
		train_op=update_op,
		training_hooks = [logging_hook]
		)		




