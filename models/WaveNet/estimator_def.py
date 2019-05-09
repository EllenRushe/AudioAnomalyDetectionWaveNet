import tensorflow as tf
from .WaveNet import WaveNet


def _parse_function(file):
    features = {
        'label': tf.FixedLenFeature([], tf.int64),
        'audio_inputs': tf.FixedLenFeature([], tf.string),
        'scene': tf.FixedLenFeature([], tf.string),
        'source_file': tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_single_example(file, features=features)
    label = []
    # Quanitised values in the range 0 to 255
    audio_quant = tf.decode_raw(parsed_features['audio_inputs'], tf.int64)
    # The data is first scaled (between 0 and 255) by dividing by
    # 128, -1 then adjusts the range to values between -1 and 1.
    # tensorflow division: '[...] if one of x or y is a float, 
    # then the result will be a float.'
    # See: https://www.tensorflow.org/api_docs/python/tf/div
    audio_inputs = tf.cast(audio_quant, dtype=tf.float32)/128 - 1
    scene = tf.decode_raw(parsed_features['scene'], tf.uint8)
    source_file = tf.decode_raw(parsed_features['source_file'], tf.uint8)
    return (audio_inputs, audio_quant), label

def _test_parse_function(file):
    features = {
        'label': tf.FixedLenFeature([], tf.int64),
        'audio_inputs': tf.FixedLenFeature([], tf.string),
        'scene': tf.FixedLenFeature([], tf.string),
        'source_file': tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_single_example(file, features=features)
    label = tf.cast(parsed_features['label'], tf.int64)
    # Quanitised values in the range 0 to 255
    audio_quant = tf.decode_raw(parsed_features['audio_inputs'], tf.int64)
    # The data is first scaled (between 0 and 255) by dividing by
    # 128, -1 then adjusts the range to values between -1 and 1.
    # tensorflow division: '[...] if one of x or y is a float, 
    # then the result will be a float.'
    # See: https://www.tensorflow.org/api_docs/python/tf/div
    audio_inputs = tf.cast(audio_quant, dtype=tf.float32)/128 - 1
    scene = tf.decode_raw(parsed_features['scene'], tf.uint8)
    source_file = tf.decode_raw(parsed_features['source_file'], tf.uint8)
    return (audio_inputs, audio_quant), label


def get_train_input_fn(filenames, batch_size, shuffle_size, num_epochs):
	def train_input_fn():
		dataset = tf.data.TFRecordDataset(filenames)
		dataset =  dataset.shuffle(shuffle_size)
		dataset = dataset.repeat(num_epochs)
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(batch_size)
		dataset = dataset.prefetch(4)
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
		
	wavenet_model = WaveNet(features[0], params) 
	# There is no need for an if statement here becaue there are no
	# different parameters if the model is in training or testing. 
	logits= wavenet_model.model()


	preds_prob= tf.nn.softmax(logits, name='softmax_op')
	preds = tf.argmax(preds_prob, 2, name='preds_argmax')

	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'probabilities': preds_prob,
			'predictions': preds,
			'targets': features[1]
			}

		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions)
	
	loss = tf.reduce_mean(
		tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=logits, 
			labels=features[1], 
			name='loss')
		)


	if mode == tf.estimator.ModeKeys.EVAL: 	
		return tf.estimator.EstimatorSpec(
			mode=mode, 
			loss=loss, 
			# eval_metrics_ops=metrics
			)

	assert mode == tf.estimator.ModeKeys.TRAIN

	# logging_hook = tf.train.LoggingTensorHook(
	# 	{"loss" : loss, "step": tf.train.get_global_step()}, 
	# 	every_n_iter=1
	# 	)


	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	# Used for switching batch normalisation. 
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
		# training_hooks = [logging_hook]
		)


