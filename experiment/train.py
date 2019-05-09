import os
import json
import logging
import numpy as np
import tensorflow as tf
from utils.LoadData import count_examples
from sklearn.metrics import roc_curve, auc


def train(train_filenames, params, scene_name):
	num_train_examples = count_examples(train_filenames)
	models_module = __import__('.'.join(['models', params.model, 'estimator_def']))
	model = getattr(models_module, params.model)
	estimator_def = model.estimator_def
	
	input_fn_train = estimator_def.get_train_input_fn(
		train_filenames, 
		params.batch_size, 
		num_train_examples,
		params.num_epochs
		)
	# https://github.com/shu-yusa/tensorflow-mirrored-strategy-sample/blob/master/cnn_mnist.py
	if params.num_gpus > 1:
		distribution = tf.contrib.distribute.MirroredStrategy(
			num_gpus=params.num_gpus)
	else: 
		distribution = None

	ckpt_hook = tf.train.CheckpointSaverHook(
		os.path.join(params.checkpoints_dir,scene_name),
		save_steps=int(np.ceil(num_train_examples/params.batch_size)/5), # Saves last 5 checkpoints (default in TF)
		checkpoint_basename='{}_{}.ckpt'.format(params.model, scene_name),
		)

	run_config = tf.estimator.RunConfig(
		model_dir=os.path.join(params.checkpoints_dir,scene_name),
		train_distribute=distribution
		)

	estimator= tf.estimator.Estimator(
		model_fn=estimator_def.model_fn,
		params=params.model_params,
		config=run_config
		)

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler('{}/{}_{}.log'.format(params.log_dir, params.model, scene_name))
	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	estimator.train(
		input_fn=input_fn_train,
		hooks = [ckpt_hook],
		steps=int(np.ceil(num_train_examples/params.batch_size))*params.num_epochs
		)
