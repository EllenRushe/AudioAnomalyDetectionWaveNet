import os
import sys
import glob
import time
import random
import io, json
import logging
import argparse
import numpy as np
import tensorflow as tf



# from experiment.val import val
from experiment.test import test
from experiment.train import train

from utils.Params import Params
from utils.LoadData import write_data



if __name__=="__main__":

	# Model name passed as first argument, must be one defined in HParams.yaml
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--model_name", 
		help="Model name as defined in HParams.yaml.", 
		type=str
		)
	parser.add_argument(
		"--hparams_file", 
		help="Name of hyperparamenters YAML.", 
		type=str
		)
	parser.add_argument(
		"--mode", 
		help="The mode for script, 'train','test' or 'val'.",
		type=str
		)
	parser.add_argument(
		"--scene_name", 
		help="Name of scene in dataset, blank to train on all scenes.",
		type=str
		)
	parser.add_argument(
		"--write_data",
		help="Download data and write to binaries, 'y' for yes, 'n' for no.",
		type=str,
		choices={'y','n'},
		default='n'
		)
	args = parser.parse_args()
	# Read in model and configuration hyperparamenters.
	params=Params(args.hparams_file , args.model_name)
	config = Params("config.yaml", 'dir_config')
 
	# Set visible devices for GPU.
	os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
	if not os.path.exists(params.log_dir):
		os.makedirs(params.log_dir)
 
	segment_length = params.segment_length
	segment_overlap = params.segment_overlap
	if args.write_data == 'y':
		write_data(segment_length,segment_overlap, config.data_dir, params.binaries_dir)
 
	# if a specific scene is selected.
	if args.scene_name is not None:
		# only negative examples in training set.
		file_list = sorted(glob.glob('data_binaries/{}/*/{}/*.tfrecord'.format(args.mode, args.scene_name)))
		print('Training model on scene: {}'.format(args.scene_name))
 
	else:
		file_list = sorted(glob.glob('data_binaries/{}/*/*/*.tfrecord'.format(args.mode)))
		args.scene_name = 'all'
		print('Training model all scenes: {}'.format(args.scene_name))
 
	if args.mode == 'train':
		train_list = [f for f in file_list if 'neg' in f]
		train(train_list, params, args.scene_name)
 
	elif args.mode == 'test':
		for d in [params.preds_dir ,params.labels_dir, params.targets_dir]:
		   if not os.path.exists(d):
			   os.makedirs(d)
		latest_ckpt = tf.train.latest_checkpoint(os.path.join(params.checkpoints_dir, args.scene_name))
		# Keep positive examples in testing phase.
		test(file_list, params, args.scene_name, latest_ckpt)
	else:
		print("Mode does not exist. Modes: 'train' and 'test'")
		exit(1)




