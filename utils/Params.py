from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams
import tensorflow as tf

class Params(HParams):
	'''Subclass tensorflows HParams: https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams'''
	def __init__(self, yaml_file, config_name):
		super().__init__()
		with open(yaml_file) as f:
			yaml = YAML(typ='safe')
			# Load and find
			yaml_map = yaml.load(f)
			config_dict = yaml_map[config_name]
			for k, v in config_dict.items():
				# Inherits add_hparam from HParams, uses setattr to create key-value pair. 
				self.add_hparam(k,v)

 