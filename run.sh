#!/bin/bash

# Model names are defined in HParams.yaml. 
model_name=$1
# Choose either 'train' or 'test'
mode=$2

scenes=( 
	"beach"
	"bus"
	"cafe_restaurant"
	"car"
	"city_center"
	"forest_path"
	"grocery_store"
	"home"
	"library"
	"metro_station"
	"office"
	"park"
	"residential_area"
	"train"
	"tram"
	)
#for scene in "${scenes[@]}"; do
	python main.py --model_name $model_name --hparams_file HParams.yaml --mode $mode --scene_name beach --write_data y
# done
# Empty scene to training on all scenes. 
# python $script_name --model_name $model_name --hparams_file HParams.yaml 
