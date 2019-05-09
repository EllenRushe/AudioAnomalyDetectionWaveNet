

** TO VIEW RESULTS **
Please see FinalResults.ipynb
This file loads in pre-computed predictions and labels which can be automatically downloaded by runnning the 'download_preds.sh' file. 

** TO RUN TESTING WITH SAVED MODELS **
To run testing using our trained models, simply run run.sh with the --download_checkpoints='y'. This will automatically download checkpoints for both models (note that these files are quite large with the total zip being ~8GB). 

** TO TRAIN MODELS AGAIN FROM SCRATCH **
The models can be run again (final scores may obviously vary slightly due to different random initialisation).

To run experiment, modify the following 3 files or leave with default values. 

1. HParmas.yaml : This contains all hyperparameters for each mode. The two models present are a CAE (convolutional autoencoder) and a WaveNet model. 

2. config.yaml: This simply contains the data directory. 

3. run.sh : This script contains a loop to train a model for each scene. It takes two command line arguments. The model name (defined in HParams.yaml) and the mode (either 'train' or 'test')


** ADDITIONAL NOTES **
* Note that the use of glob means that records may be read in a different order at text time depending on how files are sorted in your operating system. Run test set and recompute the label list to get reported scores (original experiments were run on Ubuntu so saved predictions and labels are ordered accordingly). These models were trained on GPUs therefore we cannot gaurantee that they will load outside of this environment due to limitations in the particular version of Tensorflow used. Models should however train on CPUs (although may well take longer than is acceptable. )


** IMPORTANT **
Data will be downloaded automatically and binaries will be written. Please note that writing all binaries will take up approximately 48GB as the original data is approximately 41GB. For this reason, you should be careful that you have enough space on your disk and if not you want want to modify the code to perform this incrementally. 

