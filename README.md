* Issues and pull requests temporarily closed as maintainer is away for a number of weeks.*
Ellen Rushe, and Brian Mac Namee. "Anomaly Detection in Raw Audio Using Deep Autoregressive Networks." IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019.

[Poster](https://sigport.org/documents/anomaly-detection-raw-audio-using-deep-autoregressive-networks)

### To view results
Please see FinalResults.ipynb
This file displays final results. Loading and running test set using saved model will generate prediction and label files. Checkpoints for saved models can be downloaded by running the 'download_preds.sh' file (note that these files are quite large with the total zip being ~8GB).  


### To train models again from scratch:
The models can be run again (final scores may obviously vary slightly due to different random initialisation).

To run experiment, modify the following 3 files or leave with default values. 

1. HParmas.yaml : This contains all hyperparameters for each mode. The two models present are a CAE (convolutional autoencoder) and a WaveNet model. 

2. config.yaml: This simply contains the data directory. (Note: In this file, the file path is set to be a relative path. It has been noted that for at least some systems an absolute path might be necessary to include for DCASE's dcase_util.datasets.TUTRareSoundEvents_2017_DevelopmentSet.file_meta to return the necessary data.)

3. run.sh : This script contains a loop to train a model for each scene. It takes two command line arguments. The model name (defined in HParams.yaml) and the mode (either 'train' or 'test')


### Additional notes
These models were trained on GPUs therefore we cannot gaurantee that they will load outside of this environment due to limitations in the particular version of Tensorflow used. Models should however train on CPUs (although may well take longer than is acceptable).

### __IMPORTANT__
Data will be downloaded automatically and binaries will be written. Please note that writing all binaries will take up approximately 48GB as the original data is approximately 41GB. For this reason, you should be careful that you have enough space on your disk and if not you want want to modify the code to perform this incrementally. 

