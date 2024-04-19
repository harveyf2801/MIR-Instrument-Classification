# %% Import python packages

# Run the setup script to download the packages
# * This uses pip install so if your using anaconda
#   you will have to look at setup.packages to see what
#   packages are required.
import setup
import os
from pathlib import Path
import logging

try:
    # Import the python packages required
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchaudio
    from torchsummary import summary

    import pandas as pd
    import numpy as np
    from scipy.io import wavfile

    import librosa

    import matplotlib.pyplot as plt

    from tqdm import tqdm

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold

    # Custom modules
    import dataset
    import plotting
    import feature_extraction

except ModuleNotFoundError:
    setup.run()
    logging.warning('Please re-run the script. If you see this message again, make sure you have downloaded the correct external packages by looking at `setup.py` packages')
    exit()

# %% Setting global constants
# The directory paths for the audio files
PWD = os.getcwd()
C_AUDIO_FILES = Path(PWD, 'clean')
AUDIO_FILES = Path(PWD, 'wavfiles')
ANNOTATIONS = Path(PWD, 'annotations.csv')

# Setting audio and model constants
SAMPLE_RATE = 22050
NUM_SAMPLES = 2*SAMPLE_RATE # two seconds of audio

WIN_LENGTH = 2**11
HOP_LENGTH = 2**10
NCEPS = 13 # 64?
NFILTS = 26

BATCH_SIZE = 199
K_FOLDS = 5

# Setting whether to run plots or not
PLOTTING = False

# %% Cleaning the audio files by stripping the silence
clean_files = os.listdir(C_AUDIO_FILES)
# logging.info("Cleaning the audio files")
# for file in tqdm(clean_files):
# 	# Ignoring files already cleaned and all the 'hidden' files (such as .DS_store used for MacOS)
# 	if (file in clean_files) or (file[0] == '.'):
# 		continue
# 	# Loading the audio file at 16000 Hz
# 	signal, fs = librosa.load(os.path.join(AUDIO_FILES, file), mono=True, sr=SAMPLE_RATE)
# 	# Getting the mask envelope for the audio
# 	mask = feature_extraction.envelope(signal, fs, 0.0005)
# 	# Writing this masked and resampled audio into the clean directory
# 	wavfile.write(filename=os.path.join(C_AUDIO_FILES, file), rate=fs, data=signal[mask])

# %% Create annotations file if not already created for the dataset
# or load in the annotations csv file
if not os.path.exists(ANNOTATIONS):
    # Getting the dataframe for the file annotations
    annotations = dataset.create_file_annotations(AUDIO_FILES)
    # Save the dataframe to a CSV file
    annotations.to_csv(path_or_buf=ANNOTATIONS, sep=',', encoding='utf-8', index=False)
    logging.info("Creating annotations .csv file")
else:
    annotations = pd.read_csv(ANNOTATIONS)

# Plotting the class distribution for the dataset
if PLOTTING:
    plotting.plot_class_distribution(annotations)

# %% Creating the dataset

# Creating a mel spectogram for the feature extraction / transformation
transforms = feature_extraction.ExtractMFCC(SAMPLE_RATE, NCEPS, NFILTS, WIN_LENGTH)

# Other features can be used such as the GFCC, FBank, Spectral (Centroid, Rolloff, Flatness), etc.

# Creating the dataset
# This dataset performs pre-processing on the audio files
# such as resampling, resizing (to mono), normalization, and feature extraction
audio_dataset = dataset.AudioDataset(annotations, AUDIO_FILES, transforms, SAMPLE_RATE, NUM_SAMPLES)

# %% Displaying each feature extraction for the first audio file in each class

# Creating dictionaries to hold the data and the corresponding class label as a key
tmp_signals = {}
tmp_fft = {}
tmp_zcs = {}
tmp_fbank = {}
tmp_mfccs = {}
tmp_gfccs = {}
tmp_scs = {}
tmp_sfs = {}
tmp_srs = {}

# Getting all the unique class labels
classes = list(np.unique(annotations.ClassLabel))

# Iterating through the classes and selecting the first signal from each to extract features
for c in classes:
    wav_file = annotations[annotations.ClassLabel == c].iloc[0, 0]
    signal, fs = librosa.load(os.path.join(C_AUDIO_FILES, wav_file), mono=True, sr=None)
    tmp_signals[c] = signal
    tmp_fft[c] = feature_extraction.calculate_fft(signal, fs)
    tmp_zcs[c] = feature_extraction.zero_crossing_rate(signal).T
    tmp_scs[c] = feature_extraction.spectral_centroid(y=signal, sr=fs).T
    tmp_srs[c] = feature_extraction.spectral_rolloff(y=signal+0.01, sr=fs).T
    tmp_sfs[c] = feature_extraction.spectral_flatness(y=signal).T
    tmp_fbank[c] = feature_extraction.logfbank(signal[:fs], fs, nfilt=NFILTS, nfft=WIN_LENGTH).T
    tmp_mfccs[c] = feature_extraction.mfcc(signal[:fs], fs, numcep=NCEPS, nfilt=NFILTS, nfft=WIN_LENGTH).T
    tmp_gfccs[c] = feature_extraction.gfcc(signal[:fs], fs, num_ceps=NCEPS, nfilts=NFILTS, nfft=WIN_LENGTH).T

# Plotting the feature extractions of the audio
if PLOTTING:
    plotting.plot_signals_time(tmp_signals)
    plotting.plot_ffts(tmp_fft)
    plotting.plot_time_feature(tmp_zcs, 'Zero Crossing Rate')
    plotting.plot_time_feature(tmp_scs, 'Spectral Centroid')
    plotting.plot_time_feature(tmp_srs, 'Spectral Rolloff')
    plotting.plot_time_feature(tmp_sfs, 'Spectral Flatness')
    plotting.plot_spectrogram(tmp_fbank, 'Filter Bank')
    plotting.plot_spectrogram(tmp_mfccs, 'MFCC')
    plotting.plot_spectrogram(tmp_gfccs, 'GFCC')

# del tmp_signals, tmp_fft, tmp_fbank, tmp_mfccs, tmp_gfccs, tmp_scs, tmp_sfs, tmp_srs

# %% Creating a kNN model using PCA and t-SNE for dimensionality reduction

kf = KFold(n_splits=K_FOLDS, shuffle=True)

# Loop through each fold
for fold, (train_idx, test_idx) in enumerate(kf.split(audio_dataset)):
    print(f"Fold {fold + 1}")
    print("-------")

    # Define the data loaders for the current fold
    train_loader = DataLoader(
        dataset=audio_dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    test_loader = DataLoader(
        dataset=audio_dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )

    # Train the model on the current fold
    for data, target in test_loader:
        print(target)
    
    for data, target in test_loader:
        print(target)
