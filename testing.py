# %% Import python packages

# Run the setup script to download the packages
# * This uses pip install so if your using anaconda
#   you will have to look at setup.packages to see what
#   packages are required.
import spafe.features
import spafe.features.gfcc
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

# Setting audio constants
SAMPLE_RATE = 22050
NUM_SAMPLES = 2*SAMPLE_RATE # two seconds of audio
BATCH_SIZE = 128

WIN_LENGTH = 2**11
HOP_LENGTH = 2**10
NCEPS = 13 # 64?
NFILTS = 26

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

# %% 

# Creating dictionaries to hold the data and the corresponding class label as a key
signals = {}
fft = {}
fbank = {}
mfccs = {}
gfccs = {}
scs = {}
sfs = {}
srs = {}
afte = {}

# Getting all the unique class labels
classes = list(np.unique(annotations.ClassLabel))

# Iterating through the classes and selecting the first signal from each to extract features
for c in classes:
  wav_file = annotations[annotations.ClassLabel == c].iloc[0, 0]
  signal, fs = librosa.load(os.path.join(C_AUDIO_FILES, wav_file), mono=True, sr=None)
  signals[c] = signal
  fft[c] = feature_extraction.calculate_fft(signal, fs)
  scs[c] = feature_extraction.spectral_centroid(y=signal, sr=fs)[0]
  srs[c] = feature_extraction.spectral_rolloff(y=signal+0.01, sr=fs)[0]
  sfs[c] = feature_extraction.spectral_flatness(y=signal).T
  fbank[c] = feature_extraction.logfbank(signal[:fs], fs, nfilt=NFILTS, nfft=WIN_LENGTH).T
  mfccs[c] = feature_extraction.mfcc(signal[:fs], fs, numcep=NCEPS, nfilt=NFILTS, nfft=WIN_LENGTH).T
  gfccs[c] = feature_extraction.gfcc(signal[:fs], fs, num_ceps=NCEPS, nfilts=NFILTS, nfft=WIN_LENGTH).T
  # COULD ADD zerocrossing HERE

# Plotting the feature extractions of the audio
if PLOTTING:
    plotting.plot_signals_time(signals)
    plotting.plot_ffts(fft)
    plotting.plot_spectral_feature(scs, 'Centroid')
    plotting.plot_spectral_feature(srs, 'Rolloff')
    plotting.plot_spectral_feature(sfs, 'Flatness')
    plotting.plot_spectrogram(fbank, 'Filter Bank')
    plotting.plot_spectrogram(mfccs, 'MFCC')
    plotting.plot_spectrogram(gfccs, 'GFCC')

# %% Creating the dataset

# Creating a mel spectogram for the feature extraction / transformation

transforms = feature_extraction.ExtractMFCC(SAMPLE_RATE, NCEPS, NFILTS, WIN_LENGTH)

# Creating the dataset and dataloader
audio_dataset = dataset.AudioDataset(annotations, AUDIO_FILES, transforms, SAMPLE_RATE, NUM_SAMPLES)
data_loader = DataLoader(audio_dataset, batch_size=BATCH_SIZE)

signal, fs = librosa.load(os.path.join(AUDIO_FILES, 'cello-01.wav'))

test = {}
for i, c in enumerate(classes):
    feature, classID = audio_dataset[i]
    test[c] = feature

plotting.plot_spectrogram(test, 'MFCC')