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

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.manifold import TSNE

    import pandas as pd
    import numpy as np
    import librosa

    import matplotlib.pyplot as plt

    from tqdm import tqdm

    # Custom modules
    import dataset
    import plotting
    import feature_extraction
    import dimentionality_reduction
    import models

except ModuleNotFoundError:
    setup.run()
    logging.warning('Please re-run the script. If you see this message again, make sure you have downloaded the correct external packages by looking at `setup.py` packages')
    exit()

# %% Setting global constants
# The directory paths for the audio files
PWD = os.getcwd()
AUDIO_FILES = Path(PWD, 'wavfiles')
ANNOTATIONS = Path(PWD, 'annotations.csv')
FEATURES = Path(PWD, 'features.csv')

# Setting audio and model constants
SAMPLE_RATE = 22050
NUM_SAMPLES = 2*SAMPLE_RATE # two seconds of audio

WIN_LENGTH = 2**11
HOP_LENGTH = 2**10
NCEPS = 13 # 64?
NFILTS = 26

BATCH_SIZE = 32
K_FOLDS = 5

# Setting whether to run plots or not
PLOTTING = False

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
transforms = [feature_extraction.ExtractRMS(),
              feature_extraction.ExtractSpectralFlatness(SAMPLE_RATE, WIN_LENGTH),
              feature_extraction.ExtractSpectralCentroid(SAMPLE_RATE, WIN_LENGTH),
              feature_extraction.ExtractSpectralRolloff(SAMPLE_RATE, WIN_LENGTH),
              feature_extraction.ExtractZeroCrossingRate(),
              feature_extraction.ExtractMFCC(SAMPLE_RATE, NCEPS, NFILTS, WIN_LENGTH),
              feature_extraction.ExtractGFCC(SAMPLE_RATE, NCEPS, NFILTS, WIN_LENGTH)]

# Creating the dataset
# This dataset performs pre-processing on the audio files
# such as resampling, resizing (to mono), normalization, and feature extraction
# Here just the MFCC transform is passed to the dataset object
audio_dataset = dataset.AudioDataset(annotations, AUDIO_FILES, transforms[5], SAMPLE_RATE, NUM_SAMPLES)

# %% Displaying each feature extraction for the first audio file in each class
if PLOTTING:
    # Creating temporary dictionaries to hold the data and the corresponding class label as a key
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
        signal, fs = librosa.load(os.path.join(AUDIO_FILES, wav_file), mono=True, sr=None)
        tmp_signals[c] = signal
        tmp_fft[c] = feature_extraction.calculate_fft(signal, fs)
        tmp_zcs[c] = feature_extraction.zero_crossing_rate(signal).T
        tmp_scs[c] = feature_extraction.spectral_centroid(y=signal, sr=fs).T
        tmp_srs[c] = feature_extraction.spectral_rolloff(y=signal+0.01, sr=fs).T
        tmp_sfs[c] = feature_extraction.spectral_flatness(y=signal).T
        tmp_fbank[c] = feature_extraction.logfbank(signal[:fs], fs, nfilt=NFILTS, nfft=WIN_LENGTH).T
        tmp_mfccs[c] = feature_extraction.mfcc(signal[:fs], fs, num_ceps=NCEPS, nfilts=NFILTS, nfft=WIN_LENGTH).T
        tmp_gfccs[c] = feature_extraction.gfcc(signal[:fs], fs, num_ceps=NCEPS, nfilts=NFILTS, nfft=WIN_LENGTH).T

    # Plotting the feature extractions of the audio
    plotting.plot_signals_time(tmp_signals)
    plotting.plot_ffts(tmp_fft)
    plotting.plot_time_feature(tmp_zcs, 'Zero Crossing Rate')
    plotting.plot_time_feature(tmp_scs, 'Spectral Centroid')
    plotting.plot_time_feature(tmp_srs, 'Spectral Rolloff')
    plotting.plot_time_feature(tmp_sfs, 'Spectral Flatness')
    plotting.plot_spectrogram(tmp_fbank, 'Filter Bank')
    plotting.plot_spectrogram(tmp_mfccs, 'MFCC')
    plotting.plot_spectrogram(tmp_gfccs, 'GFCC')

    # Deleting the temporary dictionaries
    del tmp_signals, tmp_fft, tmp_fbank, tmp_mfccs, tmp_gfccs, tmp_scs, tmp_sfs, tmp_srs

# %% Performing PCA

# Create a pandas DataFrame to hold the features
columns = audio_dataset.get_multiple_transformations_columns(transforms)
features = []
targets = []

for i in range(len(audio_dataset)):
    data, target = audio_dataset.get_multiple_transformations(i, transforms)
    features.append(data)
    targets.append(target)

feature_dataset = pd.DataFrame(np.array(features), columns=columns)
feature_dataset['target'] = targets
labels = audio_dataset.get_class_labels()

# Apply standardization to the MFCC features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply custom PCA
pca = dimentionality_reduction.PCA(-1)
features_pca = np.real(pca.fit_transform(features_scaled))
# Apply custom LDA
lda = dimentionality_reduction.LDA(-1)
features_lda = np.real(lda.fit_transform(features_scaled, targets))

# Testing other dimensionality reduction techniques
features_tsne = TSNE(n_components=3, method='barnes_hut').fit_transform(features_scaled, targets)

def get_best_dim(dr):
    # Getting the explained variance ratio
    explained_variance_ratio = dr.get_explained_variance()
    # Calculate the cumulative explained variance ratio
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    # Find the number of dimensions with the highest explained variance
    best_dimensions = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1
    return explained_variance_ratio, cumulative_explained_variance_ratio, best_dimensions

# Plot the PCA
if PLOTTING:
    # Calculate and plot the explained variance ratio for each principal component
    pca_evr, pca_cevr, pca_bd = get_best_dim(pca)
    plotting.plot_explained_variance(pca_evr, pca_cevr, pca_bd, 'PCA')

    # Plotting various components (first 3 dimensions) on a 3D scatter plot
    plotting.plot_dimentionality_reduction(features_pca, targets, labels, 'PCA')
    plotting.plot_dimentionality_reduction(features_tsne, targets, labels, 't-SNE')
    plotting.plot_dimentionality_reduction(features_lda, targets, labels, 'LDA')

# %% Training the model

