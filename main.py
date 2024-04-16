# Import python packages required
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchsummary import summary

import pandas as pd
import numpy as np
from scipy.io import wavfile

import librosa
from python_speech_features import logfbank, mfcc

from IPython.display import display
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
from pathlib import Path
import logging

import dataset
import plotting

# Setting global constants
# The directory paths for the audio files
AUDIO_FILES = Path(os.getcwd(), 'wavfiles')
ANNOTATIONS = Path(os.getcwd(), 'annotations.csv')
# Setting wether to run plots or not
PLOTTING = False

# Create annotations file if not already created for the dataset
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

# Defining the dataset and dataloader constants
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 128

WIN_LENGTH = 2**10
HOP_LENGTH = 2**9
NMFCC = 13 # 64?

# Creating a mel spectogram for the feature extraction / transformation
transformation = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    win_length=WIN_LENGTH,
    n_fft=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=64
)

# Defining the device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device available: {device}")

# Creating the dataset and dataloader
audio_dataset = dataset.AudioDataset(annotations, AUDIO_FILES, device, transformation, SAMPLE_RATE, NUM_SAMPLES)
data_loader = DataLoader(audio_dataset, batch_size=BATCH_SIZE)

# def spectral_features(signal):
#     spectrogram = torch.stft(signal, win_length=WIN_LENGTH, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH)
#     M = np.abs(spectrogram)
#     sc = librosa.feature.spectral_centroid(S=M)
#     sf = librosa.feature.spectral_flatness(S=M)

i = 0
for input, target in audio_dataset:
    sf = spectral_features(input)
    mfcc_mean[i, :] = np.mean(sf['mfccs'][:, 1:], axis=1)
    sc_mean[i] = np.mean(sf['spectral_centroids'])
    sf_mean[i] = np.mean(sf['spectral_flatness'])
    print(target)

    i+=1

# KNN with k-fold cross-validation
# perform k-fold cross-validation

# perform knn
n_k = 3
acc = np.zeros(c.numtestsets, 1)
pred_labels = []
true_labels = []
for k in range(c.numtestsets):
    train_data = features[c.training[k], :]
    train_ann = ann[c.training[k]]
    test_data = features[c.test[k], :]
    test_ann = ann[c.test[k]]
    training_std = zscore[train_data]
    coeff, _, _, _, explain = pca(training_std) # could also use LDA

    var_thresh = 95 # 95 percent
    cu_var = np.cumsum(explain)
    n_pc = find[cu_var >= var_thresh]
    train_data_pca = training_std * coeff[:, 0:n_pc]
    test_data_pca = (test_data - np.mean(train_data)) / np.std(train_data, axis=0) * coeff[:, 0:n_pc]

    knn_model = fitcknn(train_data_pca, train_ann, NumNeighbors=n_k)
    pred_knn = predict(knn_model, test_data_pca)
    acc[k] = np.sum(pred_knn == test_ann) / numel(test_ann)

    pred_labels = pred_labels.append(pred_knn.T)
    true_labels = true_labels.append(test_ann.T)

score = np.mean(acc)*100
C = confusionmat(true_labels, pred_labels)
confusionchart(C, class_labels)