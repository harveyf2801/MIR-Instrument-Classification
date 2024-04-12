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
plotting.plot_class_distribution(annotations)
