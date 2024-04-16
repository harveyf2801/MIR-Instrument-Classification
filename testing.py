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
    from python_speech_features import logfbank, mfcc, ssc

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

# Setting wether to run plots or not
PLOTTING = False

# %% Cleaning the audio files by stripping the silence
clean_files = os.listdir(C_AUDIO_FILES)
logging.info("Cleaning the audio files")
for file in tqdm(clean_files):
	# Ignoring files already cleaned and all the 'hidden' files (such as .DS_store used for MacOS)
	if (file in clean_files) or (file[0] == '.'):
		continue
	# Loading the audio file at 16000 Hz
	signal, fs = librosa.load(os.path.join(AUDIO_FILES, file), mono=True, sr=16000)
	# Getting the mask envelope for the audio
	mask = feature_extraction.envelope(signal, fs, 0.0005)
	# Writing this masked and resampled audio into the clean directory
	wavfile.write(filename=os.path.join(C_AUDIO_FILES, file), rate=fs, data=signal[mask])

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
# Defining the dataset and dataloader constants
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 128

WIN_LENGTH = 2**10
HOP_LENGTH = 2**9
NMFCC = 13 # 64?

# Creating a mel spectogram for the feature extraction / transformation
transformations = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        win_length=WIN_LENGTH,
        n_fft=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=64
    ),
    torchaudio.transforms.SpectralCentroid(
        sample_rate=SAMPLE_RATE,
        n_fft=WIN_LENGTH,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window_fn=torch.hann_window
    )
)
scripted_transforms = torch.jit.script(transformations)

# Defining the device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device available: {str(device).upper()}")

# Creating the dataset and dataloader
audio_dataset = dataset.AudioDataset(annotations, AUDIO_FILES, device, scripted_transforms, SAMPLE_RATE, NUM_SAMPLES)
data_loader = DataLoader(audio_dataset, batch_size=BATCH_SIZE)

print(audio_dataset[0])

# %% 

# Creating dictionaries to hold the data and the corresponding class label as a key
signals = {}
fft = {}
fbank = {}
mfccs = {}
sscs = {}

# Getting all the unique class labels
classes = list(np.unique(annotations.ClassLabel))

# Iterating through the classes and selecting the first signal from each to extract features
for c in classes:
  wav_file = annotations[annotations.ClassLabel == c].iloc[0, 0]
  signal, fs = librosa.load(os.path.join(C_AUDIO_FILES, wav_file), mono=True, sr=None)
  signals[c] = signal
  fft[c] = feature_extraction.calculate_fft(signal, fs)
  fbank[c] = logfbank(signal[:fs], fs, nfilt=26, nfft=1103).T
  mfccs[c] = mfcc(signal[:fs], fs, numcep=13, nfilt=26, nfft=1103).T
  sscs[c] = librosa.feature.spectral_centroid(y=signal[:fs], sr=fs, n_fft=1103)

# Plotting the feature extractions of the audio
plotting.plot_signals_time(signals)
plt.show()

plotting.plot_ffts(fft)
plt.show()

plotting.plot_fbanks(fbank)
plt.show()

plotting.plot_mfccs(mfccs)
plt.show()

plotting.plot_fbanks(sscs)
plt.show()

print(sscs['cello'][0].shape)

# %%
