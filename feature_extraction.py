import numpy as np
import torch
import pandas as pd

from librosa.feature import spectral_flatness, spectral_centroid, spectral_rolloff, zero_crossing_rate
from spafe.features.gfcc import gfcc
from python_speech_features import logfbank, mfcc

def calculate_fft(y, fs):
	'''
	Calculates the Fast Fourier Transform of a signal.

	Parameters:
		y (numpy array): The input signal.
		fs (int): The sampling rate of the signal.
	Return:
		(tuple): The magnitude values and frequency bins (1/sample rate).
	'''
	n = len(y)
	freqs = np.fft.rfftfreq(n, d=1/fs)
	Y = abs(np.fft.rfft(y)/n)
	return (Y, freqs)

# Torch transformation features

class ExtractMFCC(object):
    """Extract MFCC feature.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    name = 'mfcc'

    def __init__(self, fs, numcep, nfilt, nfft):
        self.fs = fs
        self.numcep = numcep
        self.nfilt = nfilt
        self.nfft = nfft

    def __call__(self, signal):
        feature = mfcc(signal, self.fs, numcep=self.numcep, nfilt=self.nfilt, nfft=self.nfft).T
        return feature

class ExtractGFCC(object):
    """Extract GFCC feature.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    name = 'gfcc'

    def __init__(self, fs, numcep, nfilt, nfft):
        self.fs = fs
        self.numcep = numcep
        self.nfilt = nfilt
        self.nfft = nfft

    def __call__(self, signal):
        feature = gfcc(signal, self.fs, num_ceps=self.numcep, nfilts=self.nfilt, nfft=self.nfft).T
        return feature


class ExtractSpectralCentroid(object):
    """Extract Spectral Centroid feature.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    name = 'spectral_centroid'

    def __init__(self, fs, nfft):
        self.fs = fs
        self.nfft = nfft

    def __call__(self, signal):
        feature = spectral_centroid(y=signal, sr=self.fs, n_fft=self.nfft).T
        return feature


class ExtractSpectralRolloff(object):
    """Extract Spectral Rolloff feature.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    name = 'spectral_rolloff'

    def __init__(self, fs, nfft):
        self.fs = fs
        self.nfft = nfft

    def __call__(self, signal):
        feature = spectral_rolloff(y=signal+0.01, sr=self.fs, n_fft=self.nfft).T
        return feature


class ExtractSpectralFlatness(object):
    """Extract Spectral Flatness feature.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    name = 'spectral_flatness'

    def __init__(self, fs, nfft):
        self.fs = fs
        self.nfft = nfft

    def __call__(self, signal):
        feature = spectral_flatness(y=signal, n_fft=self.nfft).T
        return feature

class ExtractFbank(object):
    """Extract Filter Bank (Fbank) feature.

    Args:
        fs (int): The sampling rate of the signal.
        n_fft (int): The number of FFT points.
        nfilt (int): The number of filter banks.
    """
    name = 'fbank'

    def __init__(self, fs, nfilt, nfft):
        self.fs = fs
        self.nfilt = nfilt
        self.nfft = nfft

    def __call__(self, signal):
        feature = logfbank(signal, self.fs, nfilt=self.nfilt, nfft=self.nfft).T
        return feature

def export_features(dataset, feature):
    """
    Export the features.

    Parameters:
        dataset (AudioDataset): The dataset containing the audio files.
        feature (feature_extraction class): The feature extracted from the audio files.
    """

    # Create a pandas DataFrame to hold the features
    features_df = pd.DataFrame()

    # Iterate through the dataset and calculate the mean of each feature
    dataset.set_transformations(feature)

    for i in range(int(len(dataset))):
        data, target = dataset[i]

        # Calculate the mean along axis 1
        mean_feature = np.mean(data.numpy(), axis=1)
        # Calculate the standard deviation of the data
        std_feature = np.std(data.numpy(), axis=1)

        # COULD FLATTEN THE DATA INSTEAD OF TAKING THE MEAN
        # flatten_feature = data.numpy().flatten()
        
        # Add the mean feature to the DataFrame
        for j, coeff in enumerate(mean_feature):
            features_df.at[i, f'{feature.name} avg {j+1}'] = coeff

        # Add the standard deviation to the DataFrame
        for j, coeff in enumerate(std_feature):
            features_df.at[i, f'{feature.name} std {j+1}'] = coeff

        # Add the target to the DataFrame
        features_df.at[i, 'target'] = target

    return features_df