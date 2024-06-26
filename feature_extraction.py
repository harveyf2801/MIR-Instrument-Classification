import numpy as np
import torch
import pandas as pd

from librosa.feature import spectral_flatness, spectral_centroid, spectral_rolloff, zero_crossing_rate, rms
from spafe.features.gfcc import gfcc
# from spafe.features.mfcc import mfcc
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

class ExtractRMS(object):
    """Extract Root Mean Square (RMS) feature.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    name = 'rms'

    def __call__(self, signal):
        feature = rms(y=signal)
        return feature

class ExtractZeroCrossingRate(object):
    """Extract Zero Crossing Rate (ZCR) feature.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    name = 'zcr'

    def __call__(self, signal):
        feature = zero_crossing_rate(signal).T
        return feature
    
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