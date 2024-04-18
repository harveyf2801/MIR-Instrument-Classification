import numpy as np
import scipy.fftpack as fft
import pandas as pd

from librosa.feature import spectral_flatness, spectral_centroid, spectral_rolloff
from spafe.features.gfcc import gfcc
from python_speech_features import logfbank, mfcc, ssc

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

    def __init__(self, fs, numcep, nfilt, nfft):
        self.fs = fs
        self.numcep = numcep
        self.nfilt = nfilt
        self.nfft = nfft

    def __call__(self, signal):
        feature = mfcc(signal, self.fs, numcep=self.numcep, nfilt=self.nfilt, nfft=self.nfft)

        return feature