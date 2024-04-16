import numpy as np
import scipy.fftpack as fft
import pandas as pd

def envelope(y, fs, threshold):
	'''
	Envelope function to define a mask where the audio goes below a threshold.

	Parameters:
		y (numpy array): The input signal.
		fs (int): The sampling rate of the signal.
		threshold (float): The threshold of where the mask will activate.
	Returns:
		(list): A mask of boolean values.
	'''
	mask = []
	# Applying the absolute function on the input signal
	y = pd.Series(y).apply(np.abs)
	# Getting the rolling average amplitude using windowing
	y_mean = y.rolling(window=int(fs/16), min_periods=1, center=True).mean()
	for mean in y_mean:
		if mean > threshold:
			mask.append(True)
		else:
			mask.append(False)
	return mask

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