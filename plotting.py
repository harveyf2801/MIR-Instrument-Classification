import matplotlib.pyplot as plt
import librosa
import numpy as np
from sklearn import preprocessing

def plot_class_distribution(annotations):
    # Getting the class distribution
    class_dist = annotations.groupby(['ClassLabel'])['Length'].mean()

    # Plotting the distribution
    fig, ax = plt.subplots(num='Class Distribution')
    fig.suptitle('Class Distribution')
    # ax.set_title('Class Distribution', y=1.08)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
    ax.axis('equal')
    plt.show()
    annotations.reset_index(inplace=True)

def plot_signals_time(signals):
	'''
	Plots the time domain of the signals.

	Parameters:
		signals (dict): The input signals in a list.
	'''
	
	ncols = len(signals)
	fig, ax = plt.subplots(1, ncols, figsize=(16, 3))
	fig.suptitle("Time Series", size=16)
	z = 0
	for y in range(ncols):
		ax[y].set_title((list(signals.keys())[z]).capitalize())
		ax[y].plot(list(signals.values())[z])
		ax[y].set_xticks([])
		ax[y].set_yticks([])
		ax[y].grid(False)
		ax[y].set_xlabel('Time')
		ax[y].set_ylabel('Amplitude')
		z += 1
	
	plt.tight_layout()
	plt.show()

def normalize(x, axis=0):
	# Function that normalizes the Sound Data
    return preprocessing.minmax_scale(x, axis=axis)

def plot_spectral_feature(signals, title):
	'''
	Plots spectral features of the signals.
	Such as spectral centroid, spectral flatness, and spectral roll-off.

	Parameters:
		signals (dict): The input signals in a list.
		title (str): The title of the spectral feature i.e. 'flatness' or 'centroid' etc.
	'''

	ncols = len(signals)
	fig, ax = plt.subplots(1, ncols, figsize=(16, 3))
	fig.suptitle(f'Spectral {title}', size=16)

	z = 0
	for y in range(ncols):
		cent = list(signals.values())[z]

		# Computing the time variable for visualization
		frames = range(len(cent))
		# Converts frame counts to time (seconds)
		t = librosa.frames_to_time(frames)

		ax[y].set_title((list(signals.keys())[z]).capitalize())
		ax[y].plot(normalize(cent))
		ax[y].set_xticks([])
		ax[y].set_yticks([])
		ax[y].set_xlabel('Time')
		ax[y].set_ylabel(title)
		ax[y].grid(False)
		z += 1

	plt.tight_layout()
	plt.show()

def plot_ffts(signals):
	'''
	Plots the Fast Fourier Transform of the signals.

	Parameters:
		signals (dict): The input signals in a list.
	'''
	
	ncols = len(signals)
	fig, ax = plt.subplots(1, ncols, figsize=(16, 3))
	fig.suptitle("Fourier Transform", size=16)

	z = 0
	for y in range(ncols):
		data = list(signals.values())[z]
		Y, freq = data[0], data[1]
		ax[y].set_title((list(signals.keys())[z]).capitalize())
		ax[y].plot(freq, Y)
		ax[y].set_xticks([])
		ax[y].set_yticks([])
		ax[y].set_xlabel('Frequency')
		ax[y].set_ylabel('Magnitude')
		ax[y].grid(False)
		z += 1
	
	plt.tight_layout()
	plt.show()

def plot_spectrogram(signals, title):
	'''
	Plots the Spectrogram of the signals.

	Parameters:
		signals (dict): The input signals in a list.
		title (str): The title of the feature i.e. 'Filter Bank' or 'MFCC's' etc.
	'''
	
	ncols = len(signals)
	fig, ax = plt.subplots(1, ncols, figsize=(16, 3))
	fig.suptitle(title, size=16)

	z = 0
	for y in range(ncols):
		ax[y].set_title((list(signals.keys())[z]).capitalize())
		ax[y].imshow(list(signals.values())[z],
						cmap='hot', interpolation='nearest')
		ax[y].set_xticks([])
		ax[y].set_yticks([])
		ax[y].set_xlabel('Time')
		ax[y].set_ylabel('Frequency')
		ax[y].grid(False)
		z += 1
	
	plt.tight_layout()
	plt.show()

def plot_mfccs(signals):
	'''
	Plots the Mel Frequency Cepstrum of the signals.
	
	Parameters:
		signals (dict): The input signals in a list.
	'''
	
	ncols = len(signals)
	fig, ax = plt.subplots(1, ncols, figsize=(16, 3))
	fig.suptitle("Mel Frequency Cepstrum Coeffs", size=16)
	
	z = 0
	for y in range(ncols):
		ax[y].set_title((list(signals.keys())[z]).capitalize())
		ax[y].imshow(list(signals.values())[z],
						cmap='hot', interpolation='nearest')
		ax[y].set_xticks([])
		ax[y].set_yticks([])
		ax[y].grid(False)
		z += 1
	
	plt.tight_layout()
	plt.show()