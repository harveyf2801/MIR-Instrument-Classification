import matplotlib.pyplot as plt
import librosa
import numpy as np

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
		ax[y].set_title(list(signals.keys())[z])
		ax[y].plot(list(signals.values())[z])
		ax[y].set_xticks([])
		ax[y].set_yticks([])
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
		ax[y].set_title(list(signals.keys())[z])
		ax[y].plot(freq, Y)
		ax[y].set_xticks([])
		ax[y].set_yticks([])
		ax[y].grid(False)
		z += 1
	
	plt.tight_layout()
	plt.show()

def plot_fbanks(signals):
	'''
	Plots the Filter Bank spectrogram of the signals.

	Parameters:
		signals (dict): The input signals in a list.
	'''
	
	ncols = len(signals)
	fig, ax = plt.subplots(1, ncols, figsize=(16, 3))
	fig.suptitle("Filter Bank Coeffs", size=16)

	z = 0
	for y in range(ncols):
		ax[y].set_title(list(signals.keys())[z])
		ax[y].imshow(list(signals.values())[z],
						cmap='hot', interpolation='nearest')
		ax[y].set_xticks([])
		ax[y].set_yticks([])
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
		ax[y].set_title(list(signals.keys())[z])
		ax[y].imshow(list(signals.values())[z],
						cmap='hot', interpolation='nearest')
		ax[y].set_xticks([])
		ax[y].set_yticks([])
		ax[y].grid(False)
		z += 1
	
	plt.tight_layout()
	plt.show()

def plot_sscs(signals):
	'''
	Plots the Spectral Centroid of the signals.
	
	Parameters:
		signals (dict): The input signals in a list.
	'''
	
	ncols = len(signals)
	fig, ax = plt.subplots(1, ncols, figsize=(16, 3))
	fig.suptitle("Mel Frequency Cepstrum Coeffs", size=16)
	
	z = 0
	for y in range(ncols):
		cent = list(signals.values())[z]
		
		times = librosa.times_like(cent)
		S, phase = librosa.magphase(librosa.stft(y=y))
		librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax)
        ax[y].plot(times, cent.T, label='Spectral centroid', color='w')


		ax[y].set_title(list(signals.keys())[z])
		ax[y].imshow(list(signals.values())[z],
						cmap='hot', interpolation='nearest')
		ax[y].set_xticks([])
		ax[y].set_yticks([])
		ax[y].grid(False)
		z += 1
	
	plt.tight_layout()
	plt.show()
	



    times = librosa.times_like(cent)
    fig, ax = plt.subplots()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, cent.T, label='Spectral centroid', color='w')
    ax.legend(loc='upper right')
    ax.set(title='log Power spectrogram')