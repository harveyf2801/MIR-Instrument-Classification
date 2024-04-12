import numpy as np
from scipy.signal import hamming
import scipy.fftpack as fft

def calculate_mfcc(audio_signal, sample_rate, numcep=13, frame_size=0.025,
                   frame_stride=0.01, nfilt=26, nfft=512, low_freq=0, high_freq=None):
    '''
    Calculates the MFCCs from an audio signal

    Parameters:
        audio_signal (numpy array): Audio signal from which to compute features
        sample_rate (int): The sample rate of the audio signal
        numcep (int, optional): The number of MFCCs to return. Defaults to 13.
        frame_size (float, optional): The size of the frame. Defaults to 0.025.
        frame_stride (float, optional): The stride of the frame. Defaults to 0.01.
        nfilt (int, optional): The number of filters. Defaults to 26.
        nfft (int, optional): The number of FFT points. Defaults to 512.
        low_freq (int, optional): The low frequency. Defaults to 0.
        high_freq (int, optional): The high frequency. If None, it will be set to the sample_rate / 2.

    Returns:
        numpy array: The calculated MFCCs
    '''
    # Pre-emphasis factor of 0.97 to balance the frequency spectrum and improve the signal to noise ratio
    pre_emphasis = 0.97
    # Apply pre-emphasis filter on the signal
    emphasized_signal = np.append(audio_signal[0], audio_signal[1:] - pre_emphasis * audio_signal[:-1])

    # Framing parameters
    frame_length = int(round(frame_size * sample_rate))  # Frame size in samples
    frame_step = int(round(frame_stride * sample_rate))  # Frame stride in samples
    signal_length = len(emphasized_signal)  # Length of the signal
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Number of frames

    # Padding the signal to make sure all frames have equal number of samples without truncating any samples from the original signal
    padded_signal_length = num_frames * frame_step + frame_length
    padded_signal = np.pad(emphasized_signal, (0, padded_signal_length - signal_length), 'constant')

    # Creating an array of frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = padded_signal[indices.astype(np.int32, copy=False)]

    # Windowing
    frames *= hamming(frame_length)  # Applying hamming window to each frame

    # Fourier Transform and Power Spectrum
    magnitude_spectrum = np.abs(fft.fft(frames, n=nfft))  # Computing the magnitude spectrum of the frames
    power_spectrum = (1.0 / frame_length) * np.square(magnitude_spectrum)  # Computing the power spectrum of the frames

    # Mel Filterbank
    if high_freq is None:  # If high frequency is not provided, consider it as half the sampling rate
        high_freq = sample_rate / 2

    # Converting frequencies to Mel scale
    mel_points = np.linspace(hz_to_mel(low_freq), hz_to_mel(high_freq), nfilt + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    # Creating filter banks
    filter_bank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin_points[m - 1])  # Lower frequency
        f_m = int(bin_points[m])  # Center frequency
        f_m_plus = int(bin_points[m + 1])  # Upper frequency

        # Updating filter banks
        filter_bank[m - 1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        filter_bank[m - 1, f_m:f_m_plus] = (bin_points[m + 1] - np.arange(f_m, f_m_plus)) / (bin_points[m + 1] - bin_points[m])

    # Applying Mel filter banks to the power spectrum
    filtered_spectrum = np.dot(power_spectrum, filter_bank.T)
    filtered_spectrum = np.where(filtered_spectrum == 0, np.finfo(float).eps, filtered_spectrum)  # Replacing zero values with a small constant

    # Logarithm
    log_spectrum = 10 * np.log10(filtered_spectrum)  # Applying logarithm to the filtered spectrum

    # Discrete Cosine Transform (DCT)
    dct_matrix = dct_matrix = fft.dct(np.eye(nfilt))[:numcep]  # Creating DCT matrix
    mfcc = np.dot(log_spectrum, dct_matrix.T)  # Applying DCT to the log spectrum

    return mfcc  # Returning MFCCs

def hz_to_mel(hz):
    '''
    Converts frequency from Hz to Mel scale
    '''
    return 2595 * np.log10(1 + hz / 700)  # Formula to convert Hz to Mel

def mel_to_hz(mel):
    '''
    Converts frequency from Mel scale to Hz
    '''
    return 700 * (10**(mel / 2595) - 1)  # Formula to convert Mel to Hz