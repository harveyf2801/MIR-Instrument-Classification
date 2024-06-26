import os
from pathlib import Path
import librosa
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np

def create_file_annotations(dir):
    '''
    Creates a pandas dataframe which holds the annotations of the audio dataset.

    Returns:
    (pandas.DataFrame): The dataframe which can then be stored in a .csv file
    '''
    data = []

    # Walking through the files in the dataset
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for filename in filenames:
            if(filename[0] != '.'):
                # Extracting the filename and class label
                class_name = filename.split('-')[0]
                data.append([filename, class_name])
        break

    # Creating a pandas dataframe to hold this data
    df = pd.DataFrame(data, columns=['Filename', 'ClassLabel'])

    # Create class ID numbers from the class labels for unique classes
    df['ClassID'] = df.groupby(['ClassLabel']).ngroup()

    # Create length in seconds column
    for record in df.index:
        signal, fs = librosa.load(Path(dir, df.loc[record, 'Filename']), mono=True, sr=None)
        df.at[record, 'Length'] = signal.shape[0]/fs

    return df

class AudioDataset(Dataset):
    ''' Custom Dataset class for audio instrument classification '''

    def __init__(self, annotations, audio_dir, transformation, transform_fs, num_samples):
        '''
        Parameters -
            annotations: the annotations for the dataset
                        these annotations should have the filename at index 0 and class ID at -1
            audio_dir:       the path to the audio dataset directory
            transformation:  provides the function for performing preprocessing on the data
            transform_fs:    the samplerate to resample at
            num_samples:     the length n of samples to cut / pad the data to
        '''
        # Read in the annotation file for the dataset
        self.annotations = annotations

        # Defining attributes
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.transform_fs = transform_fs
        self.num_samples = num_samples

    def __len__(self):
            ''' Magic method to provide the length of the object '''
            return len(self.annotations)

    def __getitem__(self, index):
        ''' Magic method to provide indexing for the object '''

        signal, labelID = self._get_audio_signal(index)

        # Performs transformations on the audio signal
        features = self.transformation(signal)

        return features, labelID
    
    def _get_audio_signal(self, index):
        ''' Private method to get the audio signal for the dataset '''
        # Gets the audio path and labelID for the index
        audio_sample_path, labelID = self._get_audio_sample_path_and_label(index)

        # Loads the audio input to the device
        signal, fs = torchaudio.load(audio_sample_path, normalize=True)

        # Resamples and reshapes the audio
        signal = self._resample_audio(signal, fs)
        signal = self._convert_to_mono(signal)
        signal = self._envelope_audio(signal, self.transform_fs, 0.0005)
        signal = self._reshape_audio(signal)

        return signal, labelID

    def set_transformations(self, transformations):
        ''' Public method to set the transformations for the dataset '''
        self.transformations = transformations

    def get_class_labels(self):
        ''' Public method to provide a list of the class labels '''
        return self.annotations['ClassLabel'].unique()

    def _get_audio_sample_path_and_label(self, index):
        ''' Private get method for the sample path location
                        and prediction labelID            '''
        # label = self.annotations.loc[index, 'ClassLabel']
        labelID = self.annotations.loc[index, 'ClassID']
        path = Path(self.audio_dir, self.annotations.loc[index, 'Filename'])
        return path, labelID

    def _resample_audio(self, signal, fs):
        # Resample the audio signal if needed
        if fs != self.transform_fs:
            resampler = torchaudio.transforms.Resample(fs, self.transform_fs)
            signal = resampler(signal)
        return signal

    def _convert_to_mono(self, signal):
        # Convert the signal to mono if needed
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        return signal

    def _reshape_audio(self, signal):
        # Cut the signal if needed
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]

        # Pad the signal if needed
        if signal.shape[1] < self.num_samples:
            signal = torch.nn.functional.pad(signal, (0, self.num_samples - signal.shape[1]))

        return signal

    def _envelope_audio(self, signal, fs, threshold):
        '''
        Envelope function to define a mask where the audio goes below a threshold.

        Parameters:
            signal (numpy array): The input signal.
            threshold (float): The threshold of where the mask will activate.
        Returns:
            signal (numpy array): The enveloped signal.
        '''
        window = int(fs/16)
        
        # Applying the absolute function on the input signal
        y = np.abs(signal[0].numpy())

        # Getting the rolling average amplitude using windowing
        y_mean = np.convolve(y, np.ones(window), 'same') / window
        mask = np.array(y_mean > threshold)
        
        signal = signal[:, mask]

        return signal

    def get_multiple_transformations(self, index, transformations):
        ''' Public method to get multiple transformations for the dataset '''
        signal, labelID = self._get_audio_signal(index)
        features = np.array([])
        for transformation in transformations:
            feature = transformation(signal[0].numpy())
            if transformation.name in ('mfcc', 'gfcc'):
                features = np.hstack([features, np.mean(feature, axis=1), np.std(feature, axis=1)])
            else:
                features = np.hstack([features, np.mean(feature), np.std(feature)])
        
        return features, labelID
    
    def get_multiple_transformations_columns(self, transformations):
        ''' Public method to get the columns for multiple transformations '''
        columns = []
        for transformation in transformations:
            if transformation.name in ('mfcc', 'gfcc'):
                columns.extend([f'{transformation.name}_{i}_mean' for i in range(1, 14)])
                columns.extend([f'{transformation.name}_{i}_std' for i in range(1, 14)])
            else:
                columns.extend([f'{transformation.name}_mean', f'{transformation.name}_std'])
        
        return columns