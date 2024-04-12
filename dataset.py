import os
import librosa
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

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
    df.set_index('Filename', inplace=True)
    for record in df.index:
        signal, fs = librosa.load(os.path.join(dir, record), mono=True, sr=None)
        df.at[record, 'Length'] = signal.shape[0]/fs

    return df

class AudioDataset(Dataset):
  ''' Custom Dataset class for audio instrument classification '''

  def __init__(self, annotation_file, audio_dir, device, transformation, transform_fs, num_samples):
    '''
    Parameters -
        annotation_file: a path to a csv file with the annotations for the dataset
                            these annotations should have the filename at index 0 and class ID at -1
        audio_dir:       the path to the audio dataset directory
        device:          the device in use (cuda or cpu)
        transformation:  provides the function for performing preprocessing on the data
        transform_fs:    the samplerate to resample at
        num_samples:     the length n of samples to cut / pad the data to
    '''
    # Read in the annotation file for the dataset
    self.annotations = pd.read_csv(annotation_file)

    # Defining attributes
    self.audio_dir = audio_dir
    self.device = device
    self.transformation = transformation.to(self.device) # putting the data onto a cuda device is available
    self.transform_fs = transform_fs
    self.num_samples = num_samples

    # Creating a unique ID for the classes
    # self.annotations.assign(id=self.annotations.groupby(['ClassLabel']).ngroup())

  def __len__(self):
    ''' Magic method to provide the length of the object '''
    return len(self.annotations)

  def __getitem__(self, index):
    ''' Magic method to provide indexing for the object '''

    # Gets the audio path and labelID for the index
    audio_sample_path, labelID = self._get_audio_sample_path_and_label(index)

    # Loads the audio input to the device
    signal, fs = torchaudio.load(audio_sample_path, normalize=True)
    signal = signal.to(self.device)

    # Resamples and reshapes the audio
    signal = self._resample_audio(signal, fs)
    signal = self._reshape_audio(signal)

    # Performs transformation on the device
    signal = self.transformation(signal)

    return signal, labelID

  def get_class_labels(self):
    ''' Public method to provide a list of the class labels '''
    return self.annotations['ClassLabel'].unique()

  def _get_audio_sample_path_and_label(self, index):
      ''' Private get method for the sample path location
                        and prediction labelID            '''
      label = self.annotations.loc[index, 'ClassLabel']
      labelID = self.annotations.loc[index, 'ClassID']
      path = os.path.join(self.audio_dir, label, self.annotations.loc[index, 'Filename'])
      return path, labelID

  def _resample_audio(self, signal, fs):
      # Resample the audio signal if needed
      if fs != self.transform_fs:
          resampler = torchaudio.transforms.Resample(fs, self.transform_fs).to(self.device)
          signal = resampler(signal)
      return signal

  def _reshape_audio(self, signal):
      # Convert the signal to mono if needed
      if signal.shape[0] > 1:
          signal = torch.mean(signal, dim=0, keepdim=True)

      # Cut the signal if needed
      if signal.shape[1] > self.num_samples:
          signal = signal[:, :self.num_samples]

      # Pad the signal if needed
      if signal.shape[1] < self.num_samples:
          signal = torch.nn.functional.pad(signal, (0, self.num_samples - signal.shape[1]))

      return signal

