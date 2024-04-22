import os
import librosa
import pandas as pd
import numpy as np

import feature_extraction

# Path to the folder containing audio files
audio_folder = "wavfiles"

# # List to store MFCC features
# features = []
# target_labels = []
# columns = []

# NCEPS = 13

# for f in ['rms', 'spf', 'spc', 'spr', 'zcr']:
#     for type in ['mean', 'std']:
#         columns.append(f"{f}_{type}")

# for f in ['mfcc', 'gfcc']:
#     for type in ['mean', 'std']:
#         for cep in range(NCEPS):
#             columns.append(f"{f}_{cep}_{type}")

# # Iterate over each audio file in the folder
# for filename in os.listdir(audio_folder):
#   if filename.endswith(".wav"):
#     file_path = os.path.join(audio_folder, filename)
#     # Load audio file
#     audio, sr = librosa.load(file_path, sr=22500)

#     # Pad audio signal to a fixed length
#     fixed_length = sr*2  # Assuming a fixed length of 1 second (44100 samples at 44.1 kHz)
#     padded_audio = librosa.util.fix_length(audio, size=fixed_length)

#     # Extract MFCC features
#     rms = feature_extraction.rms(y=padded_audio)
#     spf = feature_extraction.spectral_flatness(y=padded_audio, n_fft=2**10).T
#     spc = feature_extraction.spectral_centroid(y=padded_audio, sr=sr, n_fft=2**10).T
#     spr = feature_extraction.spectral_rolloff(y=padded_audio+0.01, sr=sr, n_fft=2**10).T
#     zcr = feature_extraction.zero_crossing_rate(padded_audio)
#     mfcc = feature_extraction.mfcc(padded_audio, fs=sr, num_ceps=13, nfilts=26, nfft=2**10)
#     gfcc = feature_extraction.gfcc(padded_audio, fs=sr, num_ceps=13, nfilts=26, nfft=2**10)

#     # Split filename at "-"
#     label = filename.split("-")[0]
    
#     # Append MFCC features and label to the list
#     features.append(np.hstack([rms.mean(), rms.std(),
#                                 spf.mean(), spf.std(),
#                                 spc.mean(), spc.std(),
#                                 spr.mean(), spr.std(),
#                                 zcr.mean(), zcr.std(),
#                                 mfcc.mean(axis=0), mfcc.std(axis=0),
#                                 gfcc.mean(axis=0), gfcc.std(axis=0)]))
#     target_labels.append(label)

# # Convert the list of MFCC features to a numpy array
# features = np.array(features)

# feature_df = pd.DataFrame(features, columns=columns)
# feature_df['target'] = target_labels

# print(feature_df.head())

from dataset import AudioDataset, create_file_annotations

annotations = create_file_annotations(audio_folder)
fs = 22500

transforms = [feature_extraction.ExtractRMS(),
              feature_extraction.ExtractSpectralFlatness(fs, 2**10),
              feature_extraction.ExtractSpectralCentroid(fs, 2**10),
              feature_extraction.ExtractSpectralRolloff(fs, 2**10),
              feature_extraction.ExtractZeroCrossingRate(),
              feature_extraction.ExtractMFCC(fs, 13, 26, 2**10),
              feature_extraction.ExtractGFCC(fs, 13, 26, 2**10)]

audio_dataset = AudioDataset(annotations, audio_folder, transforms[0], fs, 2*fs)

columns = audio_dataset.get_multiple_transformations_columns(transforms)

features = []
targets = []

for i in range(len(audio_dataset)):
    data, target = audio_dataset.get_multiple_transformations(i, transforms)
    features.append(data)
    targets.append(target)

feature_dataset = pd.DataFrame(np.array(features), columns=columns)
feature_dataset['target'] = targets

print(feature_dataset.head())