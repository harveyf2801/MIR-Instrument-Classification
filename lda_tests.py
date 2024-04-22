import os
import librosa
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import feature_extraction

import dimentionality_reduction

import matplotlib.pyplot as plt

# Path to the folder containing audio files
audio_folder = "wavfiles"

# List to store MFCC features
mfcc_features = []
target_labels = []
target_ids = []

# Iterate over each audio file in the folder
for filename in os.listdir(audio_folder):
  if filename.endswith(".wav"):
    file_path = os.path.join(audio_folder, filename)
    # Load audio file
    audio, sr = librosa.load(file_path, sr=22500)

    # Pad audio signal to a fixed length
    fixed_length = sr*2  # Assuming a fixed length of 1 second (44100 samples at 44.1 kHz)
    padded_audio = librosa.util.fix_length(audio, size=fixed_length)

    # Extract MFCC features
    
    spc = feature_extraction.spectral_centroid(y=padded_audio, sr=sr).T
    spr = feature_extraction.spectral_rolloff(y=padded_audio+0.01, sr=sr).T
    zcr = feature_extraction.zero_crossing_rate(padded_audio)
    mfcc = feature_extraction.mfcc(padded_audio, fs=sr, num_ceps=13, nfilts=26, nfft=2**10)
    gfcc = feature_extraction.gfcc(padded_audio, fs=sr, num_ceps=13, nfilts=26, nfft=2**10)

    # Split filename at "-"
    label = filename.split("-")[0]
    
    # Append MFCC features and label to the list
    mfcc_features.append(np.hstack([spr.mean(), spr.std(),
                                    zcr.mean(), zcr.std(),
                                    mfcc.mean(axis=0), mfcc.std(axis=0),
                                    gfcc.mean(axis=0), gfcc.std(axis=0)]))
    target_labels.append(label)

# Convert the list of MFCC features to a numpy array
mfcc_features = np.array(mfcc_features)

print('shape of mfcc_features:', mfcc_features.shape)

# Create target IDs based on target labels
unique_labels = np.unique(target_labels)
target_ids = np.arange(len(unique_labels))

# Map target labels to target IDs
target_id_mapping = dict(zip(unique_labels, target_ids))
target_ids = np.array([target_id_mapping[label] for label in target_labels])

# Apply PCA
# Apply standardization to the MFCC features
scaler = StandardScaler()
mfcc_features_scaled = scaler.fit_transform(mfcc_features, target_ids)


# Apply Linear Discriminant Analysis
import dimentionality_reduction 

def get_score(model, X_train, X_test, y_train, y_test):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  return accuracy, y_pred


# Initialize a list to store the accuracy scores for each fold
from models import KNNClassifier

k_folds = 7
kf = StratifiedKFold(n_splits=k_folds)
scores = {
  'knn': [],
  'rf': [],
  'svc': []
}

classifiers = {
  'knn': KNNClassifier(10),
  'rf': RandomForestClassifier(n_estimators=40),
  'svc': SVC(gamma=1.0)
}

for train_index, test_index in kf.split(mfcc_features_scaled, target_ids):
  X_train, X_test = mfcc_features_scaled[train_index], mfcc_features_scaled[test_index]
  y_train, y_test = target_ids[train_index], target_ids[test_index]

  lda = dimentionality_reduction.LDA()
  X_train = np.real(lda.fit_transform(X_train, y_train))
  X_test = np.real(lda.transform(X_test))

  for name, classifier in classifiers.items():
    accuracy, y_pred = get_score(classifier, X_train, X_test, y_train, y_test)
    scores[name].append(accuracy)

for key, value in scores.items():
  print(f"{key} accuracy: {value}")

tmp_Df = pd.DataFrame(np.real(X_train[:, :2]), columns=['LDA Component 1','LDA Component 2'])
tmp_Df['Class']=y_train

import seaborn as sns
sns.FacetGrid(tmp_Df, hue ="Class",
              height = 6).map(plt.scatter,
                              'LDA Component 1',
                              'LDA Component 2')

plt.legend(loc='upper right')
plt.show()

# conf_m = confusion_matrix(y_test, y_pred)

# #Display the confusion matrix as a heatmap
# plt.figure(figsize=(6, 6))
# sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()