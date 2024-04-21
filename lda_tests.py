import os
import librosa
import spafe
import python_speech_features
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import spafe.features.bfcc
import spafe.features.gfcc
import spafe.features.lfcc
import spafe.features.psrcc
import spafe.features.spfeats

import data_anaysis

import matplotlib.pyplot as plt
import spafe.features
import spafe.features.mfcc

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
    mfcc = spafe.features.mfcc.mfcc(padded_audio, fs=sr, num_ceps=13, nfilts=26, nfft=2**10)
    gfcc = spafe.features.gfcc.gfcc(padded_audio, fs=sr, num_ceps=13, nfilts=26, nfft=2**10)

    # Split filename at "-"
    label = filename.split("-")[0]
    
    # Append MFCC features and label to the list
    mfcc_features.append(np.hstack([mfcc.flatten(), gfcc.flatten()]))
    target_labels.append(label)

# Convert the list of MFCC features to a numpy array
mfcc_features = np.array(mfcc_features)

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

# Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mfcc_features_scaled, target_ids, test_size=0.2)

# Apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

tmp_Df = pd.DataFrame(X_train[:, :2], columns=['LDA Component 1','LDA Component 2'])
tmp_Df['Class']=y_train

import seaborn as sns
sns.FacetGrid(tmp_Df, hue ="Class",
              height = 6).map(plt.scatter,
                              'LDA Component 1',
                              'LDA Component 2')

plt.legend(loc='upper right')
plt.show()

# Initialize a list to store the accuracy scores for each fold
accuracy_scores = []
classifier = KNeighborsClassifier(n_neighbors=8, weights='distance')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Assume 'y_test' and 'y_pred' are already defined
accuracy = accuracy_score(y_test, y_pred)
conf_m = confusion_matrix(y_test, y_pred)

#Display the accuracy
print(f'Accuracy: {accuracy:.2f}')

#Display the confusion matrix as a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()