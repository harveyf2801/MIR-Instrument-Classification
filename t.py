import os
import librosa
import spafe
import python_speech_features
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
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
    mfcc_features.append(gfcc.flatten())
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
mfcc_features_scaled = scaler.fit_transform(mfcc_features)

# Apply PCA
pca = PCA()
mfcc_features_pca = pca.fit_transform(mfcc_features_scaled, target_ids)
# mfcc_features_pca, explained = data_anaysis.pca(mfcc_features_scaled, 2)
# mfcc_features_pca = np.real(mfcc_features_pca)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mfcc_features_pca, target_ids, test_size=0.2, random_state=42)

# Define the number of folds for cross-validation
n_folds = 5

# Initialize a list to store the accuracy scores for each fold
accuracy_scores = []

# Perform k-fold cross-validation
kf = KFold(n_splits=n_folds)
for train_index, val_index in kf.split(X_train):
  # Split the data into training and validation sets for the current fold
  X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
  y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
  
  # Initialize a KNN classifier
  knn = KNeighborsClassifier(n_neighbors=3)
  
  # Fit the classifier to the training data
  knn.fit(X_train_fold, y_train_fold)
  
  # Predict the labels for the validation data
  y_val_pred = knn.predict(X_val_fold)
  
  # Calculate the accuracy score for the current fold
  accuracy = accuracy_score(y_val_fold, y_val_pred)
  
  # Append the accuracy score to the list
  accuracy_scores.append(accuracy)

# Calculate the average accuracy score across all folds
average_accuracy = np.mean(accuracy_scores)

# Print the average accuracy score
print("Average Accuracy:", average_accuracy)

# # Calculate the explained variance ratio for each principal component
# explained_variance_ratio = pca.explained_variance_ratio_

# # Calculate the cumulative explained variance ratio
# cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

# # Find the number of dimensions with the highest explained variance
# best_dimensions = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1

# # Plot the explained variance ratio
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance_ratio, 'b-')
# plt.axvline(x=best_dimensions, color='r', linestyle='--')
# plt.xlabel('Number of Dimensions')
# plt.ylabel('Cumulative Explained Variance Ratio')
# plt.title('Explained Variance Ratio')
# plt.grid(True)
# plt.show()

# Plot the PCA with the best dimensions
plt.figure(figsize=(10, 8))
for label in unique_labels:
  indices = np.where(np.array(target_labels) == label)
  plt.scatter(mfcc_features_pca[indices, 1], mfcc_features_pca[indices, 0], label=label)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA with Best Dimensions')
plt.legend()
plt.show()