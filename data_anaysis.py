import numpy as np
from scipy.spatial import distance

def pca(X=np.array([]), no_dims=2):
    """
        Runs PCA on the numpy array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    # Get the shape of the input array X and assign the number of rows to n and the number of columns to d
    (n, d) = X.shape

    # Subtract the mean of each column from the corresponding elements in X
    X = X - np.tile(np.mean(X, 0), (n, 1))

    # Compute the eigenvalues (l) and eigenvectors (M) of the covariance matrix of X
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    
    # Project the centred data onto the selected eigenvectors to obtain the reduced-dimensional representation
    Y = np.dot(X, M[:, 0:no_dims])
    
    # Compute the explained variance ratio of each selected dimension
    explained = (l / np.sum(l))[:no_dims]
    return np.real(Y), explained

def kNN(X_train, y_train, X_test, k):
    predictions = []
    
    for test_sample in X_test:
        distances = []
        
        for train_sample in X_train:
            dist = distance.euclidean(test_sample, train_sample)
            distances.append(dist)
        
        sorted_indices = np.argsort(distances)
        k_nearest_labels = [y_train[i] for i in sorted_indices[:k]]
        
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        
        predictions.append(predicted_label)
    
    return predictions