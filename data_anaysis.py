import numpy as np
from scipy.spatial import distance

def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    explained = (l / np.sum(l))[:no_dims]
    return Y, explained

def perform_PCA(X, n_components=-1):
    # Center the data
    X_centered = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    # sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top n_components eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    
    # Project the data onto the selected eigenvectors
    X_pca = np.dot(X_centered, selected_eigenvectors)
    
    return X_pca

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