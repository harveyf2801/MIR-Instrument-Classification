import numpy as np
from scipy.spatial import distance


class PCA():
    def __init__(self, n_components=-1):
        self.n_components = n_components
        self.components = None
        self.mean = None

        self.last_eigenvalues = None
    
    def fit(self, X, y=None):
        # Calculate the mean of the input data
        self.mean = np.mean(X, axis=0)

        # Subtract the mean from the input data
        X = X - self.mean

        # Calculate the covariance matrix of the input data
        cov = np.cov(X.T)

        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        self.last_eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort the eigenvectors based on the eigenvalues
        eigenvectors = eigenvectors.T
        idxs = np.argsort(self.last_eigenvalues)[::-1]
        self.last_eigenvalues = self.last_eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Select the first n_components eigenvectors as the principal components
        self.components = eigenvectors[0:self.n_components]
    
    def fit_transform(self, X, y=None):
        # Fit the PCA model to the input data X
        self.fit(X)
        # Project the input data X onto the principal components
        return self.transform(X)
    
    def transform(self, X):
        # Subtract the mean from the input data
        X = X - self.mean

        # Project the input data onto the principal components
        return np.dot(X, self.components.T)

    def get_explained_variance(self):
        # Return the explained variance ratio of each principal component
        return self.last_eigenvalues / np.sum(self.last_eigenvalues)

class LDA():
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.linear_discriminants = None
        self.last_eigenvalues = None
    
    def fit(self, X, y):
        # Get the number of features (columns) in the input data X
        n_features = X.shape[1]
        # Get the unique class labels from the target variable y
        class_labels = np.unique(y)
        
        # Calculate the mean of each feature across all samples
        mean_overall = np.mean(X, axis=0)
        # Initialize the within-class scatter matrix S_W
        S_W = np.zeros((n_features, n_features))
        # Initialize the between-class scatter matrix S_B
        S_B = np.zeros((n_features, n_features))

        for c in class_labels:
            # Select the samples belonging to class c
            X_c = X[y == c]
            # Calculate the mean of each feature for class c
            mean_c = np.mean(X_c, axis=0)
            # Calculate the within-class scatter matrix contribution for class c
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            # Get the number of samples in class c
            n_c = X_c.shape[0]
            # Calculate the difference between the class mean and overall mean
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            # Calculate the between-class scatter matrix contribution for class c
            S_B += n_c * (mean_diff).dot(mean_diff.T)
        
        # Calculate the matrix A by multiplying the inverse of S_W with S_B
        A = np.linalg.inv(S_W).dot(S_B)
        # Compute the eigenvalues and eigenvectors of matrix A
        self.last_eigenvalues, eigenvectors = np.linalg.eig(A)
        # Transpose the eigenvectors matrix
        eigenvectors = eigenvectors.T
        # Sort the eigenvalues in descending order and get the corresponding indices
        idxs = np.argsort(abs(self.last_eigenvalues))[::-1]
        # Sort the eigenvalues based on the indices
        self.last_eigenvalues = self.last_eigenvalues[idxs]
        # Sort the eigenvectors based on the indices
        eigenvectors = eigenvectors[idxs]
        # Select the first n_components eigenvectors as the linear discriminants
        self.linear_discriminants = eigenvectors[0:self.n_components]
    
    def fit_transform(self, X, y):
        # Fit the LDA model to the input data X and target variable y
        self.fit(X, y)
        # Project the input data X onto the linear discriminants
        return self.transform(X)
    
    def transform(self, X):
        # Project the input data X onto the linear discriminants
        return np.dot(X, self.linear_discriminants.T)
    
    def get_explained_variance(self):
        # Return the explained variance ratio of each principal component
        return self.last_eigenvalues / np.sum(self.last_eigenvalues)