import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        # Calculate the Euclidean distance between the input sample (x) and each training sample (x_train)
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]

        # Sort the distances in ascending order and get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Find the most common label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]