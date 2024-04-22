import numpy as np
from collections import Counter

def euclidean_distance(x1, x2, _):
    return np.sqrt(np.sum((x1 - x2)**2))

def hamming_distance(x1, x2, _):
    return np.sum(x1 != x2)

def manhattan_distance(x1, x2, _):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1 - x2)**p)**(1/p)

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean', p=2):
        '''
        Parameters:
            k (int): The number of nearest neighbors to consider.
            distance_metric (str): The distance metric to use when calculating the distance between two points.
        '''
        self.k = k
        self.p = p

        # Set the distance metric to use based on the input string
        if distance_metric == 'hamming':
            self.distance_metric = hamming_distance
        elif distance_metric == 'manhattan':
            self.distance_metric = manhattan_distance
        elif distance_metric == 'minkowski':
            self.distance_metric = minkowski_distance
        else:
            self.distance_metric = euclidean_distance
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        # Calculate the Euclidean distance between the input sample (x) and each training sample (x_train)
        distances = [self.distance_metric(x, x_train, self.p) for x_train in self.x_train]

        # Sort the distances in ascending order and get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Find the most common label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]