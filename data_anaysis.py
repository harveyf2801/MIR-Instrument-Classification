import numpy as np
from scipy.spatial import distance

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