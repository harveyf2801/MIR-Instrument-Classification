import numpy as np
from collections import Counter
import torch
import torch.nn as nn

# Distance metrics

def euclidean_distance(x1, x2, _):
    return np.sqrt(np.sum((x1 - x2)**2))

def hamming_distance(x1, x2, _):
    return np.sum(x1 != x2)

def manhattan_distance(x1, x2, _):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1 - x2)**p)**(1/p)

class KNNClassifier:
    '''
    K-Nearest Neighbors Classifier.
    '''
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

class SVC():
    '''
    Support Vector Classifier (SVC) using the gradient descent algorithm.
    Kernel function: Linear kernel
    '''
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        '''
        Parameters:
            learning_rate (float): The step size that will be taken when moving towards a minimum of the loss function.
            no_of_iterations (int): The number of iterations that the gradient descent algorithm will run for.
            lambda_parameter (float): The regularization parameter that will be used in the loss function.
        '''
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):
        self.m, self.n = X.shape

        # Setting initial values for the weights as 0
        self.w = np.zeros(self.n)
        self.b = 0

        self.X = X
        self.Y = Y

        # Gradient Descent algorithm to optimise model
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        # Converting labels to {-1, 1} (label encoding)
        y_label = np.where(self.Y <= 0, -1, 1)

        # Performing the gradient descent update
        for index, x_i in enumerate(self.X):
            # Condition for misclassification
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

            # Updating weights
            if (condition == True):
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            else:
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def predict(self, X):
        # Making predictions
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        # Converting {-1, 1} back to {0, 1} (label decoding)
        y_hat = np.where(predicted_labels <= -1, 0, 1)
        return y_hat

class CNNNetwork(nn.Module):
  ''' Custom Convolution Neural Network for audio classification'''

  def __init__(self, num_classes):
    '''
    Parameters -
        num_classes: the number of classes used for classification
    '''
    super().__init__()

    # Define convolutional blocks
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    # Create a flatten layer to reshape the data into a 1D vector
    self.flatten = nn.Flatten()

    # Create a linear / dense layer to classify the data from the convolutional network
    self.linear = nn.Linear(in_features=128*5*4, out_features=num_classes)

    # Create a softmax layer to provide decimal probabilities to the class predictions
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_data):
    ''' Passing the data through the network '''
    x = self.conv1(input_data)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.flatten(x)
    logits = self.linear(x)
    predictions = self.softmax(logits)
    return predictions