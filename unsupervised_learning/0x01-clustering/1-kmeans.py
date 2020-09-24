#!/usr/bin/env python3
"""
0. Initialize K-means
"""

import numpy as np


def initialize(X, k):
    """ Initializes cluster centroids for K-means
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset that
            will be used for K-means clustering
            n is the number of data points
            d is the number of dimensions for each data point
        k is a positive integer containing the number of clusters
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """

    n, d = X.shape
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    return np.random.uniform(low=min, high=max, size=(k, d))


def kmeans(X, k, iterations=1000):
    """ Function that performs K-means on a dataset
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset
            n is the number of data points
            d is the number of dimensions for each data point
        k is a positive integer containing the number of clusters
        iterations is a positive integer containing the maximum number
            of iterations that should be performed
    Returns:
        C, clss, or None, None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    n, d = X.shape
    min = X.min(axis=0)
    max = X.max(axis=0)
    C = initialize(X, k)

    clss = None

    for i in range(iterations):
        # Euclidean distance
        distances = np.linalg.norm(X[:, None] - C, axis=-1)
        clss = np.argmin(distances, axis=-1)

        C_copy = np.copy(C)
        # Moving centroids
        for j in range(k):
            # Check if is [NaN]
            indices = np.argwhere(clss == j)
            if not len(indices):
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[indices], axis=0)
        if (C_copy == C).all():
            return C, clss

    distances = np.linalg.norm(X[:, None] - C, axis=-1)
    clss = np.argmin(distances, axis=-1)

    return C, clss
