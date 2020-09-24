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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None

    n, d = X.shape
    min = X.min(axis=0)
    max = X.max(axis=0)
    initialized = np.random.uniform(min, max, size=(k, d))
    return initialized
