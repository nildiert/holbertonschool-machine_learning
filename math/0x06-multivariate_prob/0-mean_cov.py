#!/usr/bin/env python3
"""
0. Mean and Covariance
"""

import numpy as np


def mean_cov(X):
    """ Calculates the mean and covariance of a data set
    Args:
        X - is a numpy.ndarray of shape (n, d) containing the data set
            n: is the number of data points
            d: is the number of dimensions in each data point
    Returns:
        mean - is a numpy.ndarray of shape (1, d) containing the mean
                of the data set
        cov - is a numpy.ndarray of shape (d, d) containing the covariance
                matrix of the data set
    """
    if (not type(X) == np.ndarray) or (len(X.shape) != 2):
        raise TypeError('X must be a 2D numpy.ndarray')
    n = X.shape[0]
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0).reshape(1, X.shape[1])
    X = X - mean
    cov = ((np.dot(X.T, X)) / (n - 1))
    return mean, cov
