#!/usr/bin/env python3
"""
1. PCA v2
"""

import numpy as np


def pca(X, ndim):
    """ Function that performs PCA on a dataset
    Args:
        X - is a numpy.ndarray of shape (n, d) where:
            n: is the number of data points
            d: is the number of dimensions in each point
        ndim - is the new dimensionality of the transformed X

    Returns:
        T, a numpy.ndarray of shape (n, ndim) containing the transformed
        version of X

    """
    # Normalize the matrix
    X_m = X - np.mean(X, axis=0)

    u, s, vh = np.linalg.svd(X_m)
    W = vh[:ndim].T

    T = np.matmul(X_m, W)
    return T
