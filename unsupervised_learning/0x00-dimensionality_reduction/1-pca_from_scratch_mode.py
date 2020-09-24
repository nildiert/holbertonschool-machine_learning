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
    n, m = X.shape
    # Find the covariance matrix
    # https://youtu.be/ZqXnPcyIAL8?t=120
    C = np.dot(X.T, X) / (n - 1)

    # Eingevalues and eingevectors
    # https://youtu.be/ZqXnPcyIAL8?t=163
    eigen_vals, eigen_vecs = np.linalg.eig(C)

    # Ordered the columns of eigen_vals
    # https://youtu.be/ZqXnPcyIAL8?t=288
    indices = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[indices]
    eigen_vecs = eigen_vecs[:, indices]

    # Dimensionality to be maintain
    # https://youtu.be/ZqXnPcyIAL8?t=433
    eigen_vals = eigen_vals[:ndim]
    W = (-1) * eigen_vecs[:, :ndim]

    T = np.matmul(X_m, W)

    T = T.astype('float64')
    return T
