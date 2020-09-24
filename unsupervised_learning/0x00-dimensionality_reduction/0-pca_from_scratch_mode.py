#!/usr/bin/env python3
"""
0. PCA
"""

import numpy as np


def pca(X, var=0.96):
    """ Function that performs PCA on a dataset
    Args:
        X - is a numpy.ndarray of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
        var - is the fraction of the variance that the PCA transformation
            should maintain
    Returns:
        The weights matrix, W, that maintains var fraction of X‘s original
        variance
    """
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

    # Cumulative sum
    # https://youtu.be/NUn6WeFM5cM?t=628
    sum_eigen_vals = np.sum(eigen_vals)

    # retention of the information per eigen value
    variance_retention = eigen_vals / sum_eigen_vals
    acum_variance = np.cumsum(variance_retention)

    r = 0
    for i in acum_variance:
        r += 1
        if i > var:
            break
    # Threshold to keep enough singular values
    # https://youtu.be/NUn6WeFM5cM?t=867
    eigen_vecs = (-1) * eigen_vecs[:, :r]
    return eigen_vecs
