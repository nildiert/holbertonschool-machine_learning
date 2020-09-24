#!/usr/bin/env python3
"""
2. Variance
"""

import numpy as np


def variance(X, C):
    """ Function that calculates the total intra-cluster variance for a
        data set
    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing the centroid
            means for each cluster
    Documentation:
        https://www.youtube.com/watch?v=_aWzGGNrcic
        https://youtu.be/xNfOheh-res?t=140
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(C, np.ndarray) or len(C.shape) != 2:
            return None
        n, d = X.shape
        # Calculate the distance
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        # Calculate min distance between points and centroid
        min_dist = np.min(distances, axis=0)
        varian = np.sum(min_dist ** 2)
        return varian
    except Exception:
        return None
