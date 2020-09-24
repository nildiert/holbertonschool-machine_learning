#!/usr/bin/env python3
"""
6. Gradients
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """ Function that calculates the gradients of Y
    Args:
        Y - is a numpy.ndarray of shape (n, ndim) containing the low
            dimensional transformation of X
        P - is a numpy.ndarray of shape (n, n) containing the P
            affinities of X
    Returns: (dY, Q)
        dY - is a numpy.ndarray of shape (n, n) containing the gradients of Y
        Q - is a numpy.ndarray of shape (n, n) containing the Q affinities of Y
    """
    n, ndim = Y.shape

    Q, num = Q_affinities(Y)
    PQ = P - Q
    dY = np.zeros((n, ndim))
    for i in range(n):
        dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i],
                                  (ndim, 1)).T * (Y[i, :] - Y), 0)
    return dY, Q
