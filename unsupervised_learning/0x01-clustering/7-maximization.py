#!/usr/bin/env python3
"""
7. Maximization
"""

import numpy as np


def maximization(X, g):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
        for each data point in each cluster
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the updated priors for
            each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated centroid
            means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
            covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    if not np.isclose(np.sum(g, axis=0), np.ones((n, ))).all():
        return None, None, None
    pi, m, s = np.zeros((k,)), np.zeros((k, d)), np.zeros((k, d, d))

    for i in range(k):
        pi[i] = np.sum(g[i]) / n
        m[i] = np.dot(g[i], X) / np.sum(g[i])
        X_m = X - m[i]
        s[i] = np.dot(g[i] * X_m.T, X_m) / np.sum(g[i])

    return pi, m, s
