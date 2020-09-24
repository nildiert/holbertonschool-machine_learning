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

    if not np.isclose(np.sum(g, axis=0), np.ones((n, ))).all():
        return None, None, None
    k, _ = g.shape
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        # Calc numerator m
        m_numerator = np.sum(g[i, :, np.newaxis] * X, axis=0)
        # Calc denominator m
        m_denominator = np.sum(g[i], axis=0)
        m[i] = m_numerator / m_denominator
        s_numerator = np.dot(g[i] * (X - m[i]).T, X - m[i])
        S[i] = s_numerator / np.sum(g[i])
        pi[i] = np.sum(g[i]) / n
    return pi, m, S
