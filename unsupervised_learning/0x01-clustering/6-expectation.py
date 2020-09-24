#!/usr/bin/env python3
"""
6. Expectation
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ Function that calculates the expectation step in the EM
        algorithm for a GMM
    Args:
        X - is a numpy.ndarray of shape (n, d) containing the data set
        pi - is a numpy.ndarray of shape (k,) containing the priors
            for each cluster
        m - is a numpy.ndarray of shape (k, d) containing the centroid
            means for each cluster
        S - is a numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster
    Returns:
        Returns: g, l, or None, None on failure
            g is a numpy.ndarray of shape (k, n) containing the posterior
                probabilities for each data point in each cluster
            l is the total log likelihood
    Documentation:
        http://www.cse.iitm.ac.in/~vplab/courses/DVP/PDF/gmm.pdf
        E step
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if X.shape[1] != m.shape[1] or X.shape[1] != S.shape[1]:
        return None, None
    if X.shape[1] != S.shape[2]:
        return None, None
    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    try:
        n, d = X.shape
        k = pi.shape[0]

        gauss_p = np.zeros((k, n))

        for i in range(k):
            P = pdf(X, m[i], S[i])
            gauss_p[i] = P * pi[i]

        g = gauss_p / np.sum(gauss_p, axis=0)

        return g, np.sum(np.log(np.sum(gauss_p, axis=0)))
    except Exception:
        return None, None
