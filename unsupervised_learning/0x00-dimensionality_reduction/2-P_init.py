#!/usr/bin/env python3
"""
2. Initialize t-SNE
"""

import numpy as np


def P_init(X, perplexity):
    """ Function that initializes all variables required to calculate
        the P affinities in t-SNE:
    Args:
        X - is a numpy.ndarray of shape (n, d) containing the dataset to
        be transformed by t-SNE
            n - is the number of data points
            d - is the number of dimensions in each point
        perplexity - is the perplexity that all Gaussian distributions
        should have
    Returns: (D, P, betas, H)
        D: a numpy.ndarray of shape (n, n) that calculates the pairwise
            distance between two data points
        P: a numpy.ndarray of shape (n, n) initialized to all 0‘s that will
            contain the P affinities
        betas: a numpy.ndarray of shape (n, 1) initialized to all 1’s that will
            contain all of the beta values
            beta_i = 1/(2*sigma_i^2)
        H is the Shannon entropy for perplexity perplexity
    """
    n, d = X.shape
    # (a-b)^2 = a^2 + b^2 - 2*a*b,
    # pairwise euclidean suqared distance of a matrix can be calculated as
    # follows:
    sum_X = np.sum(np.square(X), 1)
    
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    # This is the same
    # suqared_d = sum_X - 2*np.dot(X, X.T) + sum_X.T

    P = np.zeros([n, n], dtype='float64')
    betas = np.ones([n, 1], dtype='float64')
    H = np.log2(perplexity)

    return D, P, betas, H
