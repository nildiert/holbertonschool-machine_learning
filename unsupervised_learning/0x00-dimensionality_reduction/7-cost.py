#!/usr/bin/env python3
"""
7. Cost
"""

import numpy as np


def cost(P, Q):
    """ Function that calculates the cost of the t-SNE transformation
    Args:
        P is a numpy.ndarray of shape (n, n) containing the P affinities
        Q is a numpy.ndarray of shape (n, n) containing the Q affinities
    Returns:
        C, the cost of the transformation
    """
    # Hint: Watch out for division by 0 errors! Take the minimum of all
    # values, and almost 0 (ex. 1e-12)
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    return np.sum(P * np.log(P / Q))
