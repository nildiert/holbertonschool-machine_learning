#!/usr/bin/env python3
"""
1. Regular Chains
"""

import numpy as np


def regular(P):
    """ Function that determines the steady state probabilities of a
        regular markov chain
    Args:
        P is a is a square 2D numpy.ndarray of shape (n, n) representing
            the transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray):
        return None
    if P.shape[0] != P.shape[1]:
        return None
    for row in P:
        if not np.isclose(np.sum(row), 1):
            return None
    if np.all(P <= 0):
        return None
    try:
        n, _ = P.shape
        # Create S
        s = np.ones((1, n)) / n

        Pj = P.copy()
        while True:
            s_prev = s
            s = np.matmul(s, P)
            Pj = P * Pj
            if np.any(Pj <= 0):
                return None
            if np.all(s_prev == s):
                break
        return s
    except Exception:
        return None
