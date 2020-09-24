#!/usr/bin/env python3
"""
2. Absorbing Chains
"""

import numpy as np

def letter_range(start, stop="{", step=1):
    """Yield a range of lowercase letters."""
    for ord_ in range(ord(start.lower()), ord(stop.lower()), step):
        yield chr(ord_)

def absorbing(P):
    """ Function that determines if a markov chain is absorbing
    Args:
        P is a is a square 2D numpy.ndarray of shape (n, n)
            representing the transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    if (P == np.eye(P.shape[0])).all():
        return True
    if np.any(np.diag(P) == 1):
        for i, row in enumerate(P):
            for j, col in enumerate(row):
                if i == j and ((i + 1) < len(P)) and ((j + 1) < len(P)):
                    if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                        return False
        return True
    return False
