#!/usr/bin/env python3
"""
1. Correlation
"""

import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix
    Args:
        C - is a numpy.ndarray of shape (d, d) containing a
        covariance matrix
        d is the number of dimensions
    Returns:
        a numpy.ndarray of shape (d, d) containing the
        correlation matrix
    Documentation:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or (C.shape[0] != C.shape[1]):
        raise ValueError("C must be a 2D square matrix")

    # Cij / sqrt(Cii * Cjj)
    # Equation in documentation of numpy.corrcoef
    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    correlation = C / outer_v
    return correlation
