#!/usr/bin/env python3
"""
"""

import numpy as np

def variance(X, C):
    """
    Variance
    """
    n, d = X.shape
    k, _ = C.shape

    return np.var(X, axis=0)
    print(X.shape)
    print(C.shape)