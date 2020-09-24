#!/usr/bin/env python3
"""
This function calculates the normalization (standardization)
constants of a matrix
"""

import numpy as np


def normalization_constants(X):
    """
    Function that calculates the normalization
    constant of a matrix
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
