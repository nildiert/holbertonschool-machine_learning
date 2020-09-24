#!/usr/bin/env python3
"""
Function that shuffles the data points
in two matrices the same way
"""

import numpy as np


def shuffle_data(X, Y):
    """ Shuffles the data points in two matrices the same way """
    shuffled = np.random.permutation(len(X))
    return X[shuffled], Y[shuffled]
