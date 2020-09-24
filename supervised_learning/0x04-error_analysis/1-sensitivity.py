#!/usr/bin/env python3
"""
Sensitivity
"""

import numpy as np


def sensitivity(confusion):
    """
    Function that that calculates the sensitivity
    for each class in a confusion matrix
    """
    positive = np.sum(confusion, axis=1)
    true = np.diagonal(confusion)
    sens = true / positive
    return (sens)
