#!/usr/bin/env python3
"""
Specificity
"""

import numpy as np


def specificity(confusion):
    """
    Function that calculates the specificity
    for each class in a confusion matrix
    """
    true = np.diagonal(confusion)
    m_total = np.sum(confusion)
    array_m_total = np.full_like(confusion[0], m_total)
    cross_1 = np.sum(confusion, axis=0)
    cross_2 = np.sum(confusion.T, axis=0)
    true_neg = array_m_total + true - cross_1 - cross_2
    fp = cross1 - true
    spec = true_neg / (fp + true_neg)
    return spec
