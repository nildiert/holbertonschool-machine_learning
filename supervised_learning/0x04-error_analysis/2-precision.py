#!/usr/bin/env python3
"""
Precision
"""

import numpy as np


def precision(confusion):
    """
    Function that calculates the precision
    for each class in a confusion matrix
    """
    pos_pred_cond = np.sum(confusion, axis=0)
    post_true = np.diagonal(confusion)
    pres = post_true / pos_pred_cond
    return (pres)
