#!/usr/bin/env python3
""" Method to converts a numeric label vector into a one-hot matrix """
import numpy as np


def one_hot_encode(Y, classes):
    """ Method to converts a numeric label vector into a one-hot matrix  """

    if not isinstance(Y, np.ndarray):
        return None
    if Y.size is 0:
        return None
    if type(classes) is not int:
        return None
    if classes < Y.max() + 1:
        return None

    data = Y
    one_hot = np.zeros((classes, Y.shape[0]))
    rows = np.arange(Y.shape[0])
    one_hot[data, rows] = 1
    return one_hot
