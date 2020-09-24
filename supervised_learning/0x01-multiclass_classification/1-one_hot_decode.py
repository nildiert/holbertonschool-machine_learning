#!/usr/bin/env python3
""" This method converts a one-hot matrix into a vector of labels """

import numpy as np


def one_hot_decode(one_hot):
    """
    This method converts a one-hot matrix into a vector of labels
    """
    if not isinstance(one_hot, np.ndarray):
        return None

    copy = one_hot.T
    decode_array = np.zeros(one_hot.shape[1], dtype=int)
    for index_col, column in enumerate(copy):
        if sum(column) > 1:
            return None
        for index_row, row in enumerate(column):
            if row > 1:
                return None
            if row != 0 and row != 1:
                return None
            if row == 1:
                decode_array[index_col] = index_row
    return decode_array
