#!/usr/bin/env python3
"""
Early Stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Function that creates a layer of a neural network using dropout
    """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if (count == patience):
        boolean = True
    else:
        boolean = False
    return (boolean, count)
