#!/usr/bin/env python3
"""
L2 Regularization Cost
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Function that calculates the cost of a neural
    network with L2 regularization
    """
    w_summatory = 0
    for i in range(1, L + 1):
        w_summatory += np.linalg.norm(weights['W'+str(i)])
    l2 = cost + ((lambtha/(2*m)) * w_summatory)
    return l2
