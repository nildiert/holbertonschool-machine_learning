#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function that updates the weights and biases
    of a neural network using gradient
    descent with L2 regularization
    """

    weights2 = weights.copy()
    m = Y.shape[1]

    for net_ly in reversed(range(L)):

        if net_ly == L-1:
            dz = cache["A"+str(net_ly+1)] - Y
            dw = (np.matmul(cache["A"+str(net_ly)], dz.T) / m).T
            dw_reg = dw + (lambtha/m) * weights2["W"+str(net_ly+1)]
            db = np.sum(dz, axis=1, keepdims=True) / m

        else:
            dz1 = np.matmul(weights2["W"+str(net_ly+2)].T, current_dz)
            dz2 = 1-cache["A"+str(net_ly+1)]**2
            dz = dz1 * dz2
            dw = np.matmul(dz, cache["A"+str(net_ly)].T) / m
            dw_reg = dw + (lambtha/m) * weights2["W"+str(net_ly+1)]
            db = np.sum(dz, axis=1, keepdims=True) / m

        weights["W"+str(net_ly+1)] = (weights2["W"+str(net_ly+1)] - (alpha *
                                                                     dw_reg))
        weights["b"+str(net_ly+1)] = weights2["b"+str(net_ly+1)] - alpha * db
        current_dz = dz
