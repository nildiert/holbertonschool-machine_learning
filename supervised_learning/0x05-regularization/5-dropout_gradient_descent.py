#!/usr/bin/env python3
"""
Gradient Descent with Dropout
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Function that updates the weights of a
    neural network with Dropout regularization
    using gradient descent:
    """
    weights2 = weights.copy()
    _, m = Y.shape

    for neural_lyr in reversed(range(L)):
        if neural_lyr == L-1:
            dz = cache["A"+str(neural_lyr+1)] - Y
            dw = (np.matmul(cache["A"+str(neural_lyr)], dz.T) / m).T
            db = np.sum(dz, axis=1, keepdims=True) / m
        else:
            dz1 = np.matmul(weights2["W"+str(neural_lyr+2)].T, current_dz)
            dz2 = 1-cache["A"+str(neural_lyr+1)]**2
            dz = dz1 * dz2
            dz *= cache['D'+str(neural_lyr+1)]
            dz /= keep_prob
            dw = np.matmul(dz, cache["A"+str(neural_lyr)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
        weights["W"+str(neural_lyr+1)] = (weights2["W"+str(neural_lyr+1)] - (
            alpha * dw))
        weights["b"+str(neural_lyr+1)] = weights2["b"+str(
            neural_lyr+1)] - alpha * db
        current_dz = dz
