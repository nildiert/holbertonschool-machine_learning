#!/usr/bin/env python3
"""
This function updates a variable using the gradient
descent with momentum optimization algorithm
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Function that updates a variable
    using the gradient descent with
    momentum optimization algorithm
    """
    V_dw = (beta1 * v) + ((1 - beta1) * grad)
    W = var - (alpha * V_dw)
    return (W, V_dw)
