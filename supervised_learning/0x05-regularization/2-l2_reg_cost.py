#!/usr/bin/env python3
"""
L2 Regularization Cost
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Function that calculates the cost of a neural network L2
    """
    return cost + tf.losses.get_regularization_losses()
