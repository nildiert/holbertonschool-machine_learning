#!/usr/bin/env python3
"""
Create a Layer with L2 Regularization
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    This function  creates a tensorflow layer that includes
    L2 regularization
    """
    layer_l2_reg = tf.contrib.layers.l2_regularizer(lambtha)
    kernel_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel_init,
                            kernel_regularizer=layer_l2_reg)
    return layer(prev)
