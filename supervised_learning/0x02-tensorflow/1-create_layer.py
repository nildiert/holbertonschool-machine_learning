#!/usr/bin/env python3
""" Create layer function """

import tensorflow as tf


def create_layer(prev, n, activation):
    """Create layer method"""
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    layer = tf.layers.Dense(
        units=n,
        name='layer',
        activation=activation,
        kernel_initializer=kernel_initializer
    )

    return layer(prev)
