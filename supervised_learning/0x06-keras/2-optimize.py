#!/usr/bin/env python3
"""
Optimize model
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Method to optimize model
    """
    optimizer = K.optimizers.Adam(
        alpha, beta_1=beta1, beta_2=beta2
    )
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
