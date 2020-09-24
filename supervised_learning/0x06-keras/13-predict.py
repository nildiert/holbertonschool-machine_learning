#!/usr/bin/env python3
"""
Function that makes a prediction using a neural network
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Method that makes a prediction
    using a neural network
    """
    prediction = network.predict(data, verbose=verbose)
    return prediction
