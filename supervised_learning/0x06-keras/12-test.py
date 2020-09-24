#!/usr/bin/env python3
"""
Function to test model
"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Function that tests a neural network
    """
    test = network.evaluate(data, labels, verbose=verbose)
    return test
