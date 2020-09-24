#!/usr/bin/env python3
"""
Methods to load and save models
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Method to saves an entire model
    """
    network.save_weights(filename)
    return None


def load_weights(network, filename):
    """
    Method to loads an entire model
    """
    network.load_weights(filename)
    return None
