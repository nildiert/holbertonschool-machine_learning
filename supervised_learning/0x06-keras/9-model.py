#!/usr/bin/env python3
"""
Methods to load and save models
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Method to saves an entire model
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Method to loads an entire model
    """
    model = K.models.load_model(filename)
    return model
