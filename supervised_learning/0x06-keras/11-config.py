#!/usr/bin/env python3
"""
Methods to load and save models
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Method to saves an entire model
    """
    json_model = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_model)
    return None


def load_config(filename):
    """
    Method to loads an entire model
    """
    with open(filename, 'r') as json_file:
        json_model = K.models.model_from_json(json_file.read())
    return json_model
