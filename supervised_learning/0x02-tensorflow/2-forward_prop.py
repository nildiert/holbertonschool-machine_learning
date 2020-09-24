#!/usr/bin/env python3
"""
This function creates the forward propagation graph for the neural network
"""

import tensorflow as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Forward propagation """

    new_layer = create_layer(x, layer_sizes[0], activations[0])
    count = 1
    for count in range(1, len(layer_sizes)):
        new_layer = create_layer(
            new_layer,
            layer_sizes[count],
            activations[count])
    return new_layer
