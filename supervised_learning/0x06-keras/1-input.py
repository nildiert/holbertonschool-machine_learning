#!/usr/bin/env python3
"""
Build a Input model
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Build Input model """

    regularizers = K.regularizers.l2(lambtha)
    inputs = K.layers.Input(shape=(nx,))
    output = K.layers.Dense(layers[0], activation=activations[0],
                            kernel_regularizer=regularizers)(inputs)
    for i in range(1, len(layers)):
        dropout = K.layers.Dropout(1-keep_prob)(output)
        output = K.layers.Dense(layers[i], activation=activations[i],
                                kernel_regularizer=regularizers)(dropout)
    model = K.models.Model(inputs=inputs, outputs=output)
    return model
