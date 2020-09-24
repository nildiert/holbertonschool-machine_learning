#!/usr/bin/env python3
"""
Dense Block
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Function that builds a dense block as described
    in Densely Connected Convolutional Networks
    """

    init = K.initializers.he_normal(seed=None)

    for layer in range(layers):

        batch_1 = K.layers.BatchNormalization()(X)
        activation_1 = K.layers.Activation('relu')(batch_1)

        conv_1 = K.layers.Conv2D(filters=128, kernel_size=1, padding='same',
                                 kernel_initializer=init)(activation_1)

        batch_2 = K.layers.BatchNormalization()(conv_1)
        activation_2 = K.layers.Activation('relu')(batch_2)
        conv_2 = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                 padding='same',
                                 kernel_initializer=init)(activation_2)

        concatenate = K.layers.concatenate([X, conv_2])

        X = concatenate
        nb_filters += growth_rate

    return concatenate, nb_filters
