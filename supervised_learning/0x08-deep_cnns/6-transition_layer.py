#!/usr/bin/env python3
"""
Transition Layer
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Function that builds a transition layer as described in
    Densely Connected Convolutional Networks
    """

    init = K.initializers.he_normal(seed=None)

    filters_n = int(nb_filters*compression)

    batch_1 = K.layers.BatchNormalization()(X)
    activation_1 = K.layers.Activation('relu')(batch_1)
    conv_1 = K.layers.Conv2D(filters=filters_n,
                             kernel_size=1,
                             padding='same',
                             kernel_initializer=init)(activation_1)

    avg_pool = K.layers.AveragePooling2D(pool_size=[2, 2],
                                         strides=2,
                                         padding='same')(conv_1)

    return avg_pool, filters_n
