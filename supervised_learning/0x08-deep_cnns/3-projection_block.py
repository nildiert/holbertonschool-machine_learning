#!/usr/bin/env python3
"""
Projection Block
"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    This function builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015):
    """

    F11, F3, F12 = filters

    initializer = K.initializers.he_normal(seed=None)

    conv2d = K.layers.Conv2D(filters=F11, kernel_size=1,
                             padding='same',
                             strides=s,
                             kernel_initializer=initializer)(A_prev)
    batch_normalization = K.layers.BatchNormalization(axis=-1)(conv2d)
    activation = K.layers.Activation('relu')(batch_normalization)
    conv2d_1 = K.layers.Conv2D(filters=F3, kernel_size=3,
                               padding='same',
                               kernel_initializer=initializer)(activation)
    batch_normalization_1 = K.layers.BatchNormalization(axis=-1)(conv2d_1)
    activation_1 = K.layers.Activation('relu')(batch_normalization_1)
    conv2d_2 = K.layers.Conv2D(filters=F12, kernel_size=1,
                               padding='same',
                               kernel_initializer=initializer)(activation_1)
    conv2d_3 = K.layers.Conv2D(filters=F12, kernel_size=1,
                               padding='same',
                               strides=s,
                               kernel_initializer=initializer)(A_prev)
    batch_normalization_2 = K.layers.BatchNormalization(axis=-1)(conv2d_2)
    batch_normalization_3 = K.layers.BatchNormalization(axis=-1)(conv2d_3)

    add = K.layers.Add()([batch_normalization_2, batch_normalization_3])
    activation_2 = K.layers.Activation('relu')(add)

    return activation_2
