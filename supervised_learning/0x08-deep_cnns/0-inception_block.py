#!/usr/bin/env python3
"""
Inception Block
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    This function builds an inception block as described
    in Going Deeper with Convolutions (2014):
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    initializer = K.initializers.he_normal(seed=None)

    conv2d_1 = K.layers.Conv2D(filters=F1, kernel_size=1,
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer)(A_prev)

    conv2d_2 = K.layers.Conv2D(filters=F3R, kernel_size=1,
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer)(A_prev)

    conv2d_4 = K.layers.Conv2D(filters=F3, kernel_size=3,
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer)(conv2d_2)

    conv2d_3 = K.layers.Conv2D(filters=F5R, kernel_size=1,
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer)(A_prev)

    max_pooling2d = K.layers.MaxPooling2D(pool_size=[3, 3],
                                          strides=1,
                                          padding='same')(A_prev)

    conv2d_5 = K.layers.Conv2D(filters=F5, kernel_size=5,
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer)(conv2d_3)

    conv2d_6 = K.layers.Conv2D(filters=FPP, kernel_size=1,
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer)(max_pooling2d)
    values = [conv2d_1, conv2d_4, conv2d_5, conv2d_6]
    concatenate = K.layers.concatenate([conv2d_1,
                                        conv2d_4,
                                        conv2d_5, conv2d_6])
    return concatenate
