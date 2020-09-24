#!/usr/bin/env python3
"""
LeNet-5 (Keras)
"""

import tensorflow.keras as K


def lenet5(X):
    """
    Function that builds a modified version of the LeNet-5
    architecture using keras
    """
    initializer = K.initializers.he_normal(seed=None)

    conv_layer1 = K.layers.Conv2D(filters=6, kernel_size=5,
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer=initializer)(X)
    pool_layer1 = K.layers.MaxPooling2D(
        pool_size=[2, 2],
        strides=2)(conv_layer1)

    conv_layer_2 = K.layers.Conv2D(filters=16, kernel_size=5,
                                   padding='valid',
                                   activation='relu',
                                   kernel_initializer=initializer)(pool_layer1)

    pool_layer2 = K.layers.MaxPooling2D(pool_size=[2, 2],
                                        strides=2)(conv_layer_2)

    flat_layer = K.layers.Flatten()(pool_layer2)

    fully_c_layer1 = K.layers.Dense(120, activation='relu',
                                    kernel_initializer=initializer)(flat_layer)

    fully_c_layer2 = K.layers.Dense(84, activation='relu',
                                    kernel_initializer=initializer)(
                                        fully_c_layer1)

    out_soft_layer = K.layers.Dense(10,
                                    activation='softmax',
                                    kernel_initializer=initializer)(
                                        fully_c_layer2)

    model = K.models.Model(X, out_soft_layer)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
