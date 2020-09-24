#!/usr/bin/env python3
"""
Inception Network
"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    This function builds the inception network
    as described in Going Deeper with Convolutions (2014):
    """

    initializer = K.initializers.he_normal(seed=None)

    input_1 = K.Input(shape=(224, 224, 3))
    conv2d = K.layers.Conv2D(filters=64, kernel_size=7,
                             padding='same',
                             activation='relu',
                             strides=2,
                             kernel_initializer=initializer)(input_1)

    max_pooling2d = K.layers.MaxPooling2D(pool_size=[3, 3],
                                          strides=2, padding='same')(conv2d)

    conv2d_1 = K.layers.Conv2D(filters=64, kernel_size=1,
                               padding='same',
                               activation='relu',
                               strides=1,
                               kernel_initializer=initializer)(
                                   max_pooling2d)
    conv2d_2 = K.layers.Conv2D(filters=192, kernel_size=3,
                               padding='same',
                               activation='relu',
                               strides=1,
                               kernel_initializer=initializer)(conv2d_1)
    max_pooling2d_1 = K.layers.MaxPooling2D(pool_size=[3, 3],
                                            strides=2,
                                            padding='same')(conv2d_2)

    concatenate = inception_block(max_pooling2d_1, [64, 96, 128, 16, 32, 32])
    concatenate_1 = inception_block(concatenate, [128, 128, 192, 32, 96, 64])
    max_pooling2d_4 = K.layers.MaxPooling2D(pool_size=[3, 3],
                                            strides=2,
                                            padding='same')(concatenate_1)
    concatenate_2 = inception_block(max_pooling2d_4,
                                    [192, 96, 208, 16, 48, 64])
    concatenate_3 = inception_block(concatenate_2, [160, 112, 224, 24, 64, 64])
    concatenate_4 = inception_block(concatenate_3, [128, 128, 256, 24, 64, 64])
    concatenate_5 = inception_block(concatenate_4, [112, 144, 288, 32, 64, 64])
    concatenate_6 = inception_block(concatenate_5,
                                    [256, 160, 320, 32, 128, 128])
    max_pooling2d_5 = K.layers.MaxPooling2D(pool_size=[3, 3],
                                            strides=2,
                                            padding='same')(concatenate_6)
    concatenate_7 = inception_block(max_pooling2d_5,
                                    [256, 160, 320, 32, 128, 128])
    concatenate_8 = inception_block(concatenate_7,
                                    [384, 192, 384, 48, 128, 128])
    average_pooling2d = K.layers.AveragePooling2D(pool_size=(7, 7), strides=1,
                                                  padding='valid',
                                                  data_format=None)(
                                                      concatenate_8)
    dropout = K.layers.Dropout(1-0.2)(average_pooling2d)
    dense = K.layers.Dense(1000, activation='relu',
                           kernel_initializer=initializer)(dropout)

    model = K.models.Model(inputs=input_1, outputs=dense)
    return model
