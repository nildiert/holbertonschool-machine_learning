#!/usr/bin/env python3
"""
ResNet-50
"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Functiont that builds the ResNet-50 architecture as
    described in Deep Residual Learning for Image Recognition (2015)
    """
    initializer = K.initializers.he_normal(seed=None)

    input_1 = K.Input(shape=(224, 224, 3))
    ly_1 = K.layers.Conv2D(filters=64, kernel_size=7,
                           padding='same',
                           strides=2,
                           kernel_initializer=initializer)(input_1)
    batch_normalization = K.layers.BatchNormalization()(ly_1)
    activation = K.layers.Activation('relu')(batch_normalization)
    max_pooling2d = K.layers.MaxPooling2D(pool_size=[3, 3], strides=2,
                                          padding='same')(activation)

    # x3
    ly_2 = projection_block(max_pooling2d, [64, 64, 256], 1)
    ly_3 = identity_block(ly_2, [64, 64, 256])
    ly_4 = identity_block(ly_3, [64, 64, 256])

    # x4
    ly_5 = projection_block(ly_4, [128, 128, 512])

    ly_6 = identity_block(ly_5, [128, 128, 512])
    ly_7 = identity_block(ly_6, [128, 128, 512])
    ly_8 = identity_block(ly_7, [128, 128, 512])

    # x6

    ly_9 = projection_block(ly_8, [256, 256, 1024])

    ly_10 = identity_block(ly_9, [256, 256, 1024])
    ly_11 = identity_block(ly_10, [256, 256, 1024])
    ly_12 = identity_block(ly_11, [256, 256, 1024])
    ly_13 = identity_block(ly_12, [256, 256, 1024])
    ly_14 = identity_block(ly_13, [256, 256, 1024])

    # x3

    ly_15 = projection_block(ly_14, [512, 512, 2048])
    ly_16 = identity_block(ly_15, [512, 512, 2048])
    ly_17 = identity_block(ly_16, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7], strides=7,
                                         padding='same')(ly_17)

    dense = K.layers.Dense(1000, activation='softmax',
                           kernel_initializer=initializer)(avg_pool)

    X = input_1
    Y = dense
    model = K.models.Model(inputs=X, outputs=Y)
    return model
