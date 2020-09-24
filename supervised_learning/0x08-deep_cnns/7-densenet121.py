#!/usr/bin/env python3
"""
DenseNet-121
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Function that builds the DenseNet-121 architecture
    as described in Densely Connected Convolutional Networks:
    """

    init = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))

    batch_1 = K.layers.BatchNormalization()(X)
    activation_1 = K.layers.Activation('relu')(batch_1)

    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=7,
                             strides=2,
                             padding='same',
                             kernel_initializer=init)(activation_1)
    max_pool = K.layers.MaxPooling2D(pool_size=[3, 3],
                                     strides=2,
                                     padding='same')(conv_1)

    ly_1, nb_filters1 = dense_block(max_pool, 64, growth_rate, 6)

    ly_2, nb_filters2 = transition_layer(ly_1, nb_filters1, compression)
    ly_3, nb_filters3 = dense_block(ly_2, nb_filters2, growth_rate, 12)

    ly_4, nb_filters4 = transition_layer(ly_3, nb_filters3, compression)
    ly_5, nb_filters5 = dense_block(ly_4, nb_filters4, growth_rate, 24)

    ly_6, nb_filters6 = transition_layer(ly_5, nb_filters5, compression)
    ly_7, nb_filters7 = dense_block(ly_6, nb_filters6, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7],
                                         strides=7,
                                         padding='same')(ly_7)

    dense = K.layers.Dense(1000, activation='softmax',
                           kernel_initializer=init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=dense)
    return model
