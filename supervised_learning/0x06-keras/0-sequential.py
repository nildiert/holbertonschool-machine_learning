#!/usr/bin/env python3
"""
Sequential model
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Build Sequential model """

    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:

            model.add(K.layers.Dense(layers[i], input_shape=(nx,),
                                     activation=activations[i],
                                     kernel_regularizer=regularizer,
                                     name='dense'))
        else:
            model.add(K.layers.Dropout(1-keep_prob))
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer,
                                     name='dense_' + str(i)))

    return model
