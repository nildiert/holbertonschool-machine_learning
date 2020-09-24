#!/usr/bin/env python3
"""
Function that trains a model
using mini-batch gradient descent
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Method that trains a model using mini-batch gradient descent
    """
    callbacks = []
    if validation_data:
        early_stopping = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience)
        callbacks.append(early_stopping)
    history = network.fit(
        x=data,
        y=labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks)
    return history
