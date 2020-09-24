#!/usr/bin/env python3
"""
Function that trains a model
using mini-batch gradient descent
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):

    """
    Method that trains a model using mini-batch gradient descent
    """

    def learningRate(epoch):
        """
        Learning rate each epoch
        """
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if validation_data:
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience)
            callbacks.append(early_stop)

        if learning_rate_decay:
            learning_rate_scheduler = K.callbacks.LearningRateScheduler(
                learningRate, verbose=1)
            callbacks.append(learning_rate_scheduler)
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
