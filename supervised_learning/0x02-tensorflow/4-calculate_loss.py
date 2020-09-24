#!/usr/bin/env python3
"""  Method that calculates the softmax cross-entropy loss of a prediction """

import tensorflow as tf


def calculate_loss(y, y_pred):
    """ Calculate loss """
    loss = tf.losses.softmax_cross_entropy(
        y, y_pred, reduction=tf.losses.Reduction.MEAN
    )
    return loss
