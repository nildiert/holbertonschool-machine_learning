#!/usr/bin/env python3
""" Creates the training operation for the network """

import tensorflow as tf


def create_train_op(loss, alpha):
    """ Creates a train operation """
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return optimizer
