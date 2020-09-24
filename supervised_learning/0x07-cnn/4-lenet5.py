#!/usr/bin/env python3
"""
LeNet-5
"""

import tensorflow as tf


def lenet5(x, y):
    """
    Function that builds a modified version of
    the LeNet-5 architecture using tensorflow
    """

    initializer = tf.contrib.layers.variance_scaling_initializer()
    conv_layer1 = tf.layers.Conv2D(filters=6,
                                   kernel_size=5,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   kernel_initializer=initializer)(x)
    pool_layer1 = tf.layers.MaxPooling2D(pool_size=2,
                                         padding='valid',
                                         strides=2)(conv_layer1)
    conv_layer2 = tf.layers.Conv2D(filters=16,
                                   kernel_size=5,
                                   strides=1,
                                   padding='valid',
                                   activation=tf.nn.relu,
                                   kernel_initializer=initializer)(pool_layer1)
    pool_layer2 = tf.layers.MaxPooling2D(pool_size=2,
                                         strides=2,
                                         padding='valid')(conv_layer2)
    flat_layer = tf.layers.Flatten()(pool_layer2)
    fully_c_layer1 = tf.layers.Dense(units=120,
                                     activation=tf.nn.relu,
                                     kernel_initializer=initializer)(
                                         flat_layer)

    fully_c_layer2 = tf.layers.Dense(units=84,
                                     activation=tf.nn.relu,
                                     kernel_initializer=initializer)(
                                         fully_c_layer1)
    out_soft_layer = tf.layers.Dense(units=10,
                                     kernel_initializer=initializer)(
                                         fully_c_layer2)
    out_softmax = tf.nn.softmax(out_soft_layer)
    loss = tf.losses.softmax_cross_entropy(y, out_soft_layer)
    train = tf.train.AdamOptimizer().minimize(loss)

    val = tf.argmax(y, 1)
    pred = tf.argmax(out_soft_layer, 1)
    eq = tf.equal(pred, val)
    acc = tf.reduce_mean(tf.cast(eq, tf.float32))

    return (out_softmax, train, loss, acc)
