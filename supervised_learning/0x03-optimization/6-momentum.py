#!/usr/bin/env python3
"""
Function that creates the training operation for a neural network
"""

import tensorflow as tf

def create_momentum_op(loss, alpha, beta1):
    """
    This function creates the training
    operation for a neural network
    """
    test = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=alpha, momentum=beta1, use_locking=False, name='Momentum', use_nesterov=False
    )
    print(test)
