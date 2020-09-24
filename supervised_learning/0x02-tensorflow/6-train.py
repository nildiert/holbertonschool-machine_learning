#!/usr/bin/env python3
"""
Function that builds, trains, and saves a neural network classifier
"""

import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        layer_sizes,
        activations,
        alpha,
        iterations,
        save_path="/tmp/model.ckpt"):
    """ Train """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for iteration in range(iterations + 1):
            cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accu = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            cost_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            if iteration % 100 == 0 or iteration == iterations:
                print("After {} iterations:".format(iteration))
                print("\tTraining Cost: {}".format(cost))
                print("\tTraining Accuracy: {}".format(accu))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(acc_valid))
            if iteration < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        save_path = saver.save(sess, save_path)
    return save_path
