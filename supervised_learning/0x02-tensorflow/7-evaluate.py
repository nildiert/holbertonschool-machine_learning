#!/usr/bin/env python3
""" Function that evaluates the output of a neural network """

import tensorflow as tf


def evaluate(X, Y, save_path):
    """ Method that evaluates the output of a neural network """

    with tf.Session() as session:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(session, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]

        feed_dict = {x: X, y: Y}
        fp = session.run(y_pred, feed_dict)
        ac = session.run(accuracy, feed_dict)
        lss = session.run(loss, feed_dict)
    return (fp, ac, lss)
