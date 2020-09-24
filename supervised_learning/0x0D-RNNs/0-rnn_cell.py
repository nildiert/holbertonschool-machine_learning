#!/usr/bin/env python3
"""
0x0D. RNNs
"""
import numpy as np


class RNNCell:
    """
        Class that represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """ Method to initialize the RNNCell class
        Args:
            i - is the dimensionality of the data
            h - is the dimensionality of the hidden state
            o - is the dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, z):
        """Compute softmax values for each sets of scores in x"""
        e_z = np.exp(z)
        return e_z / e_z.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ Method that performs forward propagation for one time step
        Args:
            x_t - is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
            m - is the batche size for the data
            h_prev - is a numpy.ndarray of shape (m, h) containing the
                previous hidden state
        Return
            h_next - is the next hidden state
            y - is the output of the cell
        """
        h_next = np.tanh(np.matmul(np.hstack((h_prev, x_t)), self.Wh)
                         + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
