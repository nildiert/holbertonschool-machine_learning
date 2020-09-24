#!/usr/bin/env python3
"""
7. Bidirectional Output
"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """ Constructor method
        Args:
            i - is the dimensionality of the data
            h - is the dimensionality of the hidden states
            o - is the dimensionality of the outputs
        """
        self.Whf = np.random.randn(h+i, h)
        self.Whb = np.random.randn(h+i, h)
        self.Wy = np.random.randn(h * 2, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Method that calculates the hidden state in the forward
            direction for one time step
        Args:
            x_t - is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
            m - is the batch size for the data
            h_prev - is a numpy.ndarray of shape (m, h) containing the previous
                hidden state
        Returns:
            h_next - the next hidden state
        """
        h_next = np.tanh(np.matmul(np.hstack((h_prev, x_t)), self.Whf)
                         + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """ Method that calculates the hidden state in the backward
            direction for one time step
        x_t - is a numpy.ndarray of shape (m, i) that contains the data input
            for the cell
            * m is the batch size for the data
        h_next - is a numpy.ndarray of shape (m, h) containing the next hidden
            state

        """
        h_prev = np.tanh(np.matmul(np.hstack((h_next, x_t)), self.Whb)
                         + self.bhb)
        return h_prev

    def output(self, H):
        """ Function that calculates all outputs for the RNN:
        Args:
            H  -is a numpy.ndarray of shape (t, m, 2 * h) that contains the
                concatenated hidden states from both directions, excluding
                    their initialized states
                * t is the number of time steps
                * m is the batch size for the data
                * h is the dimensionality of the hidden states
        Returns: Y, the outputs
        """
        T, m, h2 = H.shape

        Y = []
        for t in range(T):
            y = self.softmax(np.matmul(H[t], self.Wy) + self.by)
            Y.append(y)
        return np.array(Y)
