#!/usr/bin/env python3
"""
2. Initialize Multinormal class
"""

import numpy as np


class MultiNormal:
    """
    Multinormal Class
    """

    def __init__(self, data):
        """ Constructor method
        Args:
            data - is a numpy.ndarray of shape (d, n) containing
                the data set:
                n is the number of data points
                d is the number of dimensions in each data point
        """

        if (not type(data) == np.ndarray) or (len(data.shape) != 2):
            raise TypeError("data must be a 2D numpy.ndarray")
        n = data.shape[1]
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean_cov(data)

    def mean_cov(self, X):
        """ Calculates the mean and covariance of a data set
        Args:
            X - is a numpy.ndarray of shape (n, d) containing the data set
                n: is the number of data points
                d: is the number of dimensions in each data point
        Returns:
            mean - is a numpy.ndarray of shape (1, d) containing the mean
                    of the data set
            cov - is a numpy.ndarray of shape (d, d) containing the covariance
                    matrix of the data set
        """
        d = X.shape[0]
        n = X.shape[1]
        self.mean = np.mean(X, axis=1).reshape(d, 1)
        X = X - self.mean
        self.cov = ((np.dot(X, X.T)) / (n - 1))

    def pdf(self, x):
        """ Calculates the PDF at a data point
        Args:
            x is a numpy.ndarray of shape (d, 1) containing the data point
            whose PDF should be calculated
        Returns:
            The value of the PDF
        """

        if (not type(x) == np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if (len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        x_m = x - self.mean
        pdf = (1 / (np.sqrt((2 * np.pi)**d * np.linalg.det(self.cov)))
               * np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2))

        return float(pdf)
