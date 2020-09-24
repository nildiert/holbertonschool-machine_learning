#!/usr/bin/env python3
"""
2. Update Gaussian Process
"""


import numpy as np


class GaussianProcess:
    """
        Class that represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Parameters:
            X_init - is a numpy.ndarray of shape (t, 1) representing
                the inputs already sampled with the black-box function
            Y_init - is a numpy.ndarray of shape (t, 1) representing
                the outputs of the black-box function for each input in X_init
            t - is the number of initial samples
            l - is the length parameter for the kernel
            sigma_f - is the standard deviation given to the output of the
                black-box function
        Documentation:
            http://krasserm.github.io/2018/03/19/gaussian-processes/
        """
        self.X = X_init
        self.Y = Y_init

        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Parameters:
            X1 - is a numpy.ndarray of shape (m, 1)
            X2 - is a numpy.ndarray of shape (n, 1)
            the kernel should use the Radial Basis Function (RBF)
        Returns:
            The covariance kernel matrix as a numpy.ndarray of shape (m, n)
        Documentation:
            http://krasserm.github.io/2018/03/19/gaussian-processes/ Eq 6

        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + \
            np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        Parameters:
            X_s - is a numpy.ndarray of shape (s, 1) containing all of the
                points whose mean and standard deviation should be calculated
            s - is the number of sample points
        Returns: mu, sigma
            mu is a numpy.ndarray of shape (s,) containing the mean for each
                point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,) containing the standard
                deviation for each point in X_s, respectively
        Documentation:
            http://krasserm.github.io/2018/03/19/gaussian-processes/
            Equations 4, 5
        """
        K_inv = np.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        mu = K_s.T.dot(K_inv).dot(self.Y)
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu.T[0], np.diagonal(cov)

    def update(self, X_new, Y_new):
        """Method that updates a Gaussian Process
            Parameters:
                X_new - is a numpy.ndarray of shape (1,) that represents
                    the new sample point
                Y_new - is a numpy.ndarray of shape (1,) that represents
                    the new sample function value
                Updates - the public instance attributes X, Y, and K
        """
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
